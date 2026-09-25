"""
G11.2.4 — PPO rollout adapter.

WHY THIS MODULE EXISTS
----------------------
``PPOTrainer.train_step()`` was written for a flat-tensor PPO environment: it
indexes ``states[batch_idx]`` and calls ``policy.forward(states, {})``. The
encoder-decoder policy does not expose a flat action tensor. Its on-policy
action is an *autoregressive token sequence* whose probability must be
re-evaluated through ``policy.evaluate_actions()``, which additionally needs
the per-timestep fleet snapshot and the raw sampled token indices.

The previous workaround was to re-implement PPO inline in
``LinerShippingTrainer.perform_ppo_update()`` as a single clipped full-batch
step, which made ``ppo_epochs``, ``minibatch_size`` and ``target_kl``
decorative. This module supplies the *small* compatibility layer that lets the
real epoch/minibatch/KL loop drive the encoder-decoder path.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not flatten, pad, encode or reconstruct the decoder action into a
tensor. It only *carries* the stored raw-action payload from rollout to
re-evaluation.

RAW ACTION != EXECUTED ACTION
-----------------------------
``rollout`` stores, per step:
  * ``decoded_port_sequence``    — the order the decoder SELECTED
  * ``executed_port_sequence``   — the order after P6 TSP / fallback repair
  * ``old_log_prob``             — log P(decoded sequence | state)

``build_evaluate_fn`` re-evaluates ``substep_selected`` (the raw token
sequence). ``executed_action`` is never used as the PPO action. This is the
single most important invariant in this file; ``test_g11_2_4_ppo_wiring.py``
pins it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# Evaluator returns (new_log_probs, new_entropies, new_values), all shape (B,).
EvaluatorFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class RolloutBatch:
    """
    Stored rollout, packaged for the PPO epoch/minibatch loop.

    Holds the *raw on-policy action payload* for every step alongside the
    scalars the loss needs. Construction is pure bookkeeping: nothing here
    runs the policy, mutates a mask, repairs a fallback, or touches the
    environment.
    """

    def __init__(
        self,
        policy_type: str,
        payloads: Sequence[Dict[str, Any]],
        critic_inputs: Sequence[torch.Tensor],
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        stored_entropies: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.policy_type = policy_type
        self.payloads = list(payloads)
        self.critic_inputs = list(critic_inputs)
        self.old_log_probs = old_log_probs
        self.old_values = old_values
        # Stored entropies are DIAGNOSTIC ONLY. The entropy term in the loss
        # uses the *current* policy's entropy, recomputed by evaluate_actions().
        # Feeding stored entropies into the loss is what previously made the
        # entropy bonus contribute exactly zero gradient.
        self.stored_entropies = stored_entropies
        self.rewards = rewards
        self.dones = dones

    def __len__(self) -> int:
        return len(self.payloads)

    @property
    def n_samples(self) -> int:
        return len(self.payloads)

    def minibatch_size(self, configured: int) -> int:
        """Configured minibatch size, clamped to at least one sample."""
        return max(1, min(int(configured), self.n_samples))

    def n_minibatches(self, configured: int) -> int:
        """Number of minibatches per epoch for a given configured size."""
        size = self.minibatch_size(configured)
        return (self.n_samples + size - 1) // size

    # ---- per-step re-evaluation ----

    def _evaluate_one(self, idx: int, policy: Any, critic: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate ONE stored step under the current policy and critic."""
        payload = self.payloads[idx]
        bundle = payload["state"]
        fleet_snapshot = payload["fleet_remaining"]

        with torch.enable_grad():
            if self.policy_type == "encoder_decoder":
                new_lp, new_entropy = policy.evaluate_actions(
                    bundle, fleet_snapshot,
                    substep_selected=payload["substep_selected"],
                    n_substeps=payload["n_substeps"],
                )
            elif self.policy_type == "encoder_only":
                new_lp, new_entropy = policy.evaluate_actions(
                    bundle, fleet_snapshot,
                    selected_mask=payload["selected_mask"],
                )
            else:
                raise ValueError(f"Unknown policy type: {self.policy_type}")

            # G10.2: the critic forward MUST stay in the autograd graph so the
            # value loss reaches critic.parameters().
            new_value = critic(self.critic_inputs[idx]).squeeze(-1)

        return new_lp.squeeze(), new_entropy.squeeze(), new_value.squeeze()

    def build_evaluate_fn(self, policy: Any, critic: Any) -> EvaluatorFn:
        """
        Build the (batch_idx) -> (log_probs, entropies, values) callback that
        ``PPOTrainer.train_step()`` calls once per minibatch per epoch.

        The callback re-runs ``policy.evaluate_actions()`` from scratch on every
        call, so each minibatch in each epoch sees a *fresh* computational graph
        and a freshly-updated policy.
        """

        def _evaluate(batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            lps: List[torch.Tensor] = []
            ents: List[torch.Tensor] = []
            vals: List[torch.Tensor] = []
            for raw_idx in batch_idx.tolist():
                lp, ent, val = self._evaluate_one(int(raw_idx), policy, critic)
                lps.append(lp)
                ents.append(ent)
                vals.append(val)
            return torch.stack(lps), torch.stack(ents), torch.stack(vals)

        return _evaluate

    # ---- diagnostics ----

    def measure_ratios(
        self,
        policy: Any,
        critic: Any,
        clip_epsilon: float,
    ) -> Dict[str, float]:
        """
        Ratio statistics on the CURRENT policy WITHOUT optimizing anything.

        Used to establish the PPO ratio baseline (TEST 6): on an unchanged
        policy the ratio must be ~1, because new_log_prob is then the same
        quantity as old_log_prob.
        """
        lps: List[torch.Tensor] = []
        for i in range(self.n_samples):
            lp, _, _ = self._evaluate_one(i, policy, critic)
            lps.append(lp.detach())
        new_lp = torch.stack(lps)
        ratios = torch.exp(new_lp - self.old_log_probs)
        lo, hi = 1.0 - clip_epsilon, 1.0 + clip_epsilon
        return {
            "old_log_prob_mean": float(self.old_log_probs.mean()),
            "new_log_prob_mean": float(new_lp.mean()),
            "ratio_mean": float(ratios.mean()),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "ratio_outside_clip_fraction": float(((ratios < lo) | (ratios > hi)).float().mean()),
        }


def build_rollout_batch(
    trajectory: Sequence[Dict[str, Any]],
    policy_type: str,
    instance: Any,
    device: torch.device,
) -> RolloutBatch:
    """
    Convert a collected trajectory into a :class:`RolloutBatch`.

    This is the ONLY place the rollout dict layout is interpreted. It reads:
      ``state``, ``critic_input``, ``fleet_remaining``, ``old_log_prob``,
      ``old_value``, ``entropy``, ``reward``, ``done``, and the RAW action
      payload (``substep_selected`` for encoder-decoder, ``selected_mask`` for
      encoder-only).

    It deliberately does NOT read ``executed_action``,
    ``executed_port_sequence`` or the fallback-repaired port list for the
    purpose of computing the PPO action.
    """
    payloads: List[Dict[str, Any]] = []
    critic_inputs: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    old_values: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    dones: List[float] = []

    port_codes_list = sorted(instance.ports.keys())
    n_ports = len(port_codes_list)

    for t in trajectory:
        bundle = t["state"]
        fleet_snapshot = t.get("fleet_remaining", {}).copy()

        payload: Dict[str, Any] = {
            "state": bundle,
            "fleet_remaining": fleet_snapshot,
        }

        if policy_type == "encoder_decoder":
            substep_selected = list(t.get("substep_selected", []))
            payload["substep_selected"] = substep_selected
            payload["n_substeps"] = len(substep_selected)
        elif policy_type == "encoder_only":
            # Prefer the exact tensor captured at rollout time (G9 F1 fix);
            # fall back to rebuilding from raw_sampled_ports.
            mask = t.get("selected_mask")
            if mask is None:
                mask = torch.zeros(n_ports, dtype=torch.bool, device=bundle.device)
                for pc in t.get("raw_sampled_ports", []):
                    if pc in port_codes_list:
                        mask[port_codes_list.index(pc)] = True
            payload["selected_mask"] = mask
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        payloads.append(payload)
        critic_inputs.append(t["critic_input"].to(device))
        old_log_probs.append(t["old_log_prob"].to(device))
        old_values.append(t["old_value"].to(device))
        entropies.append(t["entropy"].to(device))
        rewards.append(float(t["reward"]))
        dones.append(1.0 if t["done"] else 0.0)

    return RolloutBatch(
        policy_type=policy_type,
        payloads=payloads,
        critic_inputs=critic_inputs,
        old_log_probs=torch.stack(old_log_probs),
        old_values=torch.stack(old_values),
        stored_entropies=torch.stack(entropies),
        rewards=torch.tensor(rewards, device=device),
        dones=torch.tensor(dones, device=device),
    )
