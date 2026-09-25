"""
PPO Trainer — Core training engine.

Implements:
  - GAE / TD(lambda) advantage estimation
  - PPO policy loss with clipping
  - Value function loss
  - Entropy bonus
  - KL diagnostics
  - Minibatch sampling
  - Checkpointing

All math is implemented from scratch (no external RL library dependency)
to ensure transparency and reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PDiagnostics:
    """Diagnostics for a single PPO update."""
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    gradient_norm: float
    advantage_mean: float
    advantage_std: float
    value_mean: float
    value_std: float
    # ---- G11.2.4 execution evidence ----
    # Every field below is ``None``/0 for callers that do not go through the
    # instrumented multi-epoch path, so the dataclass stays backward
    # compatible with every existing construction site.
    epochs_configured: int = 0
    epochs_executed: int = 0
    minibatch_size: int = 0
    minibatches_per_epoch: int = 0
    total_optimizer_steps: int = 0
    kl_early_stopped: bool = False
    applied_grad_norm: float = 0.0
    policy_param_delta: float = 0.0
    critic_param_delta: float = 0.0
    old_log_prob_mean: float = 0.0
    new_log_prob_mean: float = 0.0
    ratio_mean: float = 0.0
    ratio_min: float = 0.0
    ratio_max: float = 0.0
    ratio_outside_clip_fraction: float = 0.0
    epoch_trace: List[Dict[str, float]] = field(default_factory=list)


class PPOTrainer:
    """
    PPO training engine for LSNDP policies.

    Parameters
    ----------
    policy : nn.Module
        The policy network (P8 or P9). Must have forward(), log_prob(), entropy().
    critic : nn.Module
        Value function network. Must accept same state_repr as policy.
    config : PPOConfig
        Training configuration.
    """

    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config: Any,
    ) -> None:
        self.policy = policy
        self.critic = critic
        self.config = config

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Training metrics
        self._update_count: int = 0
        self._total_env_steps: int = 0

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                list(self.policy.parameters()) + list(self.critic.parameters()),
                lr=self.config.learning_rate,
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                list(self.policy.parameters()) + list(self.critic.parameters()),
                lr=self.config.learning_rate,
                **self.config.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def compute_returns_and_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncated: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discounted returns and GAE advantages.

        Parameters
        ----------
        values : Tensor, shape (T,)
            Value estimates V(s_t) for each timestep.
        rewards : Tensor, shape (T,)
            Rewards r_t received at each timestep.
        dones : Tensor, shape (T,)
            Boolean mask for episode termination.
        truncated : Tensor, shape (T,), optional
            Boolean mask for truncation.

        Returns
        -------
        returns : Tensor, shape (T,)
            Discounted returns G_t.
        advantages : Tensor, shape (T,)
            GAE advantages A_t.
        """
        T = rewards.size(0)
        last_value = values[-1]

        # Initialize tensors
        advantages = torch.zeros(T, device=values.device)
        returns = torch.zeros(T, device=values.device)

        gae = 0.0
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t].float()) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t].float()) * gae

            advantages[t] = gae
            returns[t] = gae + values[t]

        return returns, advantages

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute PPO clipped surrogate loss.

        ratio = exp(new_log_prob - old_log_prob)
        clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
        loss = -mean(min(ratio * adv, clipped_ratio * adv))

        Parameters
        ----------
        log_probs : Tensor, shape (batch,)
            Current policy log probabilities.
        old_log_probs : Tensor, shape (batch,)
            Old policy log probabilities (from buffer).
        advantages : Tensor, shape (batch,)
            GAE advantages.
        clip_epsilon : float
            PPO clip parameter.

        Returns
        -------
        loss : Tensor, scalar
            Policy loss (negative, to be minimized).
        clip_fraction : float
            Fraction of samples where clipping occurred.
        approx_kl : float
            Approximate KL divergence between old and new policies.
        """
        # Policy ratio
        ratios = torch.exp(log_probs - old_log_probs)

        # Clipped ratios
        clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        # Surrogate losses
        surr1 = ratios * advantages
        surr2 = clipped_ratios * advantages

        # Take minimum (conservative PPO)
        loss = -torch.min(surr1, surr2).mean()

        # Clip fraction
        with torch.no_grad():
            clip_fraction = (torch.abs(ratios - clipped_ratios) > 1e-8).float().mean().item()

        # Approximate KL (using Taylor expansion: KL ≈ 0.5 * (ratio - 1)^2)
        with torch.no_grad():
            approx_kl = 0.5 * ((ratios - 1.0) ** 2).mean().item()

        return loss, clip_fraction, approx_kl

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value function loss (MSE between predicted and target values).

        Parameters
        ----------
        values : Tensor, shape (batch,)
            Current value predictions.
        returns : Tensor, shape (batch,)
            Target values (computed returns).

        Returns
        -------
        value_loss : Tensor, scalar
        """
        return nn.MSELoss()(values, returns)

    def compute_entropy_bonus(
        self,
        entropies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy bonus (to encourage exploration).

        Parameters
        ----------
        entropies : Tensor, shape (batch,)
            CURRENT-policy entropies, differentiable. These must come from a
            fresh forward pass of the policy under evaluation — not from the
            stored rollout entropies, which are detached scalars and would make
            the entropy term contribute exactly zero gradient.

        Returns
        -------
        entropy_loss : Tensor, scalar
            Negative entropy (to be subtracted from loss).
        """
        return -self.config.entropy_coefficient * entropies.mean()

    def _param_snapshot(self, module: Any) -> List[torch.Tensor]:
        """Clone a module's parameters (for parameter-delta evidence)."""
        return [p.detach().clone() for p in module.parameters()]

    def _param_delta(
        self, module: Any, before: Sequence[torch.Tensor],
    ) -> float:
        """L2 norm of the parameter change since ``before``."""
        total = 0.0
        for p, prev in zip(module.parameters(), before):
            total += float((p.detach() - prev).pow(2).sum())
        return math.sqrt(total)

    def train_step_adapter(
        self,
        batch: Any,
    ) -> PDiagnostics:
        """
        Perform a full PPO update over a :class:`~mcf.ppo_engine.adapter.RolloutBatch`.

        This is the REAL PPO update. It executes, in order:

          1. ``config.ppo_epochs`` passes over the rollout
          2. minibatch splitting by ``config.minibatch_size`` (shuffled per epoch)
          3. clipped surrogate objective
          4. value loss against computed returns
          5. entropy bonus on the CURRENT policy distribution
          6. global gradient clipping by ``config.max_grad_norm``
          7. approximate-KL monitoring after every epoch
          8. target-KL early stopping that terminates the epoch loop

        The action is re-evaluated through ``batch.build_evaluate_fn()``, which
        routes to ``policy.evaluate_actions()`` for the stored RAW sampled
        action. The adapter carries the decoder's token sequence, the per-step
        fleet snapshot and the critic input unchanged; nothing is flattened into
        a synthetic tensor action.

        Parameters
        ----------
        batch : RolloutBatch
            Packaged rollout from ``build_rollout_batch()``.

        Returns
        -------
        PDiagnostics
            Aggregated diagnostics plus per-epoch execution evidence.
        """
        n_samples = batch.n_samples
        ppo_epochs = self.config.ppo_epochs
        mb_size = batch.minibatch_size(self.config.minibatch_size)
        n_minibatches = batch.n_minibatches(self.config.minibatch_size)

        # Returns / advantages are computed ONCE on the stored values, then
        # reused by every epoch (standard PPO).
        returns, advantages = self.compute_returns_and_advantages(
            batch.old_values, batch.rewards, batch.dones,
        )
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        policy_before = self._param_snapshot(self.policy)
        critic_before = self._param_snapshot(self.critic)

        # Ratio statistics on the UNCHANGED policy: on-policy the ratio must be
        # ~1. Recorded before any optimizer step.
        ratio_stats = batch.measure_ratios(
            self.policy, self.critic, self.config.clip_epsilon,
        )

        evaluate_fn = batch.build_evaluate_fn(self.policy, self.critic)

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        clip_fraction_sum = 0.0
        approx_kl_sum = 0.0
        grad_norm_sum = 0.0
        total_optimizer_steps = 0
        epochs_executed = 0
        kl_early_stopped = False
        epoch_trace: List[Dict[str, float]] = []
        last_grad_norm = 0.0
        last_applied_norm = 0.0

        params = list(self.policy.parameters()) + list(self.critic.parameters())

        for epoch in range(ppo_epochs):
            indices = torch.randperm(n_samples, device=batch.old_log_probs.device)

            ep_policy = 0.0
            ep_value = 0.0
            ep_entropy = 0.0
            ep_clip = 0.0
            ep_kl = 0.0
            ep_grad = 0.0
            ep_steps = 0

            for start in range(0, n_samples, mb_size):
                batch_idx = indices[start:start + mb_size]

                new_log_probs, new_entropies, new_values = evaluate_fn(batch_idx)

                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                policy_loss, clip_fraction, approx_kl = self.compute_ppo_loss(
                    new_log_probs, batch.old_log_probs[batch_idx], b_adv,
                    self.config.clip_epsilon,
                )
                value_loss = self.compute_value_loss(
                    new_values.view_as(b_returns), b_returns,
                )
                # CURRENT policy entropy => a real gradient reaches the policy.
                entropy_loss = self.compute_entropy_bonus(new_entropies)

                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    + entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, self.config.max_grad_norm,
                )
                self.optimizer.step()
                total_optimizer_steps += 1
                ep_steps += 1

                last_grad_norm = float(grad_norm) if hasattr(grad_norm, "item") else float(grad_norm)
                # Post-clip norm is what the optimizer actually receives. It is
                # min(pre_clip, max_grad_norm) for a single global clip.
                last_applied_norm = min(last_grad_norm, self.config.max_grad_norm)

                ep_policy += policy_loss.item()
                ep_value += value_loss.item()
                ep_entropy += entropy_loss.item()
                ep_clip += clip_fraction
                ep_kl += approx_kl
                ep_grad += last_grad_norm

            epochs_executed += 1

            # Epoch-level averages and the KL actually observed this epoch.
            denom = max(1, ep_steps)
            ep_kl_mean = ep_kl / denom
            epoch_trace.append({
                "epoch": epoch + 1,
                "minibatches": ep_steps,
                "optimizer_steps": ep_steps,
                "policy_loss": ep_policy / denom,
                "value_loss": ep_value / denom,
                "entropy_loss": ep_entropy / denom,
                "clip_fraction": ep_clip / denom,
                "approx_kl": ep_kl_mean,
                "grad_norm_pre_clip": ep_grad / denom,
                "grad_norm_applied": min(ep_grad / denom, self.config.max_grad_norm),
            })

            policy_loss_sum += ep_policy
            value_loss_sum += ep_value
            entropy_sum += ep_entropy
            clip_fraction_sum += ep_clip
            approx_kl_sum += ep_kl
            grad_norm_sum += ep_grad

            # target-KL early stopping, measured on THIS epoch's rollout.
            if self.should_early_stop_kl(ep_kl_mean):
                kl_early_stopped = True
                break

        denom_total = max(1, epochs_executed * n_minibatches)
        with torch.no_grad():
            adv_mean = float(advantages.mean())
            adv_std = float(advantages.std()) if advantages.std() > 0 else 0.0
            val_mean = float(batch.old_values.mean())
            val_std = float(batch.old_values.std()) if batch.old_values.std() > 0 else 0.0

        self._update_count += 1

        return PDiagnostics(
            policy_loss=policy_loss_sum / denom_total,
            value_loss=value_loss_sum / denom_total,
            entropy_loss=entropy_sum / denom_total,
            total_loss=(
                policy_loss_sum / denom_total
                + self.config.value_coefficient * (value_loss_sum / denom_total)
                + (entropy_sum / denom_total)
            ),
            approx_kl=approx_kl_sum / denom_total,
            clip_fraction=clip_fraction_sum / denom_total,
            gradient_norm=grad_norm_sum / denom_total,
            advantage_mean=adv_mean,
            advantage_std=adv_std,
            value_mean=val_mean,
            value_std=val_std,
            epochs_configured=ppo_epochs,
            epochs_executed=epochs_executed,
            minibatch_size=mb_size,
            minibatches_per_epoch=n_minibatches,
            total_optimizer_steps=total_optimizer_steps,
            kl_early_stopped=kl_early_stopped,
            applied_grad_norm=last_applied_norm,
            policy_param_delta=self._param_delta(self.policy, policy_before),
            critic_param_delta=self._param_delta(self.critic, critic_before),
            old_log_prob_mean=ratio_stats["old_log_prob_mean"],
            new_log_prob_mean=ratio_stats["new_log_prob_mean"],
            ratio_mean=ratio_stats["ratio_mean"],
            ratio_min=ratio_stats["ratio_min"],
            ratio_max=ratio_stats["ratio_max"],
            ratio_outside_clip_fraction=ratio_stats["ratio_outside_clip_fraction"],
            epoch_trace=epoch_trace,
        )

    def train_step(
        self,
        states: torch.Tensor,
        actions: Any,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
        batches: int,
    ) -> PDiagnostics:
        """
        Legacy flat-tensor PPO update, retained for the original P10 tests.

        NOTE [G11.2.4]: this signature cannot represent the encoder-decoder
        action (a GraphTensors bundle plus an autoregressive token sequence and
        a fleet snapshot), which is why the active training path now calls
        :meth:`train_step_adapter` instead. The production entry point is
        :meth:`train_step_adapter`; this method remains only for the
        flat-tensor contract exercised by ``tests/test_ppo_engine.py``.

        The core loss primitives, minibatch loop and config fields are shared
        with that path; see :meth:`train_step_adapter` for the instrumented,
        target-KL-gated implementation that production uses.
        """
        clip_fraction_sum = 0.0
        approx_kl_sum = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0

        n_samples = states.size(0)
        batch_size = max(1, n_samples // batches)

        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(n_samples, device=states.device)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_old_values = old_values[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_entropies = entropies[batch_idx]

                self.optimizer.zero_grad()

                policy_output = self.policy.forward(batch_states, {})
                new_log_probs = self.policy.log_prob(policy_output)
                new_entropy = self.policy.entropy(policy_output)

                value_preds = self.critic(batch_states).squeeze(-1)

                policy_loss, cf, kl = self.compute_ppo_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages, self.config.clip_epsilon,
                )
                value_loss = self.compute_value_loss(value_preds, batch_returns)
                entropy_loss = self.compute_entropy_bonus(batch_entropies)

                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    + entropy_loss
                )

                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                clip_fraction_sum += cf
                approx_kl_sum += kl

        n_updates = max(1, self.config.ppo_epochs * batches)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        avg_clip_fraction = clip_fraction_sum / n_updates
        avg_approx_kl = approx_kl_sum / n_updates

        with torch.no_grad():
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item() if advantages.std() > 0 else 0.0
            val_mean = old_values.mean().item()
            val_std = old_values.std().item() if old_values.std() > 0 else 0.0

        self._update_count += 1

        return PDiagnostics(
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            entropy_loss=avg_entropy_loss,
            total_loss=avg_policy_loss + avg_value_loss + avg_entropy_loss,
            approx_kl=avg_approx_kl,
            clip_fraction=avg_clip_fraction,
            gradient_norm=grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
            advantage_mean=adv_mean,
            advantage_std=adv_std,
            value_mean=val_mean,
            value_std=val_std,
        )

    def should_early_stop_kl(self, approx_kl: float) -> bool:
        """
        Check if KL divergence exceeds target (early stopping criterion).

        Parameters
        ----------
        approx_kl : float
            Current approximate KL divergence.

        Returns
        -------
        bool
            True if training should stop early.
        """
        return approx_kl > self.config.target_kl

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint.

        Parameters
        ----------
        path : str
            File path for checkpoint.
        metadata : dict, optional
            Additional metadata to save (update count, config, etc.).
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "update_count": self._update_count,
            "total_env_steps": self._total_env_steps,
        }
        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        """
        Load training checkpoint.

        Parameters
        ----------
        path : str
            File path for checkpoint.
        map_location : str
            Device to map tensors to.

        Returns
        -------
        dict
            Loaded checkpoint data.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "update_count" in checkpoint:
            self._update_count = checkpoint["update_count"]
        if "total_env_steps" in checkpoint:
            self._total_env_steps = checkpoint["total_env_steps"]

        return checkpoint

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps

    def step(self) -> None:
        """Increment environment step counter."""
        self._total_env_steps += 1
