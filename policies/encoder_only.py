"""
P8 — Encoder-only policy for LSNDP.

Implements the paper's one-shot port-selection pathway ([PAPER] Section 4.2,
Eqs. 12-14):

    P5 NeuralState
        ↓
    P7 GAT + Transformer                     [PAPER] Eq. 11
        ↓
    port logits → sigmoid probabilities      [PAPER] Eqs. 12
        ↓
    independent Bernoulli per port           [PAPER] Eq. 13
        ↓
    selected port set Ã_p                    [PAPER] Eq. 13
        ↓
    rule-based largest available vessel      [PAPER] Section 4.2
        ↓
    P6 approximate TSP                       [PAPER] Eq. 14
        ↓
    ServiceAction

Scope boundaries enforced in this module:
  * No PPO, no training, no optimizer, no rollout collection.
  * The TSP and vessel-selection logic are REUSED from P6, never reimplemented.
  * No service-construction logic — P6 owns that.

The policy DOES expose log-probability and entropy (P8.8) because P10's PPO will
need them. Exposing them is not implementing PPO: there is no advantage, no
ratio, no clipping, and no value loss here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from actions.service_generator import (
    ServiceGenerator,
    ServiceValidationResult,
    select_largest_available_vessel,
)
from data.instance import LINERLIBInstance
from env.action import ServiceAction
from neural.backbone import BackboneOutput, NeuralBackbone
from neural.config import ArchitectureConfig
from neural.tensors import GraphTensors

from .common import PolicyDiagnostics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of ports for a valid cycle. Reuses P6's rule ([P6]
# `_MIN_PORT_COUNT = 2`); a service must return to its origin port.
MIN_SERVICE_PORTS: int = 2

# Numerical floor for log() of a probability. Keeps log-prob finite when a
# Bernoulli probability saturates to exactly 0 or 1 in float32.
_PROB_EPS: float = 1e-7


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class EncoderOnlyOutput:
    """
    Everything the encoder-only policy produces for one decision.

    CONTRACT FOR P10
    ----------------
    This policy has an **inference-time repair step** (the fallback) that
    may change the port set after the stochastic Bernoulli draw.  The
    fields below are named so P10 can never confuse the *sampled* action
    with the *executed* action.

      - ``raw_sampled_ports``       — ports from the Bernoulli draw
                                      (BEFORE any repair).
      - ``executed_ports``          — ports actually passed to P6
                                      (AFTER repair, if any).
      - ``fallback_applied``        — True iff a repair was performed.
      - ``raw_log_prob``            — log P(raw_sampled_ports | state);
                                      the only probability we own.
      - ``executed_log_prob``       — *alias* for raw_log_prob. We do NOT
                                      fabricate a likelihood for the repaired
                                      action; it would be off-policy.

    When ``fallback_applied`` is False the two port lists are identical and
    ``raw_log_prob`` is the log-probability of the service that will be
    submitted to P6.  When it is True P10 must treat the sample as
    **off-policy** (or discard it) — the paper's PPO derivation assumes
    every trajectory comes directly from π_θ, and a post-hoc port addition
    violates that assumption.

    Attributes
    ----------
    port_logits : Tensor, shape (P,)
        Raw logits from the linear head ([PAPER] Eq. 12, before sigmoid).
    port_probabilities : Tensor, shape (P,)
        ñ_p = sigmoid(logits) ∈ [0, 1]^P  [PAPER] Eq. 12.
    port_mask : BoolTensor, shape (P,)
        Availability mask used for this decision ("keep" convention).
    selected_mask : BoolTensor, shape (P,)
        The raw Bernoulli draw X_p ∈ {0,1}^P  [PAPER] Eq. 13 — BEFORE any
        fallback repair.  This is the action whose likelihood is
        ``raw_log_prob``.
    raw_sampled_ports : list[str]
        Port UNLOCODEs from the raw Bernoulli draw, in node-index order.
        Equal to ``executed_ports`` when no fallback fired.
    executed_ports : list[str]
        Port UNLOCODEs actually handed to P6 for ordering and validation,
        AFTER the deterministic fallback (if applied).
    fallback_applied : bool
        True when the Bernoulli draw yielded fewer than
        MIN_SERVICE_PORTS ports and the top-probability fallback added
        more.
    vessel_class : str or None
        Vessel chosen by the P6 rule-based selector.
    service_action : ServiceAction or None
        The validated action from P6, or None when structural validation
        failed (e.g. no vessel available).
    validation : ServiceValidationResult
        P6's structural verdict on the assembled action.
    raw_log_prob : Tensor, scalar
        log P(raw_sampled_ports) under the independent Bernoulli model
        ([PAPER] Eq. 13). **This is the only log-probability this policy
        owns.** It describes the raw Bernoulli action, NOT the repaired
        executed action.
    entropy : Tensor, scalar
        Sum of per-port binary entropies — the entropy of the product
        distribution over all P ports. Diagnostic for P10's entropy bonus.
    backbone : BackboneOutput
        Encoder output exposed so downstream code need not re-run the
        encoder.
    diagnostics : PolicyDiagnostics
        Shapes, feasibility and fallback bookkeeping.

    Fallback contract (must not be violated by P10)
    -----------------------------------------------
    Fallback-repaired samples must not be treated as ordinary on-policy
    PPO samples unless P10 explicitly accounts for the repair
    transformation.  Concretely:

      * The importance-weight correction for a fallback sample would
        require log P(executed_ports | state) − log P(raw_ports | state),
        but we do NOT compute the former because it is not well-defined
        under the Bernoulli model (the repaired ports were not drawn).
      * The safe default for P10 is to **discard** fallback-repaired
        samples from the experience buffer (they are exploration errors,
        not policy outputs) or to clamp their importance weight to 1.0
        with an explicit comment explaining why.

    P10 owns the final decision; this module only makes the distinction
    visible via the fields above.
    """

    port_logits: "torch.Tensor"
    port_probabilities: "torch.Tensor"
    port_mask: "torch.Tensor"
    selected_mask: "torch.Tensor"
    raw_sampled_ports: List[str]
    executed_ports: List[str]
    fallback_applied: bool
    vessel_class: Optional[str]
    service_action: Optional[ServiceAction]
    validation: ServiceValidationResult
    raw_log_prob: "torch.Tensor"
    entropy: "torch.Tensor"
    backbone: BackboneOutput
    diagnostics: PolicyDiagnostics

    @property
    def executed_log_prob(self) -> "torch.Tensor":
        """Alias for ``raw_log_prob``. We do not fabricate a probability
        for the repaired action; the alias exists so callers that think in
        terms of 'executed' semantics can write uniform code."""
        return self.raw_log_prob

    @property
    def num_selected(self) -> int:
        return len(self.executed_ports)

    @property
    def is_valid_action(self) -> bool:
        return self.validation.is_valid and self.service_action is not None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class EncoderOnlyPolicy(nn.Module):
    """
    Encoder-only (one-shot) policy.

    Parameters
    ----------
    backbone : NeuralBackbone
        The shared P7 encoder. Not duplicated, not modified.
    instance : LINERLIBInstance
        Supplies ports, vessel types and fleet data for P6.
    generator : ServiceGenerator
        P6's service generator. ALL service construction, ordering and
        validation is delegated here.
    port_codes : sequence[str], optional
        Port UNLOCODEs in node order. Defaults to sorted(instance.ports),
        which is P5's node ordering. Must match the backbone's input ordering.
    fallback_strategy : str
        [ENGINEERING DECISION] What to do when the Bernoulli draw yields fewer
        than MIN_SERVICE_PORTS ports:
          "top_probability" (default) — add the highest-probability ports until
              the minimum is met.
          "invalid" — return an invalid action rather than adjusting the draw.
        Documented in docs/P8_FINAL_REPORT.md.
    """

    def __init__(
        self,
        backbone: NeuralBackbone,
        instance: LINERLIBInstance,
        generator: ServiceGenerator,
        port_codes: Optional[Sequence[str]] = None,
        fallback_strategy: str = "top_probability",
    ) -> None:
        super().__init__()
        if fallback_strategy not in ("top_probability", "invalid"):
            raise ValueError(
                "fallback_strategy must be 'top_probability' or 'invalid', "
                f"got {fallback_strategy!r}."
            )
        self.backbone = backbone
        self.instance = instance
        self.generator = generator
        self.port_codes = list(port_codes or sorted(instance.ports.keys()))
        self.fallback_strategy = fallback_strategy

        # [PAPER] Eq. 12: W_p maps the graph embedding to a P-dimensional
        # vector. Here it maps each port embedding to a single logit, which is
        # the per-port form of the same linear map.
        self.port_head = nn.Linear(backbone.hidden_dim, 1, bias=True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Sanity: every port code must exist in the instance.
        missing = [c for c in self.port_codes if c not in instance.ports]
        if missing:
            raise ValueError(
                f"port_codes contains ports absent from the instance: {missing}"
            )

    # ---- probabilities ----

    def port_logits(
        self,
        backbone_out: BackboneOutput,
        port_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Raw logits over ports.

        [PAPER] Eq. 12 computes ñ_p = sigmoid((W_p ĥ_p)^T). The sigmoid is
        applied by the caller; logits are returned separately so log-probability
        can be computed without the numerical loss of inverting a sigmoid.
        """
        logits = self.port_head(backbone_out.port_embeddings).squeeze(-1)
        if port_mask is not None:
            if port_mask.shape != logits.shape:
                raise ValueError(
                    f"port_mask shape {tuple(port_mask.shape)} != logits shape "
                    f"{tuple(logits.shape)}."
                )
        return logits

    def port_probabilities(
        self,
        backbone_out: BackboneOutput,
        port_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """ñ_p = sigmoid(logits) ∈ [0, 1]^P  [PAPER] Eq. 12."""
        return torch.sigmoid(self.port_logits(backbone_out, port_mask))

    # ---- action construction ----

    def forward(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        port_mask: Optional["torch.Tensor"] = None,
    ) -> EncoderOnlyOutput:
        """
        Compute port probabilities and build an action by sampling.

        Equivalent to `sample_action` with an unseeded draw; provided so
        `policy(graph, fleet)` reads naturally. Use `sample_action` when a
        reproducible seed is required.
        """
        return self.sample_action(graph, fleet_remaining, seed=None,
                                  port_mask=port_mask)

    @torch.no_grad()
    def sample_action(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        seed: Optional[int] = None,
        port_mask: Optional["torch.Tensor"] = None,
    ) -> EncoderOnlyOutput:
        """
        Sample an action: sigmoid probabilities → Bernoulli → vessel → TSP.

        Parameters
        ----------
        graph : GraphTensors
            P5-derived tensors for the current state.
        fleet_remaining : dict[str, float]
            Remaining vessel count per class (from P4).
        seed : int, optional
            Seed for the Bernoulli draw. A LOCAL generator is created; no
            global RNG state is read or written, so this cannot perturb
            unrelated code paths.
        port_mask : BoolTensor, optional
            Availability mask in "keep" convention.

        Returns
        -------
        EncoderOnlyOutput
        """
        backbone_out = self.backbone.encode_graph(graph)
        return self._decide(
            backbone_out, graph, fleet_remaining,
            deterministic=False, seed=seed, port_mask=port_mask,
        )

    @torch.no_grad()
    def deterministic_action(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        port_mask: Optional["torch.Tensor"] = None,
    ) -> EncoderOnlyOutput:
        """
        Deterministic inference: threshold the Bernoulli at 0.5.

        Each port is included iff its probability is at least 0.5, which is the
        per-port argmax of an independent Bernoulli. The minimum-service
        fallback still applies afterwards.
        """
        backbone_out = self.backbone.encode_graph(graph)
        return self._decide(
            backbone_out, graph, fleet_remaining,
            deterministic=True, seed=None, port_mask=port_mask,
        )

    # ---- internals ----

    def _decide(
        self,
        backbone_out: BackboneOutput,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        deterministic: bool,
        seed: Optional[int],
        port_mask: Optional["torch.Tensor"],
    ) -> EncoderOnlyOutput:
        """Shared implementation behind the sampling entry points."""
        num_ports = backbone_out.num_ports
        if num_ports != len(self.port_codes):
            raise ValueError(
                f"Graph has {num_ports} ports but the policy was built for "
                f"{len(self.port_codes)}. port_codes must match P5's node "
                f"ordering."
            )
        if num_ports < MIN_SERVICE_PORTS:
            raise ValueError(
                f"Instance has {num_ports} ports; at least "
                f"{MIN_SERVICE_PORTS} are required for a service."
            )

        if port_mask is None:
            port_mask = torch.ones(
                num_ports, dtype=torch.bool, device=graph.device,
            )
        else:
            port_mask = port_mask.to(graph.device).bool()
            if port_mask.shape != (num_ports,):
                raise ValueError(
                    f"port_mask must have shape ({num_ports},), got "
                    f"{tuple(port_mask.shape)}."
                )

        logits = self.port_logits(backbone_out, port_mask)
        probs = torch.sigmoid(logits)

        # ---- [PAPER] Eq. 13: X_p ~ Bernoulli(ñ_p), independent per port ----
        if deterministic:
            selected_mask = probs >= 0.5
        else:
            generator = torch.Generator(device="cpu")
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(int(seed))
            # Sample from a CPU generator then move, so a seeded draw gives
            # identical results regardless of the model's device.
            selected_mask = (
                torch.rand(num_ports, generator=generator) < probs.cpu()
            ).to(graph.device)

        # Masked ports cannot be selected, whatever the draw produced.
        selected_mask = selected_mask & port_mask

        # ---- log-probability and entropy of the Bernoulli model ----
        log_prob = self._bernoulli_log_prob(logits, selected_mask, port_mask)
        entropy = self._bernoulli_entropy(logits, port_mask)

        sampled_idx = [
            i for i in range(num_ports) if bool(selected_mask[i])
        ]
        raw_sampled_ports = [self.port_codes[i] for i in sampled_idx]

        # ---- [ENGINEERING DECISION] minimum-service fallback ----
        fallback_applied = False
        if len(sampled_idx) < MIN_SERVICE_PORTS:
            if self.fallback_strategy == "invalid":
                diagnostics = self._diagnostics(
                    num_ports, port_mask, sampled_idx, fallback_applied=False,
                )
                return EncoderOnlyOutput(
                    port_logits=logits,
                    port_probabilities=probs,
                    port_mask=port_mask,
                    selected_mask=selected_mask,
                    raw_sampled_ports=raw_sampled_ports,
                    executed_ports=raw_sampled_ports,
                    fallback_applied=False,
                    vessel_class=None,
                    service_action=None,
                    validation=ServiceValidationResult.invalid([
                        f"Bernoulli draw selected {len(sampled_idx)} port(s); "
                        f"at least {MIN_SERVICE_PORTS} are required and the "
                        f"'invalid' strategy forbids adjusting the draw."
                    ]),
                    raw_log_prob=log_prob,
                    entropy=entropy,
                    backbone=backbone_out,
                    diagnostics=diagnostics,
                )
            chosen_idx = self._fallback_ports(
                probs, port_mask, sampled_idx, fleet_remaining,
            )
            selected_mask = torch.zeros_like(selected_mask)
            selected_mask[chosen_idx] = True
            fallback_applied = True
        else:
            chosen_idx = sampled_idx

        # ---- [PAPER] Section 4.2: rule-based vessel selection (P6's rule) ----
        vessel_class = select_largest_available_vessel(
            fleet_remaining, self.instance.vessel_types,
        )

        # ---- [PAPER] Eq. 14: P6 approximate TSP over the selected set ----
        service_action: Optional[ServiceAction] = None
        executed_ports = [self.port_codes[i] for i in chosen_idx]
        if vessel_class is None:
            validation = ServiceValidationResult.invalid([
                "No vessel class has remaining fleet (fleet_remaining is all "
                "zero), so the encoder-only rule-based selector returned None."
            ])
        else:
            # Delegate ordering AND validation to P6. P8 implements no TSP.
            validation = self.generator.generate_service(
                vessel_class,
                self.generator.order_ports(executed_ports, vessel_class),
            )
            if validation.is_valid:
                service_action = validation.service_action

        diagnostics = self._diagnostics(
            num_ports, port_mask, chosen_idx,
            fallback_applied=fallback_applied,
            vessel_class=vessel_class,
            validation=validation,
        )

        return EncoderOnlyOutput(
            port_logits=logits,
            port_probabilities=probs,
            port_mask=port_mask,
            selected_mask=selected_mask,
            raw_sampled_ports=raw_sampled_ports,
            executed_ports=executed_ports,
            fallback_applied=fallback_applied,
            vessel_class=vessel_class,
            service_action=service_action,
            validation=validation,
            raw_log_prob=log_prob,
            entropy=entropy,
            backbone=backbone_out,
            diagnostics=diagnostics,
        )

    def _fallback_ports(
        self,
        probs: "torch.Tensor",
        port_mask: "torch.Tensor",
        already: List[int],
        fleet_remaining: Dict[str, float],
    ) -> List[int]:
        """
        Supply ports when the Bernoulli draw fell below the minimum.

        [ENGINEERING DECISION] Deterministic, and free of economic reasoning:

          1. Candidate order is by DESCENDING probability, with ties broken by
             ASCENDING node index — so the result is reproducible.
          2. Ports that are draft-feasible for the shipable vessel classes come
             first, so the assembled service is likely to pass P6's structural
             validation. This is structural feasibility only; no profitability
             or reward information is consulted (P8.6 forbids it).
          3. Fill up to MIN_SERVICE_PORTS.

        The fallback never removes a port the Bernoulli draw already selected;
        it only adds.

        Returns
        -------
        list[int]
            Node indices for the action, the already-selected ones first.
        """
        chosen = list(already)
        needed = MIN_SERVICE_PORTS - len(chosen)
        if needed <= 0:
            return chosen

        # Structural preference: ports reachable by at least one vessel class
        # with remaining fleet. Uses P6's own feasibility predicate.
        feasible: Set[str] = set()
        for vc, qty in fleet_remaining.items():
            if qty <= 0:
                continue
            for code in self.port_codes:
                if self.generator.can_visit_port(vc, code):
                    feasible.add(code)

        candidates = [
            i for i in range(len(self.port_codes))
            if bool(port_mask[i]) and i not in chosen
        ]
        # Sort: feasible first, then probability desc, then index asc.
        candidates.sort(key=lambda i: (
            0 if self.port_codes[i] in feasible else 1,
            -float(probs[i].item()),
            i,
        ))
        chosen.extend(candidates[:needed])
        return chosen

    # Vessel availability is threaded through the fallback via this attribute,
    # set immediately before _decide uses it. Kept explicit rather than passing
    # fleet_remaining through several private signatures.
    _fleet_snapshot: Dict[str, float] = {}

    def _diagnostics(
        self,
        num_ports: int,
        port_mask: "torch.Tensor",
        chosen_idx: List[int],
        fallback_applied: bool,
        vessel_class: Optional[str] = None,
        validation: Optional[ServiceValidationResult] = None,
    ) -> PolicyDiagnostics:
        return PolicyDiagnostics(
            num_ports=num_ports,
            num_candidates=int(port_mask.sum()),
            num_selected=len(chosen_idx),
            fallback_applied=fallback_applied,
            vessel_class=vessel_class,
            validation_reasons=list(validation.reasons) if validation else [],
            extra={
                "selected_node_indices": list(chosen_idx),
                "selected_port_codes": [self.port_codes[i] for i in chosen_idx],
                "fallback_strategy": self.fallback_strategy,
                "min_service_ports": MIN_SERVICE_PORTS,
            },
        )

    # ---- log probability and entropy (for P10's PPO; PPO itself is not here) ----

    @staticmethod
    def _bernoulli_log_prob(
        logits: "torch.Tensor",
        selected: "torch.Tensor",
        port_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        log P(X_p) for the independent Bernoulli product ([PAPER] Eq. 13).

        Computed from LOGITS via logsigmoid, not from the squashed
        probabilities, so that a saturated port does not collapse the result to
        log(0) = -inf.

        Masked ports contribute nothing: they are not part of the decision, and
        including them would charge the action for a probability the policy was
        never allowed to act on.
        """
        sel = selected.to(logits.dtype)
        # log p for selected, log(1-p) for unselected — both via logits.
        log_p = torch.nn.functional.logsigmoid(logits)
        log_1mp = torch.nn.functional.logsigmoid(-logits)
        per_port = sel * log_p + (1.0 - sel) * log_1mp
        per_port = torch.where(
            port_mask, per_port, torch.zeros_like(per_port),
        )
        return per_port.sum()

    @staticmethod
    def _bernoulli_entropy(
        logits: "torch.Tensor",
        port_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Entropy of the independent Bernoulli product, in nats.

        H(X) = Σ_i [ -p_i log p_i - (1-p_i) log(1-p_i) ]

        Masked ports are excluded, matching `_bernoulli_log_prob`. Computed in
        logit space for stability: for a Bernoulli with logit z,
        H = log(1+e^z) - z·σ(z) = softplus(z) - z·σ(z).
        """
        p = torch.sigmoid(logits)
        # softplus(z) - z*sigmoid(z), the stable closed form.
        per_port = torch.nn.functional.softplus(logits) - logits * p
        per_port = torch.where(
            port_mask, per_port, torch.zeros_like(per_port),
        )
        return per_port.sum()

    # ---- interfaces ----

    def log_prob(self, output: EncoderOnlyOutput) -> "torch.Tensor":
        """
        log P(action | state) for the Bernoulli port decision.

        Returns ``output.raw_log_prob`` — the likelihood of the raw
        Bernoulli action. When ``output.fallback_applied`` is True this
        does NOT describe the executed (repaired) action; see the class
        docstring for the P10 contract.
        """
        return output.raw_log_prob

    def entropy(self, output: EncoderOnlyOutput) -> "torch.Tensor":
        """Entropy of the port-selection distribution, in nats."""
        return output.entropy

    # ---- PPO evaluation path (differentiable, no sampling) ----

    def evaluate_actions(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        *,
        selected_mask: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Evaluate log-probability and entropy of a FIXED Bernoulli action mask.

        Used by PPO to re-compute log π(a|s) with gradients during the
        update phase. Does NOT sample, does NOT call sample_action().

        Parameters
        ----------
        graph : GraphTensors
            The P5-derived state tensor bundle.
        fleet_remaining : dict[str, float]
            Fleet counts at decision time (for vessel availability).
        selected_mask : BoolTensor, shape (P,)
            The raw Bernoulli selection mask from rollout.

        Returns
        -------
        log_prob : Tensor, scalar, requires_grad
            Differentiable log P(selected_mask | state).
        entropy : Tensor, scalar, requires_grad
            Differentiable entropy of the current Bernoulli distribution.
        """
        backbone_out = self.backbone.encode_graph(graph)
        port_mask = torch.ones(
            backbone_out.num_ports, dtype=torch.bool, device=graph.device,
        )
        logits = self.port_logits(backbone_out, port_mask)
        log_prob = self._bernoulli_log_prob(logits, selected_mask, port_mask)
        entropy = self._bernoulli_entropy(logits, port_mask)
        return log_prob, entropy
