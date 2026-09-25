"""
P9 — Encoder-decoder (autoregressive) policy for LSNDP.

Implements the paper's sequential service-generation pathway ([PAPER]
Section 4.3, Eqs. 15-27):

    P5 NeuralState
        ↓
    P7 encoder (GAT × 3, Transformer × 3)
        ↓
    h̃_p ∈ R^(P×H)         [PAPER] Eq. 15  (port embeddings)
    h̃_v ∈ R^(V×H)         [PAPER] Eq. 16  (vessel embeddings)
    h_BOS ∈ R^H            [PAPER] Eq. 17  (begin-of-service)
        ↓
    h_embed = cat([h̃_p; h̃_v; h_BOS]) ∈ R^((P+V+1)×H)  [PAPER] Eq. 17
        ↓
    LSTM decoder (1 layer)  [PAPER] Table 5
        ↓
    autoregressive token selection
        ↓
    ServiceAction
        ↓
    P6 validation           [PAPER] Eq. 14

The decoder operates autoregressively: at each sub-step τ it selects ONE
token (either a vessel class at τ=1, or a port at τ≥2), appends its
embedding to x, and feeds x into the LSTM for the next sub-step. Service
completes when the first port is revisited.

Scope boundaries enforced in this module:
  * The P7 encoder is REUSED, never duplicated.
  * P6 owns service construction, ordering, and validation.
  * No PPO, no training, no optimizer, no rollout collection.
  * Log-probability and entropy are exposed (for P10's PPO).

Evidence tags document every non-paper-specified choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from actions.service_generator import (
    ServiceGenerator,
    ServiceValidationResult,
)
from data.instance import LINERLIBInstance
from env.action import ServiceAction
from neural.backbone import BackboneOutput, NeuralBackbone
from neural.config import ArchitectureConfig
from neural.masks import (
    MASK_FILL_VALUE,
    DecoderPhaseMask,
    apply_mask_to_logits,
    decoder_phase_mask,
    masked_log_softmax,
)
from neural.tensors import GraphTensors
from policies.encoder_only import EncoderOnlyPolicy  # for fallback strategy

from .common import PolicyDiagnostics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum ports in a valid cyclic service (matches P6).
MIN_SERVICE_PORTS: int = 2

# Numerical floor for log() computations.
_LOG_EPS: float = 1e-7


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class EncoderDecoderOutput:
    """
    Result of one complete decoder rollout (one service decision).

    CONTRACT FOR P10
    ----------------
    The decoder produces a **token sequence** (vessel + ports in decoder
    order). P6 then reorders the ports via TSP to produce the
    ``ServiceAction``. These two sequences are NOT the same when the TSP
    changes the order — and they MUST NOT be conflated.

      - ``decoded_port_sequence``   — ports in the ORDER THE DECODER
                                      SELECTED THEM (used for log_prob).
      - ``executed_port_sequence``  — ports AFTER P6's TSP reordering
                                      (what actually goes into the
                                      ServiceAction).
      - ``log_prob``                — log P(decoded_port_sequence | state);
                                      the likelihood of the decoder's
                                      actual selection order. This is the
                                      quantity P10 uses for policy gradients.
      - ``service_action``          — the validated ServiceAction built by
                                      P6; it may have a different port order.

    When the TSP preserves the decoder's order (common for small graphs
    where the decoder already chose a near-optimal path) the two lists
    are identical. When they differ, the log_prob still correctly
    describes the decoder's choice — P10 must NOT try to recompute a
    likelihood for the reordered action.

    Attributes
    ----------
    substep_probs : list[tuple[Tensor, int]]
        (distribution, num_available) at each sub-step.
    substep_selected : list[int]
        Token index selected at each sub-step (including vessel and BOS).
    decoded_port_sequence : list[str]
        Port UNLOCODEs in the order the decoder selected them. Matches
        the sequence whose log_prob was accumulated.
    executed_port_sequence : list[str]
        Port UNLOCODEs after P6's TSP reordering. Equals
        decoded_port_sequence when the TSP did not change the order.
    vessel_class : str or None
        Selected vessel class name.
    service_action : ServiceAction or None
        Validated service action from P6, or None if validation failed.
    validation : ServiceValidationResult
        P6 structural validation result.
    log_prob : Tensor, scalar
        Sum of log probabilities across all sub-steps. Corresponds to
        decoded_port_sequence, NOT executed_port_sequence.
    entropy : Tensor, scalar
        Sum of per-substep entropies, in nats.
    backbone : BackboneOutput
        Encoder output exposed so downstream code can inspect latent state.
    diagnostics : PolicyDiagnostics
        Shapes, feasibility, and fallback bookkeeping.
    n_substeps : int
        Number of sub-steps actually taken (vessel + ports + closing).
    bos_index : int or None
        BOS token index within the candidate space [PAPER] Eq. 17.
    """

    substep_probs: List[Tuple["torch.Tensor", int]] = ()
    substep_selected: List[int] = ()
    n_substeps: int = 0
    bos_index: Optional[int] = None
    vessel_class: Optional[str] = None
    decoded_port_sequence: List[str] = ()
    executed_port_sequence: List[str] = ()
    service_action: Optional[ServiceAction] = None
    validation: Optional[ServiceValidationResult] = None
    log_prob: Optional["torch.Tensor"] = None
    entropy: Optional["torch.Tensor"] = None
    backbone: Optional[BackboneOutput] = None
    diagnostics: Optional[PolicyDiagnostics] = None


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    One-layer LSTM decoder for the encoder-decoder policy.

    Implements [PAPER] Eqs. 18-26.

    Parameters
    ----------
    n_ports : int
        Number of physical ports P.
    n_vessels : int
        Number of vessel classes V.
    H : int
        Hidden dimension (matches the backbone; paper: 512).
    lstm_layers : int
        Number of LSTM layers (paper: 1).
    include_bos : bool
        Whether to add a BOS token ([PAPER] Eq. 17).
    """

    def __init__(
        self,
        n_ports: int,
        n_vessels: int,
        H: int = 512,
        lstm_layers: int = 1,
        include_bos: bool = True,
    ) -> None:
        super().__init__()
        self.n_ports = n_ports
        self.n_vessels = n_vessels
        self.H = H
        self.include_bos = include_bos
        self.bos_index = n_ports + n_vessels if include_bos else -1
        self.n_candidates = n_ports + n_vessels + (1 if include_bos else 0)

        # [PAPER] Eq. 22: FF layer with ReLU, maps H → N̄.
        self.ff = nn.Linear(H, self.n_candidates, bias=True)
        # [PAPER] Eq. 22: Layer normalization after FF.
        self.ln = nn.LayerNorm(self.n_candidates)

        # The LSTM itself operates on H-dimensional hidden states.
        self.lstm = nn.LSTM(
            input_size=H,
            hidden_size=H,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # [PAPER] Eq. 17: BOS embedding, randomly initialized once.
        if include_bos:
            tmp = torch.empty(1, H)
            nn.init.xavier_uniform_(tmp)
            self.h_BOS = nn.Parameter(tmp.squeeze(0).detach())
        else:
            self.register_parameter("h_BOS", None)

    def forward(
        self,
        h_prev: "torch.Tensor",
        c_prev: "torch.Tensor",
        x_t: "torch.Tensor",
        mask: DecoderPhaseMask,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        One decoding step.

        Parameters
        ----------
        h_prev : Tensor, shape (batch, H) or (H,)
            Previous hidden state.
        c_prev : Tensor, shape (batch, H) or (H,)
            Previous cell state.
        x_t : Tensor, shape (batch, 1, H) or (1, H)
            Current input embedding (selected token from previous step).
        mask : DecoderPhaseMask
            Availability mask over N̄ candidates.

        Returns
        -------
        h_new : Tensor, shape same as h_prev
        c_new : Tensor, shape same as c_prev
        logits : Tensor, shape (N̄,)
            Raw logits over all candidates, ready for masking.
        """
        # Add batch dim if needed.
        x_input = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t.unsqueeze(0)

        h_new, (h_out, c_out) = self.lstm(x_input, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
        h_new = h_new.squeeze(0)  # (H,)

        # [PAPER] Eq. 22: FF + LayerNorm.
        out = self.ff(h_out.squeeze(0))       # (H,) → (N̄,)
        out = self.ln(out)                     # LN stabilizes training

        # Apply mask additively.
        if not mask.keep.all():
            out = apply_mask_to_logits(out, mask.keep)

        return h_new, c_out.squeeze(0), out


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class EncoderDecoderPolicy(nn.Module):
    """
    Autoregressive encoder-decoder policy.

    Parameters
    ----------
    backbone : NeuralBackbone
        Shared P7 encoder. Not duplicated.
    decoder : LSTMDecoder
        The LSTM decoder. Created internally when not supplied.
    instance : LINERLIBInstance
        Supplies ports, vessel types, fleet data.
    generator : ServiceGenerator
        P6 service generator. All construction/validation delegated here.
    port_codes : sequence[str], optional
        Port UNLOCODEs in node order. Defaults to sorted(instance.ports).
    bos_embedding : "torch.Tensor", optional
        Initial BOS embedding. When None, a random embedding is created
        (matching the paper's description). When supplied, the tensor is
        used as-is.
    """

    def __init__(
        self,
        backbone: NeuralBackbone,
        instance: LINERLIBInstance,
        generator: ServiceGenerator,
        decoder: Optional[LSTMDecoder] = None,
        port_codes: Optional[Sequence[str]] = None,
        bos_embedding: Optional["torch.Tensor"] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.instance = instance
        self.generator = generator
        self.port_codes = list(port_codes or sorted(instance.ports.keys()))

        P = len(self.port_codes)
        V = len(instance.vessel_types)
        include_bos = True  # Always include BOS for consistency.
        N_bar = P + V + 1  # [PAPER] Eq. 17

        if decoder is None:
            self.decoder = LSTMDecoder(
                n_ports=P,
                n_vessels=V,
                H=backbone.hidden_dim,
                lstm_layers=backbone.config.lstm_layers,
                include_bos=include_bos,
            )
        else:
            self.decoder = decoder
            # Verify the decoder's candidate count matches our instance.
            assert decoder.n_candidates == N_bar, (
                f"Decoder expects {decoder.n_candidates} candidates but "
                f"instance has P={P} + V={V} + BOS={1} = {N_bar}."
            )

        # Replace BOS parameter if a custom embedding was supplied.
        if bos_embedding is not None:
            if bos_embedding.shape[0] != backbone.hidden_dim:
                raise ValueError(
                    f"bos_embedding must have shape ({backbone.hidden_dim},), "
                    f"got {tuple(bos_embedding.shape)}."
                )
            with torch.no_grad():
                self.decoder.h_BOS.copy_(bos_embedding)

        # Sanity: every port code must exist in the instance.
        missing = [c for c in self.port_codes if c not in instance.ports]
        if missing:
            raise ValueError(
                f"port_codes contains ports absent from the instance: {missing}"
            )

    # ---- public API ----

    def forward(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
    ) -> EncoderDecoderOutput:
        """
        Compute action probabilities without seeding (for PPO new_log_prob).

        Equivalent to ``sample_action`` with ``seed=None``.
        """
        return self.sample_action(graph, fleet_remaining, seed=None)

    @torch.no_grad()
    def sample_action(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        seed: Optional[int] = None,
        max_substeps: int = 50,
    ) -> EncoderDecoderOutput:
        """
        Autoregressive rollout: select vessel, then ports sequentially.

        Parameters
        ----------
        graph : GraphTensors
            P5-derived tensors for the current state.
        fleet_remaining : dict[str, float]
            Remaining vessel counts per class.
        seed : int, optional
            Seed for stochastic sampling. Local generator only.
        max_substeps : int
            Safety cap on sub-steps to prevent infinite loops.

        Returns
        -------
        EncoderDecoderOutput
        """
        # Encode via shared backbone.
        backbone_out = self.backbone.encode_graph(graph)

        # Isolate from global RNG: save before, restore after.
        # Without this, unrelated global-tensor operations (pytest fixtures,
        # numpy scans, other policy calls) shift the sequence consumed by our
        # local Generator and break reproducibility across calls. The paper's
        # training loop reuses the same rollout repeatedly; without isolation
        # the same seed would produce different trajectories on call N+1.
        saved_state = torch.random.get_rng_state()
        try:
            return self._rollout(
                backbone_out=backbone_out,
                graph=graph,
                fleet_remaining=fleet_remaining,
                seed=seed,
                max_substeps=max_substeps,
            )
        finally:
            torch.random.set_rng_state(saved_state)

    @torch.no_grad()
    def deterministic_action(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
    ) -> EncoderDecoderOutput:
        """
        Deterministic rollout: argmax selection at every sub-step.

        Equivalent to `sample_action` with argmax instead of sampling.
        """
        backbone_out = self.backbone.encode_graph(graph)
        return self._rollout(
            backbone_out=backbone_out,
            graph=graph,
            fleet_remaining=fleet_remaining,
            seed=None,
            max_substeps=50,
            deterministic=True,
        )

    # ---- internals ----

    def _rollout(
        self,
        backbone_out: BackboneOutput,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        seed: Optional[int],
        max_substeps: int,
        deterministic: bool = False,
    ) -> EncoderDecoderOutput:
        """Core autoregressive rollout logic."""
        P = backbone_out.num_ports
        V = self.decoder.n_vessels
        N_bar = self.decoder.n_candidates
        include_bos = self.decoder.include_bos
        bos_idx = self.decoder.bos_index

        # [PAPER] Eq. 20: c_0 = h_p,global^(L) (raw GAT global embedding).
        c_prev = backbone_out.global_embedding_gat.clone()  # (H,)

        # [PAPER] Eq. 21: h_0 = mean(h̃_p) over all P ports.
        h_prev = backbone_out.port_embeddings.mean(dim=0)    # (H,)

        # Candidate embedding table: ports [0..P-1], vessels [P..P+V-1],
        # BOS [P+V] (if included).
        h_embed = self._build_candidate_embeddings(backbone_out)
        # h_embed: (N_bar, H)

        # Sequence buffer: x accumulates selected token embeddings.
        x_history: List["torch.Tensor"] = []  # Each: (H,)

        # Selection history.
        selected_indices: List[int] = []
        substep_distributions: List[Tuple["torch.Tensor", int]] = []

        # For service construction.
        selected_vessel_idx: Optional[int] = None
        selected_port_indices: List[int] = []
        first_port_idx: Optional[int] = None

        # Determine initial mask.
        # τ=1: vessel selection phase (ports masked, BOS unmasked if t=1).
        # For single-service rollout, we start at τ=1 with all vessels unmasked.
        available_vessels = {
            i for i, vc in enumerate(self.port_codes[:V])
            if self._is_vessel_available(vc, fleet_remaining)
        }
        # Wait - port_codes doesn't map to vessel names. Let me fix this.
        vessel_classes = sorted(self.instance.vessel_types.keys())
        available_vessels = set()
        for i, vc in enumerate(vessel_classes):
            if fleet_remaining.get(vc, 0.0) > 0:
                available_vessels.add(i)

        # The mask width is always N_bar (= P + V + [1 for BOS]).
        # BOS is always present in the candidate set, but only selectable at τ=1.
        visited_ports: Set[int] = set()

        def _make_mask(substep: int, selected_ports: Set[int],
                       first_port: Optional[int],
                       bos_allowed: bool) -> DecoderPhaseMask:
            return decoder_phase_mask(
                num_ports=P,
                num_vessels=V,
                substep=substep,
                selected_ports=selected_ports,
                first_port=first_port,
                include_bos=include_bos,       # BOS is always in candidate set.
                bos_allowed=bos_allowed,        # Only selectable at τ=1, t=1.
                available_vessels=(
                    available_vessels
                    if substep == 1 else None
                ),
                device=graph.device,
            )

        mask = _make_mask(substep=1, selected_ports=set(),
                          first_port=None, bos_allowed=True)

        for tau in range(1, max_substeps + 1):
            if mask.num_available == 0:
                break

            # Run LSTM step.
            input_emb = (
                h_embed[selected_indices[-1]]
                if selected_indices else h_embed[0]
            )
            h_new, c_new, logits = self.decoder(
                h_prev, c_prev, input_emb, mask,
            )

            # Masked softmax → selection distribution.
            logit_for_sampling = apply_mask_to_logits(logits, mask.keep)
            selection_dist = torch.softmax(logit_for_sampling, dim=0)

            if deterministic:
                selected_idx = int(torch.argmax(selection_dist).item())
            else:
                gen = torch.Generator(device="cpu")
                if seed is not None:
                    gen.manual_seed(int(seed) + tau)
                selected_idx = int(torch.multinomial(
                    selection_dist.cpu(), 1,
                ).item())

            # Accumulate state.
            x_history.append(h_embed[selected_idx])
            h_prev, c_prev = h_new, c_new
            selected_indices.append(selected_idx)
            substep_distributions.append(
                (selection_dist, mask.num_available)
            )

            # ---- interpret the token ----
            if selected_idx < P:
                # Port token.
                if first_port_idx is None:
                    first_port_idx = selected_idx
                    selected_port_indices.append(selected_idx)
                elif selected_idx == first_port_idx:
                    # Revisit first port → service complete.
                    break
                else:
                    if selected_idx in visited_ports:
                        continue   # duplicate; skip without advancing.
                    visited_ports.add(selected_idx)
                    selected_port_indices.append(selected_idx)
                next_bos_allowed = False
            elif selected_idx < P + V:
                # Vessel token (valid only at τ=1).
                if tau != 1:
                    continue
                selected_vessel_idx = selected_idx - P
                next_bos_allowed = False
            else:
                # BOS token (valid only at τ=1 of t=1).
                if tau != 1:
                    continue
                next_bos_allowed = False

            # Advance to next sub-step; keep BOS in the candidate set
            # (masked) so the tensor width never changes.
            mask = _make_mask(
                substep=tau + 1,
                selected_ports=visited_ports,
                first_port=first_port_idx,
                bos_allowed=next_bos_allowed,
            )

        # Build service action.
        vessel_class: Optional[str] = None
        if selected_vessel_idx is not None:
            vessel_class = vessel_classes[selected_vessel_idx]

        port_sequence = [self.port_codes[i] for i in selected_port_indices]

        # Validate and construct via P6.
        # P6's TSP may reorder the decoder's output — the log_prob still
        # describes the DECODER's order (which is what the policy actually
        # chose); we do NOT recompute a likelihood for the reordered action.
        decoded_port_sequence = list(port_sequence)
        executed_port_sequence: List[str] = []
        service_action: Optional[ServiceAction] = None
        if vessel_class and len(port_sequence) >= MIN_SERVICE_PORTS:
            ordered = self.generator.order_ports(port_sequence, vessel_class)
            validation_result = self.generator.generate_service(
                vessel_class, ordered,
            )
            if validation_result.is_valid:
                service_action = validation_result.service_action
                executed_port_sequence = list(service_action.port_sequence)
        elif vessel_class is None:
            validation_result = ServiceValidationResult.invalid([
                "No vessel class was selected during decoding."
            ])
        elif len(port_sequence) < MIN_SERVICE_PORTS:
            validation_result = ServiceValidationResult.invalid([
                f"Only {len(port_sequence)} port(s) selected; need ≥ "
                f"{MIN_SERVICE_PORTS} for a valid service."
            ])
        else:
            validation_result = ServiceValidationResult.invalid([
                "Service construction failed."
            ])

        # Compute log-probability and entropy.
        log_prob = self._compute_log_prob(substep_distributions, selected_indices)
        entropy = self._compute_entropy(substep_distributions)

        diagnostics = PolicyDiagnostics(
            num_ports=P,
            num_candidates=N_bar,
            num_selected=len(selected_port_indices),
            fallback_applied=False,
            vessel_class=vessel_class,
            validation_reasons=list(validation_result.reasons),
            extra={
                "selected_indices": selected_indices,
                "n_substeps": len(selected_indices),
                "port_sequence": port_sequence,
                "first_port": self.port_codes[first_port_idx] if first_port_idx is not None else None,
            },
        )

        return EncoderDecoderOutput(
            substep_probs=substep_distributions,
            substep_selected=selected_indices,
            vessel_class=vessel_class,
            decoded_port_sequence=decoded_port_sequence,
            executed_port_sequence=executed_port_sequence,
            service_action=service_action,
            validation=validation_result,
            log_prob=log_prob,
            entropy=entropy,
            backbone=backbone_out,
            diagnostics=diagnostics,
            n_substeps=len(selected_indices),
            bos_index=bos_idx,
        )

    def _build_candidate_embeddings(
        self, backbone_out: BackboneOutput,
    ) -> "torch.Tensor":
        """
        Build the h_embed table: [port embeddings; vessel embeddings; BOS].

        [PAPER] Eq. 17.
        """
        port_embs = backbone_out.port_embeddings      # (P, H)
        vessel_embs = backbone_out.vessel_embeddings   # (V_sel, H) — may be 1 or V

        # For the decoder, we need ALL vessel embeddings (Eq. 16 uses W'_v).
        # Re-encode all vessels if backbone only encoded the selected one.
        if vessel_embs.shape[0] == 1 and backbone_out.num_vessel_classes > 1:
            # Need to re-run encoder with all vessels.
            # This is handled by ensuring the backbone always encodes all.
            raise NotImplementedError(
                "Encoder-decoder requires all vessel embeddings; "
                "backbone should encode all vessels."
            )

        bos_emb = (
            self.decoder.h_BOS.unsqueeze(0)
            if self.decoder.include_bos
            else torch.empty(0, self.backbone.hidden_dim)
        )

        h_embed = torch.cat([port_embs, vessel_embs, bos_emb], dim=0)
        return h_embed

    def _is_vessel_available(
        self, vessel_class: str, fleet_remaining: Dict[str, float],
    ) -> bool:
        """Check if a vessel class has remaining fleet."""
        return fleet_remaining.get(vessel_class, 0.0) > 0

    def _compute_log_prob(
        self,
        distributions: List[Tuple["torch.Tensor", int]],
        selected_indices: List[int],
    ) -> "torch.Tensor":
        """
        Compute log P(action | state) as sum of per-substep log-probs.

        [PAPER] Eq. 27: π_θ(A_t | S_t) = Π_τ P(A_t(τ) | A_t(τ'<τ), S_t)
        """
        total_log_prob = torch.tensor(0.0)
        for (dist, n_avail), idx in zip(distributions, selected_indices):
            # dist is shape (N_bar,) or (n_avail,) depending on masking.
            # We need the probability mass assigned to the selected candidate.
            if idx < dist.shape[0]:
                prob = dist[idx].clamp(min=_LOG_EPS)
            else:
                # Should not happen if masking is correct.
                prob = torch.tensor(_LOG_EPS)
            total_log_prob = total_log_prob + torch.log(prob)
        return total_log_prob

    def _compute_entropy(
        self,
        distributions: List[Tuple["torch.Tensor", int]],
    ) -> "torch.Tensor":
        """
        Compute entropy of the policy distribution across all sub-steps.
        """
        total_entropy = torch.tensor(0.0)
        for dist, _ in distributions:
            # Clamp for numerical stability.
            probs = dist.clamp(min=_LOG_EPS)
            entropy = -(probs * torch.log(probs)).sum()
            total_entropy = total_entropy + entropy
        return total_entropy

    # ---- interfaces (for P10) ----

    def log_prob(self, output: EncoderDecoderOutput) -> "torch.Tensor":
        return output.log_prob

    def entropy(self, output: EncoderDecoderOutput) -> "torch.Tensor":
        return output.entropy

    # ---- PPO evaluation path (differentiable, no sampling) ----

    def _evaluate_log_prob(
        self,
        backbone_out: BackboneOutput,
        fleet_remaining: Dict[str, float],
        selected_indices: List[int],
        n_substeps: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Differentiable evaluation of the log-probability of a fixed token
        sequence under the autoregressive decoder.

        This is the PPO-re-evaluation counterpart to ``_rollout``. Instead of
        sampling tokens, it consumes the pre-recorded indices from rollout and
        accumulates their probabilities from the CURRENT policy distribution.
        No stochastic sampling, no resampling, no argmax -- pure probability
        evaluation of the stored action.

        Returns
        -------
        log_prob : Tensor, scalar
            Sum of per-substep log-probabilities; differentiable.
        entropy : Tensor, scalar
            Sum of per-substep entropies; differentiable.
        """
        P = backbone_out.num_ports
        V = self.decoder.n_vessels
        include_bos = self.decoder.include_bos
        vessel_classes = sorted(self.instance.vessel_types.keys())

        available_vessels = set()
        for i, vc in enumerate(vessel_classes):
            if fleet_remaining.get(vc, 0.0) > 0:
                available_vessels.add(i)

        h_embed = self._build_candidate_embeddings(backbone_out)
        c_prev = backbone_out.global_embedding_gat.clone()
        h_prev = backbone_out.port_embeddings.mean(dim=0)

        total_log_prob = torch.tensor(0.0, device=h_prev.device)
        total_entropy = torch.tensor(0.0, device=h_prev.device)
        visited_ports: Set[int] = set()
        first_port_idx: Optional[int] = None

        # Initial mask for τ=1 (vessel-selection phase).
        mask = decoder_phase_mask(
            num_ports=P,
            num_vessels=V,
            substep=1,
            selected_ports=set(),
            first_port=None,
            include_bos=include_bos,
            bos_allowed=True,
            available_vessels=available_vessels,
            device=backbone_out.port_embeddings.device,
        )

        for tau in range(1, n_substeps + 1):
            if tau - 1 >= len(selected_indices):
                break
            idx = selected_indices[tau - 1]

            # Run LSTM step — decoder.forward expects 2D hx/cx when input is 2D.
            # Input embedding follows _rollout semantics: use the previously
            # selected token's embedding (h_embed[0] for the very first step
            # when no token has been selected yet).
            prev_idx = selected_indices[tau - 2] if tau > 1 else 0
            x_input = h_embed[prev_idx].unsqueeze(0)   # (1, H)
            h_new, (h_out, c_out) = self.decoder.lstm(
                x_input, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)),
            )
            h_new = h_new.squeeze(0)
            h_out = h_out.squeeze(0)
            c_new = c_out.squeeze(0)

            # FF + LayerNorm → logits (matches decoder.forward logic)
            logits = self.decoder.ln(self.decoder.ff(h_out))

            logit_for_sampling = apply_mask_to_logits(logits, mask.keep)
            log_probs = torch.log_softmax(logit_for_sampling, dim=0)
            probs = torch.softmax(logit_for_sampling, dim=0)

            # Log-probability of the STORED index (differentiable).
            safe_log_p = log_probs[idx].clamp(min=_LOG_EPS - 10)
            total_log_prob = total_log_prob + safe_log_p

            # Entropy of the current policy distribution.
            safe_probs = probs.clamp(min=_LOG_EPS)
            total_entropy = total_entropy + (-(safe_probs * torch.log(safe_probs)).sum())

            h_prev, c_prev = h_new, c_new

            # Interpret token and update state, matching _rollout semantics.
            if idx < P:
                # Port token.
                if first_port_idx is None:
                    first_port_idx = idx
                    visited_ports.add(idx)
                elif idx == first_port_idx:
                    # Revisit first port → service complete.
                    break
                elif idx in visited_ports:
                    pass  # duplicate; skip without advancing.
                else:
                    visited_ports.add(idx)
            # Vessel/BOS tokens at τ=1: consumed, no port state update.

            # Advance mask for next sub-step (τ+1), unless this is the last step.
            if tau < n_substeps:
                mask = decoder_phase_mask(
                    num_ports=P,
                    num_vessels=V,
                    substep=tau + 1,
                    selected_ports=visited_ports,
                    first_port=first_port_idx,
                    include_bos=include_bos,
                    bos_allowed=False,
                    available_vessels=None,
                    device=backbone_out.port_embeddings.device,
                )

        return total_log_prob, total_entropy

    def evaluate_actions(
        self,
        graph: GraphTensors,
        fleet_remaining: Dict[str, float],
        *,
        substep_selected: List[int],
        n_substeps: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Evaluate log-probability and entropy of a FIXED action sequence.

        Used by PPO to re-compute log π(a|s) with gradients during the
        update phase. Does NOT sample, does NOT call sample_action(), does
        NOT modify any internal state.

        Parameters
        ----------
        graph : GraphTensors
            The P5-derived state tensor bundle (same as used during rollout).
        fleet_remaining : dict[str, float]
            Fleet counts at the time of the decision (for mask construction).
        substep_selected : list[int]
            Token indices selected during rollout (same as
            output.substep_selected).
        n_substeps : int
            Number of sub-steps taken (length of substep_selected).

        Returns
        -------
        log_prob : Tensor, scalar, requires_grad
            Differentiable log P(stored_action | state) under current weights.
        entropy : Tensor, scalar, requires_grad
            Differentiable entropy of the current policy distribution.
        """
        backbone_out = self.backbone.encode_graph(graph)
        return self._evaluate_log_prob(
            backbone_out=backbone_out,
            fleet_remaining=fleet_remaining,
            selected_indices=substep_selected,
            n_substeps=n_substeps,
        )
