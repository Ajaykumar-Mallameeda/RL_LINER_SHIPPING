"""
P7 — Masking interfaces.

Masks are produced here as reusable ARCHITECTURAL inputs. Two rules govern
what belongs in this module:

  1. Only masks the architecture actually needs are defined. The paper's
     encoder-only pathway (Eqs. 12-13) applies NO mask at all — it samples an
     independent Bernoulli per port — so `encoder_only_port_mask` returns the
     all-True default and nothing else is invented for it.
  2. Masks may only use information available at decision time. No mask here
     may reference reward, profit, or any future quantity. The masks below are
     structural (draft feasibility, already-visited ports, sub-step phase).

Two conventions coexist in this module and are named explicitly to avoid the
classic inversion bug:

  * "keep" convention (this module's default, and what P8/P9 sampling uses):
        True  = the candidate IS available
        False = masked out / forbidden
  * PyTorch `key_padding_mask` convention (what nn.Transformer consumes):
        True  = IGNORE this position
        False = attend
    `to_pytorch_padding_mask` performs the inversion.

This module owns NO policy decision, NO PPO masking, and NO learned selection.
Producing the mask is P7's job; acting under it is P8/P9's.

Evidence tags on every mask below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .tensors import GraphTensors


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASK_FILL_VALUE: float = -1e9
"""Additive value used to suppress a masked logit before a softmax.

Finite rather than -inf so that a softmax over an entirely-masked row yields a
uniform distribution instead of NaN — the caller decides whether that state is
legal. [ENGINEERING DECISION]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_available(n: int, device=None, dtype=None) -> "torch.Tensor":
    """A (n,) boolean tensor with every candidate available."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return torch.ones(n, dtype=torch.bool, device=device)


def none_available(n: int, device=None) -> "torch.Tensor":
    """A (n,) boolean tensor with every candidate masked out."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return torch.zeros(n, dtype=torch.bool, device=device)


def to_pytorch_padding_mask(keep_mask: "torch.Tensor") -> "torch.Tensor":
    """
    Invert a "keep" mask into PyTorch's `key_padding_mask` convention.

    keep_mask: True = available → returns True = IGNORE.
    """
    return ~keep_mask.bool()


def apply_mask_to_logits(
    logits: "torch.Tensor",
    keep_mask: "torch.Tensor",
    fill_value: float = MASK_FILL_VALUE,
) -> "torch.Tensor":
    """
    Suppress masked logits additively, returning a NEW tensor.

    The input is never modified in place: P8/P9 keep raw logits for the log-
    probability computation, and a masked copy must not corrupt them.
    """
    if logits.shape != keep_mask.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != keep_mask shape "
            f"{tuple(keep_mask.shape)}."
        )
    return torch.where(
        keep_mask.bool(),
        logits,
        torch.full_like(logits, fill_value),
    )


def masked_log_softmax(
    logits: "torch.Tensor",
    keep_mask: "torch.Tensor",
    dim: int = -1,
) -> "torch.Tensor":
    """
    log_softmax over the available candidates only.

    Masked entries receive log-probability log(MASK_FILL_VALUE-ish), i.e. a very
    negative number, never -inf and never NaN.
    """
    masked = apply_mask_to_logits(logits, keep_mask)
    return torch.log_softmax(masked, dim=dim)


# ---------------------------------------------------------------------------
# Port masks
# ---------------------------------------------------------------------------

def encoder_only_port_mask(
    num_ports: int,
    device=None,
) -> "torch.Tensor":
    """
    Port availability mask for the encoder-only pathway.

    [PAPER] Eqs. 12-13: the encoder-only policy computes an independent
    sigmoid probability per port and samples each Bernoulli independently. The
    paper defines NO eligibility mask for this pathway.

    Returning an explicit all-True mask (rather than None) keeps the interface
    uniform while remaining faithful: no port is filtered, no economic
    criterion is introduced, and nothing about future reward is consulted.

    Returns
    -------
    BoolTensor, shape (P,) — all True.
    """
    return all_available(num_ports, device=device)


def draft_feasible_port_mask(
    bundle: "GraphTensors",
    vessel_draft: float,
    port_drafts: Sequence[Optional[float]],
    tolerance: float = 1e-6,
) -> "torch.Tensor":
    """
    Structural port mask: a port is available iff the vessel can enter it.

    [ENGINEERING DECISION] — but note the feasibility rule itself is NOT
    invented here: P6's `ServiceGenerator._is_draft_feasible` implements
    exactly `vessel_draft >= port_draft - tolerance`, and P6's
    `get_vessel_classes_for_ports` uses it to decide which classes may serve a
    port set. This function is that same rule expressed as a mask, so the two
    paths cannot disagree.

    A port with an unknown draft (None) is treated as available, matching P6.

    Parameters
    ----------
    bundle : GraphTensors
        Provides P (the mask covers physical ports, not the global node).
    vessel_draft : float
        Draft of the vessel class under consideration.
    port_drafts : Sequence[float or None]
        Draft per port, in P5 node order. Length must equal bundle.num_ports.
    tolerance : float
        Comparison slack.

    Returns
    -------
    BoolTensor, shape (P,)
    """
    if len(port_drafts) != bundle.num_ports:
        raise ValueError(
            f"port_drafts has {len(port_drafts)} entries but the graph has "
            f"{bundle.num_ports} ports."
        )
    keep = [
        True if d is None else (vessel_draft >= d - tolerance)
        for d in port_drafts
    ]
    return torch.tensor(keep, dtype=torch.bool, device=bundle.device)


def already_selected_port_mask(
    num_ports: int,
    selected: Set[int],
    allow: Optional[Set[int]] = None,
    device=None,
) -> "torch.Tensor":
    """
    Port mask excluding already-selected ports, with optional re-allowance.

    Needed by the autoregressive decoder (P9), where a port already in the
    current service must not be selected again — except that revisiting the
    FIRST port closes the service and is therefore permitted.

    [PAPER] Section 4.3: "Ports that have already been visited in step t remain
    masked, except for the first port, as revisiting it indicates the completion
    of a service generation."

    Parameters
    ----------
    num_ports : int
    selected : set[int]
        Port node indices already used in the current service.
    allow : set[int], optional
        Indices to re-allow even if selected (used to permit the closing port).

    Returns
    -------
    BoolTensor, shape (P,)
    """
    allow = allow or set()
    for idx in list(selected) + list(allow):
        if not 0 <= idx < num_ports:
            raise ValueError(
                f"Port index {idx} out of range for {num_ports} ports."
            )
    keep = [(i not in selected) or (i in allow) for i in range(num_ports)]
    return torch.tensor(keep, dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------
# Node / edge masks
# ---------------------------------------------------------------------------

def global_node_mask(
    num_ports: int,
    include_global: bool = True,
    device=None,
) -> "torch.Tensor":
    """
    Node mask over the P+1 graph nodes, marking which nodes are ports.

    [PAPER] Eq. 9 separates the GAT output into port embeddings and a single
    global-node embedding, and Section 4.2 notes "only the port embeddings are
    utilized, while the global embedding is used by the decoder". The global
    node is therefore never a port-selection candidate — it is masked out here
    so downstream code that indexes node embeddings cannot select it by
    accident.

    Returns
    -------
    BoolTensor, shape (P+1,) — True for the P ports, and for the global node
    only if `include_global`.
    """
    keep = [True] * num_ports + [bool(include_global)]
    return torch.tensor(keep, dtype=torch.bool, device=device)


def edge_padding_mask(
    bundle: "GraphTensors",
    keep: "torch.Tensor",
) -> "torch.Tensor":
    """
    Edge mask for variable-size batches.

    [ENGINEERING DECISION] There is no edge padding in the single-instance
    representation (E is exact per instance), so the default is all-True. This
    exists so a future batched loader can zero the attention contribution of
    padded edges at the single point where the message-passing weight is
    applied.

    Parameters
    ----------
    bundle : GraphTensors
    keep : BoolTensor, shape (E,) — True = real edge, False = padding.

    Returns
    -------
    BoolTensor, shape (E,)
    """
    if keep.shape != (bundle.num_edges,):
        raise ValueError(
            f"keep must have shape ({bundle.num_edges},), got "
            f"{tuple(keep.shape)}."
        )
    return keep.bool()


def apply_edge_mask(
    alpha: "torch.Tensor",
    keep: "torch.Tensor",
) -> "torch.Tensor":
    """
    Zero the attention weight of padded edges.

    Used together with `edge_padding_mask`. Returns a new tensor.
    """
    if alpha.shape[0] != keep.shape[0]:
        raise ValueError(
            f"alpha has {alpha.shape[0]} edges but keep has {keep.shape[0]}."
        )
    return alpha * keep.to(alpha.dtype)


# ---------------------------------------------------------------------------
# Decoder phase mask (consumed by P9)
# ---------------------------------------------------------------------------

@dataclass
class DecoderPhaseMask:
    """
    Mask for one autoregressive sub-step τ of the LSTM decoder.

    [PAPER] Section 4.3, masking rule:
      * τ = 1: only vessels are selectable; ALL ports masked.
      * τ ≥ 2: only ports are selectable; ALL vessels masked.
      * Ports already visited in step t stay masked, except the first port
        (revisiting it completes the service).
      * The BOS embedding is unmasked only at τ = 1 of t = 1.

    The mask is expressed over the candidate index space implied by [PAPER]
    Eq. 17: indices [0, P) are ports, [P, P+V) are vessel classes and, when a
    BOS token is present, index P+V is BOS.

    Attributes
    ----------
    keep : BoolTensor, shape (P + V + [1],)
        True = selectable at this sub-step.
    phase : str
        "vessel" for τ = 1, "port" for τ ≥ 2.
    """

    keep: "torch.Tensor"
    phase: str

    @property
    def num_candidates(self) -> int:
        return int(self.keep.shape[0])

    @property
    def num_available(self) -> int:
        return int(self.keep.sum())


def decoder_phase_mask(
    num_ports: int,
    num_vessels: int,
    substep: int,
    selected_ports: Optional[Set[int]] = None,
    first_port: Optional[int] = None,
    include_bos: bool = False,
    bos_allowed: bool = False,
    available_vessels: Optional[Set[int]] = None,
    draft_keep_ports: Optional["torch.Tensor"] = None,
    device=None,
) -> DecoderPhaseMask:
    """
    Build the phase mask for one decoder sub-step.

    Parameters
    ----------
    num_ports, num_vessels : int
        Sizes of the candidate blocks (P and V).
    substep : int
        τ, 1-based (τ = 1 selects the vessel).
    selected_ports : set[int], optional
        Ports already chosen in the current service (masked from τ ≥ 2).
    first_port : int, optional
        The service's first port — re-allowed so the service can close.
    include_bos : bool
        Whether the candidate space includes a BOS token.
    bos_allowed : bool
        [PAPER] BOS is unmasked only at τ = 1 of t = 1; the caller passes True
        exactly then.
    available_vessels : set[int], optional
        Vessel indices with remaining fleet (structural availability). Vessel
        index i is eligible iff i ∈ available_vessels. None means all classes
        are eligible.
    draft_keep_ports : BoolTensor, optional
        Pre-computed (P,) structural port mask (e.g. draft feasibility). Folded
        into the port phase via logical AND. No economic filtering is applied.

    Returns
    -------
    DecoderPhaseMask
    """
    if num_ports < 0 or num_vessels < 0:
        raise ValueError("num_ports and num_vessels must be >= 0.")
    if substep < 1:
        raise ValueError(f"substep must be >= 1, got {substep}")

    selected_ports = selected_ports or set()
    total = num_ports + num_vessels + (1 if include_bos else 0)
    keep = torch.zeros(total, dtype=torch.bool, device=device)

    if substep == 1:
        # ---- vessel phase: ports masked, vessels (and BOS) unmasked ----
        phase = "vessel"
        if available_vessels is None:
            keep[num_ports:num_ports + num_vessels] = True
        else:
            for i in range(num_vessels):
                keep[num_ports + i] = i in available_vessels
        if include_bos:
            keep[num_ports + num_vessels] = bool(bos_allowed)
    else:
        # ---- port phase: vessels masked, ports unmasked subject to rules ----
        phase = "port"
        for i in range(num_ports):
            if i in selected_ports and i != first_port:
                continue
            keep[i] = True
        if draft_keep_ports is not None:
            if draft_keep_ports.shape != (num_ports,):
                raise ValueError(
                    f"draft_keep_ports must have shape ({num_ports},), got "
                    f"{tuple(draft_keep_ports.shape)}."
                )
            keep[:num_ports] = keep[:num_ports] & draft_keep_ports.bool()

    return DecoderPhaseMask(keep=keep, phase=phase)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe_mask(keep: "torch.Tensor") -> Dict[str, object]:
    """Small summary used in tests and diagnostics."""
    n = int(keep.numel())
    available = int(keep.sum())
    return {
        "total": n,
        "available": available,
        "masked": n - available,
        "empty": available == 0,
    }
