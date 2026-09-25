"""
P7 — Graph Attention Network layers.

Implements the paper's GAT stack:

    h_p^(1) = GAT^(1)(f_p, f_e)      ∈ R^((P+1)×H)     [PAPER] Eq. 7
    h_p^(l) = GAT^(l)(h_p^(l-1), f_e) ∈ R^((P+1)×H)     [PAPER] Eq. 8

with L = 3 layers [PAPER] Table 5.

Paper-specified properties implemented here:
  * Only NODE features are transformed; edge features "remain unchanged"
    through the stack [PAPER, Section 4.1] — this module never reassigns the
    edge feature tensor, it only reads it.
  * Node features enter at the true input dimensionality D_in (= 2 per
    Eq. 37-38) and are projected to H inside the first layer. Projecting the
    input up to H outside the GAT would silently insert a layer the paper does
    not specify.

Implementation choices where the paper is silent are tagged
[ENGINEERING DECISION] and documented in docs/P7_FINAL_REPORT.md:

  * Attention head count per GAT layer (paper gives a head count for the
    Transformer only). Default 1.
  * Activation on the layer output. Default ELU.
  * How edge features reach the attention logit: the paper requires edge
    information to flow into node representations (otherwise Eq. 7-8 would be
    edge-blind), and states that edge features are not transformed across
    layers. "additive" projects the (untouched) edge features into the
    attention logit, which satisfies both statements. Default "additive".
  * Raw logits are used to index edges — no `softmax` over a dense P×P matrix,
    so memory is O(E) and behaviour is independent of port ordering.

Additive / sum attention differs from the original GAT mean-aggregation; the
paper does not specify the aggregation, so the sum form is used and recorded.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ArchitectureConfig


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class EdgeFeatureAttention(nn.Module):
    """
    Single-head GAT attention over the directed edge set.

    For each edge (i, j):

        e_ij = a_src(h_i) + a_dst(h_j) + a_edge(fe_ij)
        α_ij = leaky_relu(e_ij)                       [ENGINEERING DECISION]
        α_ij = softmax over incoming edges of j       [PAPER, GAT formulation]

    and the update is

        h'_j = Σ_i α_ij · W h_i                       [ENGINEERING DECISION: sum]

    Parameters
    ----------
    in_dim : int
        Input node feature dimension for THIS layer (D_in for layer 1, H after).
    out_dim : int
        Output node feature dimension (H).
    edge_dim : int
        Edge feature dimension (D_e).
    negative_slope : float
        LeakyReLU slope.
    use_edge_features : bool
        When False, edge features do not enter the attention logit.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        negative_slope: float = 0.2,
        use_edge_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.use_edge_features = use_edge_features

        # Node transform W (Eq. 7-8: node features are what gets transformed).
        self.w_src = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_dst = nn.Parameter(torch.empty(in_dim, out_dim))

        # Attention vectors a_src, a_dst over the projected features.
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))

        # Edge-feature projection into the scalar attention logit.
        if use_edge_features:
            self.w_edge = nn.Parameter(torch.empty(edge_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Glorot-style init, mirroring the reference GAT initialisation."""
        nn.init.xavier_uniform_(self.w_src)
        nn.init.xavier_uniform_(self.w_dst)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)
        if self.use_edge_features:
            # Small init: edge features are already normalised by P5, and a
            # large initial edge bias destabilises early attention.
            nn.init.normal_(self.w_edge, std=0.1)

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        x : Tensor, shape (N, in_dim)
        edge_index : LongTensor, shape (2, E) — rows are (src, dst)
        edge_attr : Tensor, shape (edge_dim, E) — column i is edge i's features

        Returns
        -------
        Tensor, shape (N, out_dim)
        """
        N = x.shape[0]
        E = edge_index.shape[1]

        if E == 0:
            # Degenerate graph: no messages, so the update is a zero embedding
            # of the right shape. Callers that need a non-trivial fallback
            # should guarantee at least one edge.
            return x.new_zeros((N, self.out_dim))

        src = edge_index[0]
        dst = edge_index[1]

        # ---- per-node projections ----
        x_src = x @ self.w_src   # (N, out_dim)
        x_dst = x @ self.w_dst   # (N, out_dim)

        # ---- gather endpoint projections per edge ----
        h_src = x_src.index_select(0, src)   # (E, out_dim)
        h_dst = x_dst.index_select(0, dst)   # (E, out_dim)

        # ---- attention logits ----
        logits = (h_src * self.a_src).sum(dim=1) + (h_dst * self.a_dst).sum(dim=1)

        if self.use_edge_features:
            # edge_attr is (edge_dim, E); column i holds edge i's features.
            logits = logits + (self.w_edge @ edge_attr)

        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        # ---- softmax over the incoming edges of each destination node ----
        alpha = _segment_softmax(logits, dst, N)

        # ---- message passing (sum aggregation) ----
        messages = h_src * alpha.unsqueeze(1)          # (E, out_dim)
        out = x.new_zeros((N, self.out_dim))
        out = out.index_add(0, dst, messages)

        return out


def _segment_softmax(
    logits: "torch.Tensor",
    index: "torch.Tensor",
    num_segments: int,
) -> "torch.Tensor":
    """
    Numerically stable softmax of `logits` grouped by `index`.

    Equivalent to, for each segment g, softmax over {logits[e] : index[e] = g}.

    Uses the max-subtraction trick, so it is safe for large graphs where raw
    logits could overflow exp(). Implemented with index_add/index_reduce so it
    stays O(E) with no dense P×P intermediate.

    A segment with no incoming edges contributes nothing (its output rows are
    simply never touched), which is the correct GAT behaviour for isolated or
    source-only nodes.
    """
    if logits.numel() == 0:
        return logits

    # Per-segment max. scatter_reduce (unlike index_reduce) is out of beta and
    # handles segments with no incoming edges, which stay -inf.
    seg_max = torch.full(
        (num_segments,), float("-inf"), dtype=logits.dtype, device=logits.device,
    ).scatter_reduce(0, index, logits, reduce="amax", include_self=False)
    # Segments with no members stay -inf; map those to 0 to avoid NaN below.
    seg_max = torch.where(
        torch.isfinite(seg_max), seg_max, torch.zeros_like(seg_max)
    )

    shifted = logits - seg_max.index_select(0, index)
    exp_shifted = torch.exp(shifted)

    # Per-segment denominator.
    denom = logits.new_zeros((num_segments,))
    denom = denom.index_add(0, index, exp_shifted)

    # Guard against an all-zero denominator for a segment that has members
    # (can only occur if every logit underflowed); fall back to uniform.
    denom_safe = torch.where(
        denom > 0, denom, torch.ones_like(denom)
    )
    alpha = exp_shifted / denom_safe.index_select(0, index)

    # Rows whose denominator was zero become 0 weight rather than 1/0.
    zero_mask = (denom.index_select(0, index) <= 0)
    alpha = torch.where(zero_mask, torch.zeros_like(alpha), alpha)
    return alpha


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    One GAT layer with residual connection + layer normalisation.

    h^(l) = LN( h^(l-1) + act( MultiHeadAttention(h^(l-1), fe) ) )

    The paper specifies neither the residual nor the normalisation at the GAT
    level ([PAPER] Section 4.1 gives only Eqs. 7-8 and states that edge
    features are unchanged). Both are [ENGINEERING DECISION]s, adopted because
    a 3-layer attention stack without them does not train stably. The residual
    is skipped when the input dimension differs from H (the first layer), where
    a linear shortcut is used instead so the identity path still exists.

    Parameters
    ----------
    in_dim : int
        Input feature dimension for this layer.
    config : ArchitectureConfig
        Supplies hidden_dim, head count, activation, dropout, edge-feature mode.
    is_first : bool
        True for layer 1 (input dim D_in ≠ H).
    """

    def __init__(
        self,
        in_dim: int,
        config: ArchitectureConfig,
        is_first: bool = False,
        edge_dim: int = 4 + 2 + 100,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = config.hidden_dim
        self.num_heads = config.gat_heads
        self.head_dim = config.gat_head_dim
        self.is_first = is_first
        # Reserved edge-feature width: static(4) + dynamic(2 + |S|_max).
        # P5 produces 4 + 2 + num_services_active columns; the missing service
        # rows are zero-padded per forward pass (see _pad_edge_attr).
        self.edge_dim = edge_dim

        self.heads = nn.ModuleList([
            EdgeFeatureAttention(
                in_dim=in_dim,
                out_dim=self.head_dim,
                edge_dim=edge_dim,
                negative_slope=config.gat_negative_slope,
                use_edge_features=(config.gat_edge_features == "additive"),
            )
            for _ in range(self.num_heads)
        ])

        # Output projection: concatenated heads (H) -> H.
        self.out_proj = nn.Linear(
            self.head_dim * self.num_heads, self.out_dim, bias=False,
        )

        # Residual shortcut, needed whenever in_dim != out_dim.
        self.residual = (
            nn.Identity() if in_dim == self.out_dim
            else nn.Linear(in_dim, self.out_dim, bias=False)
        )

        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.activation_name = config.gat_activation

    def _pad_edge_attr(self, edge_attr: "torch.Tensor") -> "torch.Tensor":
        """
        Zero-pad or truncate the edge feature block to the reserved width.

        P5 emits 4 + 2 + num_services_active feature rows, which equals the
        reserved 4 + 2 + |S|_max width only once the service cap is reached.
        Reserved-but-absent rows are zero, which is exactly what a zero-weight
        row would contribute, so padding is behaviour-preserving for the
        trained weights — and at initialisation the pad rows contribute
        nothing because they are zero.

        Truncation is refused rather than performed silently: losing a real
        feature row would corrupt every edge's representation.

        Raises
        ------
        ValueError
            If the block is WIDER than the reserved width (a genuine schema
            violation — P5 produced more features than the architecture was
            built for).
        """
        rows = edge_attr.shape[0]
        if rows == self.edge_dim:
            return edge_attr
        if rows > self.edge_dim:
            raise ValueError(
                f"edge feature block has {rows} rows but the GAT was built for "
                f"{self.edge_dim}. P5 emitted more features than the "
                f"architecture reserves; rebuild the GAT with the correct "
                f"edge_dim rather than silently dropping features."
            )
        pad = edge_attr.new_zeros((self.edge_dim - rows, edge_attr.shape[1]))
        return torch.cat([edge_attr, pad], dim=0)

    def _activate(self, x: "torch.Tensor") -> "torch.Tensor":
        name = self.activation_name
        if name == "elu":
            return F.elu(x)
        if name == "relu":
            return F.relu(x)
        if name == "gelu":
            return F.gelu(x)
        if name == "tanh":
            return torch.tanh(x)
        if name == "leaky_relu":
            return F.leaky_relu(x)
        raise ValueError(f"Unsupported GAT activation {name!r}.")

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        x : Tensor, shape (N, in_dim)
        edge_index : LongTensor, shape (2, E)
        edge_attr : Tensor, shape (edge_dim, E)

        Returns
        -------
        Tensor, shape (N, out_dim)
        """
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"GATLayer expects {self.in_dim} input features, got "
                f"{x.shape[1]}."
            )
        edge_attr = self._pad_edge_attr(edge_attr)

        head_outputs = [
            head(x, edge_index, edge_attr) for head in self.heads
        ]
        h = torch.cat(head_outputs, dim=1) if len(head_outputs) > 1 \
            else head_outputs[0]
        h = self.out_proj(h)
        h = self._activate(h)
        h = self.dropout(h)

        h = h + self.residual(x)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

class GATStack(nn.Module):
    """
    L stacked GAT layers (L = 3 by default, [PAPER] Table 5).

    Runs on the physical/indexed graph. `edge_index` and `edge_attr` are passed
    unchanged to every layer — the paper states edge features remain unchanged
    through the stack ([PAPER], Section 4.1).

    The stack returns embeddings for ALL nodes including the global node at
    index P, which the paper separates out after the final layer:
        h_p^(L) = [h_p,port^(L), h_p,global^(L)]   [PAPER] Eq. 9
    """

    def __init__(
        self,
        in_dim: int,
        config: ArchitectureConfig,
        edge_dim: int = 4 + 2 + 100,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.num_layers = config.gat_layers
        self.edge_dim = edge_dim

        self.layers = nn.ModuleList([
            GATLayer(
                in_dim=in_dim if l == 0 else config.hidden_dim,
                config=config,
                is_first=(l == 0),
                edge_dim=edge_dim,
            )
            for l in range(self.num_layers)
        ])

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        x : Tensor, shape (P+1, D_in)
            Initial node features f_p ([PAPER] Eq. 37-38).
        edge_index : LongTensor, shape (2, E)
        edge_attr : Tensor, shape (edge_dim, E)
            Concatenated static + dynamic edge features. Never modified here.

        Returns
        -------
        Tensor, shape (P+1, H) — h_p^(L).
        """
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h
