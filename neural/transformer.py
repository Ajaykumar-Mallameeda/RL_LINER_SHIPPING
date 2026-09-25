"""
P7 — Transformer encoder over the joint port/vessel token sequence.

Implements [PAPER] Eq. 11:

    [ĥ_p, ĥ_v] = Transformer(h_p,port^(L), h_v) ∈ R^((P+1)×H)

Key paper-specified properties:

  * Positional embeddings are OMITTED — "our embedding structure does not have
    inherent temporal properties" [PAPER, Section 4.1]. No position encoding
    is added anywhere in this module.
  * Configuration: 3 layers, 8 heads, H = 512 [PAPER] Table 5.
  * The encoder produces contextualised port AND vessel embeddings. In the
    encoder-only pathway the vessel token is the selected class (Eq. 10); in
    the encoder-decoder pathway it is every class (Eq. 16). Both are just a
    different number of vessel tokens, so one implementation serves both.

Token ordering (deterministic, and the reason this is safe):
    positions [0 .. P-1]  → physical ports, in P5 node order (alphabetical)
    position  [P]         → the global node, when include_global=True
    then                  → vessel tokens, in P5 vessel-class order

Nothing is keyed by a dict, so two runs over the same state always produce the
same token order. P8/P9 read the split back out by slicing on the known
boundaries returned in `EncoderOutput`.

Implementation choices where the paper is silent are tagged
[ENGINEERING DECISION]:

  * Transformer feed-forward expansion 4×, activation ReLU (the paper uses ReLU
    in the decoder FF layer, so ReLU is the consistent choice).
  * Post-LN layer ordering (PyTorch's standard `nn.TransformerEncoderLayer`
    default), i.e. the "standard Transformer encoding layer" of [PAPER].
  * Dropout is a config knob defaulting to 0.0 so forward passes are exactly
    reproducible without train/eval bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import ArchitectureConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """
    Result of a joint port/vessel Transformer pass.

    The slices recorded here are authoritative — downstream P8/P9 code should
    use them rather than recomputing offsets.

    Attributes
    ----------
    tokens : Tensor, shape (N_tokens, H)
        Full contextualised token sequence, in the order described above.
    port_embeddings : Tensor, shape (P, H)
        Contextualised physical-port embeddings ĥ_p. [PAPER] Eq. 12 consumes
        this (the global node is excluded).
    global_embedding : Tensor or None, shape (H,)
        Contextualised global-node embedding. Present only when the global
        token was included. [PAPER] Eq. 20 uses the GAT global embedding as the
        LSTM cell-state init; this contextualised version is exposed for P9.
    vessel_embeddings : Tensor, shape (V_sel, H)
        Contextualised vessel embeddings ĥ_v. V_sel = 1 for encoder-only
        (Eq. 10) and V for encoder-decoder (Eq. 16).
    num_ports : int
        P.
    has_global : bool
        Whether the global token participated.
    num_vessel_tokens : int
        V_sel.
    """

    tokens: "torch.Tensor"
    port_embeddings: "torch.Tensor"
    global_embedding: Optional["torch.Tensor"]
    vessel_embeddings: "torch.Tensor"
    num_ports: int
    has_global: bool
    num_vessel_tokens: int


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

class TransformerEncoderStack(nn.Module):
    """
    L-layer Transformer encoder with no positional encoding.

    Wraps `nn.TransformerEncoderLayer` with `batch_first=True` and the paper's
    head/FFN configuration. Batching over the leading token dimension is
    supported by the underlying module, which is what P7.8's batching interface
    relies on.

    Parameters
    ----------
    config : ArchitectureConfig
        Supplies hidden_dim, transformer_layers, transformer_heads,
        transformer_ffn_multiplier, transformer_activation, dropout.
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.transformer_layers
        self.num_heads = config.transformer_heads

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.dropout,
            activation=config.transformer_activation,
            batch_first=True,
            norm_first=False,  # [ENGINEERING DECISION] post-LN, PyTorch default
        )
        # enable_nested_tensor=False keeps behaviour identical whether or not a
        # padding mask is supplied (the fast path is not available for all
        # device/dtype combinations, and we want one deterministic code path).
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        tokens: "torch.Tensor",
        key_padding_mask: Optional["torch.Tensor"] = None,
        attn_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Parameters
        ----------
        tokens : Tensor, shape (B, N_tokens, H) or (N_tokens, H)
        key_padding_mask : BoolTensor, optional
            Shape (B, N_tokens); True marks a PADDING position (ignored).
            This is the PyTorch convention — note it is inverted relative to
            the "True = keep" convention used by `neural.masks`.
        attn_mask : Tensor, optional
            Shape (N_tokens, N_tokens) additive or boolean attention mask.

        Returns
        -------
        Tensor with the same shape as `tokens`.
        """
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"TransformerEncoderStack expects {self.hidden_dim} features, "
                f"got {tokens.shape[-1]}."
            )
        if tokens.dim() not in (2, 3):
            raise ValueError(
                f"tokens must be 2-D or 3-D, got shape {tuple(tokens.shape)}."
            )
        return self.encoder(
            tokens,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )


# ---------------------------------------------------------------------------
# Joint port/vessel encoding
# ---------------------------------------------------------------------------

def joint_encoder_forward(
    stack: TransformerEncoderStack,
    port_embeddings: "torch.Tensor",
    vessel_embeddings: "torch.Tensor",
    global_embedding: Optional["torch.Tensor"] = None,
    key_padding_mask: Optional["torch.Tensor"] = None,
) -> EncoderOutput:
    """
    Run the Transformer jointly over port tokens and vessel tokens.

    Implements the token construction implied by [PAPER] Eq. 11 (and Eqs. 15-17
    for the decoder variant). Returns the split-back result.

    Parameters
    ----------
    stack : TransformerEncoderStack
    port_embeddings : Tensor, shape (P, H)
        h_p,port^(L) from the GAT stack ([PAPER] Eq. 9, global node excluded).
    vessel_embeddings : Tensor, shape (V_sel, H)
        h_v from Eq. 10 (V_sel = 1) or Eq. 16 (V_sel = V).
    global_embedding : Tensor, optional, shape (H,)
        h_p,global^(L). When supplied it is inserted as a token between the
        ports and the vessels, so the Transformer can contextualise it; the
        paper reuses the GAT global embedding for the LSTM init (Eq. 20), and
        exposing a contextualised variant gives P9 the choice.
    key_padding_mask : BoolTensor, optional
        (N_tokens,) padding mask in PyTorch convention (True = pad/ignore).

    Returns
    -------
    EncoderOutput
    """
    if port_embeddings.dim() != 2 or vessel_embeddings.dim() != 2:
        raise ValueError(
            "port_embeddings and vessel_embeddings must be 2-D, got "
            f"{tuple(port_embeddings.shape)} and "
            f"{tuple(vessel_embeddings.shape)}."
        )
    if port_embeddings.shape[1] != stack.hidden_dim:
        raise ValueError(
            f"port_embeddings feature dim {port_embeddings.shape[1]} != "
            f"hidden_dim {stack.hidden_dim}."
        )
    if vessel_embeddings.shape[1] != stack.hidden_dim:
        raise ValueError(
            f"vessel_embeddings feature dim {vessel_embeddings.shape[1]} != "
            f"hidden_dim {stack.hidden_dim}."
        )
    if vessel_embeddings.shape[0] < 1:
        raise ValueError("vessel_embeddings must contain at least one token.")

    P = int(port_embeddings.shape[0])
    V_sel = int(vessel_embeddings.shape[0])

    parts = [port_embeddings]
    has_global = global_embedding is not None
    if has_global:
        if global_embedding.dim() != 1 or global_embedding.shape[0] != stack.hidden_dim:
            raise ValueError(
                "global_embedding must be 1-D of length hidden_dim, got "
                f"{tuple(global_embedding.shape)}."
            )
        parts.append(global_embedding.unsqueeze(0))
    parts.append(vessel_embeddings)

    tokens = torch.cat(parts, dim=0)          # (P + [1] + V_sel, H)
    n_tokens = tokens.shape[0]

    if key_padding_mask is not None:
        if key_padding_mask.shape[-1] != n_tokens:
            raise ValueError(
                f"key_padding_mask has {key_padding_mask.shape[-1]} entries "
                f"but the token sequence has {n_tokens}."
            )
        # This function adds the batch dimension itself, so a 1-D mask must be
        # promoted to (1, n_tokens) to match. PyTorch's fast path determines
        # the mask type from the mask's rank, so leaving it 1-D would make it
        # misinterpret the mask as an attention mask and fail on the shape.
        if key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        elif key_padding_mask.dim() != 2 or key_padding_mask.shape[0] != 1:
            raise ValueError(
                "key_padding_mask must be 1-D (n_tokens,) or 2-D (1, "
                f"n_tokens); got shape {tuple(key_padding_mask.shape)}."
            )

    out = stack(tokens.unsqueeze(0), key_padding_mask=key_padding_mask)
    out = out.squeeze(0)

    # ---- split back, using the same boundaries used to build the sequence ----
    port_out = out[:P]
    cursor = P
    global_out = None
    if has_global:
        global_out = out[cursor]
        cursor += 1
    vessel_out = out[cursor:cursor + V_sel]

    return EncoderOutput(
        tokens=out,
        port_embeddings=port_out,
        global_embedding=global_out,
        vessel_embeddings=vessel_out,
        num_ports=P,
        has_global=has_global,
        num_vessel_tokens=V_sel,
    )
