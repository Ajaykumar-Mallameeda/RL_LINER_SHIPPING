"""
P7 — Reusable neural backbone: GAT stack → Transformer encoder.

Assembles the encoder of [PAPER] Section 4.1:

    Graph state (f_p, f_e)
        ↓  GAT × 3                              [PAPER] Eqs. 7-8, Table 5
    h_p^(L) ∈ R^((P+1)×H) = [h_p,port, h_p,global]   [PAPER] Eq. 9
        ↓  vessel encoding                      [PAPER] Eq. 10 / Eq. 16
        ↓  Transformer × 3, 8 heads             [PAPER] Eq. 11, Table 5
    [ĥ_p, ĥ_v]                                  [PAPER] Eq. 11

This is the SHARED backbone. It makes no policy decision and consumes no
reward. P8 attaches a sigmoid head; P9 attaches an LSTM decoder. Neither
duplicates the GAT/Transformer layers.

Scope boundaries enforced in this module:
  * No PPO, no advantage, no value head, no optimizer, no training loop.
  * No service generation — the backbone never constructs a ServiceAction.
  * No action sampling — the backbone emits embeddings and logits only.

Variable graph sizes (P7.7):
  P, E and V are read from the input tensors on every forward pass. Nothing is
  cached from construction, so the same module instance serves Baltic (12
  ports) and WorldLarge (201 ports) without reconfiguration. The only
  dimension fixed at construction is the input FEATURE width (D_in = 2 for
  nodes, 4 + 2 + |S|_max for edges), which is a property of P5's schema rather
  than of any instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .config import PAPER_ARCHITECTURE, ArchitectureConfig
from .gat import GATStack
from .masks import to_pytorch_padding_mask
from .tensors import GraphTensors, neural_state_to_tensors
from .transformer import EncoderOutput, TransformerEncoderStack, joint_encoder_forward


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class BackboneOutput:
    """
    Everything the policy layers need from the P7 encoder.

    Attributes
    ----------
    port_embeddings : Tensor, shape (P, H)
        ĥ_p — contextualised port embeddings. [PAPER] Eq. 12 (encoder-only) and
        Eq. 15 (decoder) both consume these.
    vessel_embeddings : Tensor, shape (V_sel, H)
        ĥ_v. V_sel = 1 when `selected_vessel_index` was given (encoder-only,
        Eq. 10); V when it was None (encoder-decoder, Eq. 16).
    global_embedding_gat : Tensor, shape (H,)
        h_p,global^(L) — the RAW GAT global embedding. [PAPER] Eq. 20 uses
        exactly this vector to initialise the LSTM cell state, so it is exposed
        separately from the contextualised version.
    global_embedding : Tensor or None, shape (H,)
        Contextualised global embedding, present when `include_global=True`.
    node_embeddings : Tensor, shape (P+1, H)
        Full GAT output including the global node (Eq. 9).
    tokens : Tensor, shape (P + [1] + V_sel, H)
        Post-Transformer token sequence, for callers that need the raw layout.
    encoder_output : EncoderOutput
        The structured Transformer result.
    num_ports : int
    num_nodes : int
    num_edges : int
    num_vessel_classes : int
    num_vessel_tokens : int
    """

    port_embeddings: "torch.Tensor"
    vessel_embeddings: "torch.Tensor"
    global_embedding_gat: "torch.Tensor"
    global_embedding: Optional["torch.Tensor"]
    node_embeddings: "torch.Tensor"
    tokens: "torch.Tensor"
    encoder_output: EncoderOutput
    num_ports: int
    num_nodes: int
    num_edges: int
    num_vessel_classes: int
    num_vessel_tokens: int


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class NeuralBackbone(nn.Module):
    """
    GAT × L → Transformer × M encoder, shared by both policy pathways.

    Parameters
    ----------
    config : ArchitectureConfig, optional
        Defaults to the paper configuration (H=512, 3 GAT, 3 Transformer,
        8 heads). Pass `ArchitectureConfig.tiny()` in unit tests.
    node_feature_dim : int
        D_in for node features. Default 2 ([PAPER] Eqs. 37-38).
    edge_feature_dim : int
        Width of the concatenated static+dynamic edge feature block.
        Default 4 + 2 + 100, matching P5's schema with its |S|_max = 100 cap.
    include_global : bool
        When True (default) the global node takes part in the Transformer as a
        token, so a contextualised global embedding is available to P9. The
        paper's own LSTM init uses the GAT (pre-Transformer) global embedding,
        which is exposed either way as `global_embedding_gat`.

    Notes
    -----
    Determinism: with `config.dropout == 0.0` (the default) a forward pass is a
    pure function of its inputs on CPU. No stochastic operation is used
    anywhere in this module.
    """

    def __init__(
        self,
        config: Optional[ArchitectureConfig] = None,
        node_feature_dim: int = 2,
        edge_feature_dim: int = 4 + 2 + 100,
        include_global: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or ArchitectureConfig()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.include_global = include_global

        if self.config.hidden_dim % self.config.transformer_heads != 0:
            raise ValueError(
                f"hidden_dim {self.config.hidden_dim} must be divisible by "
                f"transformer_heads {self.config.transformer_heads}."
            )

        self.gat = GATStack(
            in_dim=node_feature_dim,
            config=self.config,
            edge_dim=edge_feature_dim,
        )
        self.transformer = TransformerEncoderStack(self.config)

        # [PAPER] Eq. 10: h_v = W_v (v_t)_v ∈ R^H — a linear map from the
        # vessel feature vector to the embedding space. Reused for all vessel
        # tokens in the decoder pathway (Eq. 16 uses a distinct matrix W'_v;
        # P9 owns that separate projection so the two never share weights).
        self.vessel_projection = nn.Linear(
            self._vessel_feature_dim(), self.config.hidden_dim, bias=True,
        )

        # Layer norms keep the raw P5 features (which include un-normalised
        # columns such as quantity, panama and suez fees) in a sane range
        # before the first attention op. [ENGINEERING DECISION]
        self.node_norm = nn.LayerNorm(node_feature_dim)

    # ---- dimensions ----

    @staticmethod
    def _vessel_feature_dim() -> int:
        """D_v = 11 — [PAPER] Appendix A.1."""
        return 11

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def num_gat_layers(self) -> int:
        return self.config.gat_layers

    @property
    def num_transformer_layers(self) -> int:
        return self.config.transformer_layers

    @property
    def num_transformer_heads(self) -> int:
        return self.config.transformer_heads

    # ---- encoding ----

    def encode_graph(
        self,
        bundle: GraphTensors,
        selected_vessel_index: Optional[int] = None,
        key_padding_mask: Optional["torch.Tensor"] = None,
    ) -> BackboneOutput:
        """
        Encode a `GraphTensors` bundle into port and vessel embeddings.

        Parameters
        ----------
        bundle : GraphTensors
            Output of `neural_state_to_tensors`.
        selected_vessel_index : int, optional
            Encoder-only pathway: encode ONLY this vessel class, giving a
            single vessel token h_v ∈ R^H ([PAPER] Eq. 10). When None, encode
            every vessel class ([PAPER] Eq. 16), as the decoder needs.
        key_padding_mask : BoolTensor, optional
            Padding mask over the Transformer token sequence, in PyTorch
            convention (True = ignore). Not applied to the GAT.

        Returns
        -------
        BackboneOutput
        """
        return self._encode(
            node_features=bundle.node_features,
            edge_index=bundle.edge_index,
            edge_attr=self._edge_attr(bundle),
            vessel_features=bundle.vessel_features,
            selected_vessel_index=selected_vessel_index,
            key_padding_mask=key_padding_mask,
        )

    def encode_state(
        self,
        state: Any,
        selected_vessel_index: Optional[int] = None,
        key_padding_mask: Optional["torch.Tensor"] = None,
        bundle: Optional[GraphTensors] = None,
    ) -> BackboneOutput:
        """
        Convenience path straight from a P5 `NeuralState`.

        The numpy→torch conversion is delegated to
        `neural.tensors.neural_state_to_tensors`, so device/dtype policy and
        the edge-alignment assertion live in exactly one place.
        """
        if bundle is None:
            bundle = neural_state_to_tensors(
                state,
                config=self.config,
                device=self._device(),
                dtype=self._dtype(),
            )
        return self.encode_graph(
            bundle,
            selected_vessel_index=selected_vessel_index,
            key_padding_mask=key_padding_mask,
        )

    # ---- internals ----

    def _edge_attr(self, bundle: GraphTensors) -> "torch.Tensor":
        """
        Concatenate static and dynamic edge features along the FEATURE axis.

        P5 stores both blocks as (rows, E); GAT attention wants (features, E).
        The concatenation is column-aligned by construction, so edge i keeps
        edge i's static AND dynamic features — the P7.3 alignment invariant.
        """
        static = bundle.static_edge_features
        dynamic = bundle.dynamic_edge_features
        if static.shape[1] != dynamic.shape[1]:
            raise ValueError(
                f"static edge block has {static.shape[1]} edges, dynamic has "
                f"{dynamic.shape[1]} — edge alignment is broken."
            )
        return torch.cat([static, dynamic], dim=0)

    def _encode(
        self,
        node_features: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
        vessel_features: "torch.Tensor",
        selected_vessel_index: Optional[int],
        key_padding_mask: Optional["torch.Tensor"],
    ) -> BackboneOutput:
        """Shared implementation for the tensor-level entry points."""
        if node_features.dim() != 2 or node_features.shape[1] != self.node_feature_dim:
            raise ValueError(
                f"node_features must be (N, {self.node_feature_dim}), got "
                f"{tuple(node_features.shape)}."
            )
        if vessel_features.dim() != 2:
            raise ValueError(
                f"vessel_features must be 2-D, got "
                f"{tuple(vessel_features.shape)}."
            )
        V = int(vessel_features.shape[0])
        if V < 1:
            raise ValueError("vessel_features must contain at least one class.")

        # ---- 1. GAT stack (Eqs. 7-8) ----
        h = self.gat(self.node_norm(node_features), edge_index, edge_attr)
        # h: (P+1, H)

        # ---- 2. split off the global node (Eq. 9) ----
        # node index P is the global node, by P5's construction.
        port_h = h[:-1]                 # (P, H)
        global_h_gat = h[-1]            # (H,)

        # ---- 3. vessel encoding (Eq. 10 / Eq. 16) ----
        if selected_vessel_index is None:
            vessel_h = self.vessel_projection(vessel_features)      # (V, H)
        else:
            if not 0 <= selected_vessel_index < V:
                raise ValueError(
                    f"selected_vessel_index {selected_vessel_index} out of "
                    f"range for {V} vessel classes."
                )
            selected = vessel_features[selected_vessel_index]
            vessel_h = self.vessel_projection(selected).unsqueeze(0)  # (1, H)

        # ---- 4. joint Transformer (Eq. 11) ----
        enc = joint_encoder_forward(
            stack=self.transformer,
            port_embeddings=port_h,
            vessel_embeddings=vessel_h,
            global_embedding=global_h_gat if self.include_global else None,
            key_padding_mask=key_padding_mask,
        )

        return BackboneOutput(
            port_embeddings=enc.port_embeddings,
            vessel_embeddings=enc.vessel_embeddings,
            global_embedding_gat=global_h_gat,
            global_embedding=enc.global_embedding,
            node_embeddings=h,
            tokens=enc.tokens,
            encoder_output=enc,
            num_ports=int(port_h.shape[0]),
            num_nodes=int(h.shape[0]),
            num_edges=int(edge_index.shape[1]),
            num_vessel_classes=V,
            num_vessel_tokens=int(vessel_h.shape[0]),
        )

    # ---- device / dtype plumbing ----

    def _device(self) -> "torch.device":
        """Device of this module's parameters (never hard-coded CUDA)."""
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover - module always has params
            return torch.device("cpu")

    def _dtype(self) -> "torch.dtype":
        """Floating dtype of this module's parameters."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover
            return torch.float32

    def parameter_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> dict:
        """Machine-readable description of the assembled architecture."""
        return {
            "hidden_dim": self.config.hidden_dim,
            "gat_layers": self.config.gat_layers,
            "gat_heads": self.config.gat_heads,
            "transformer_layers": self.config.transformer_layers,
            "transformer_heads": self.config.transformer_heads,
            "transformer_ffn_dim": self.config.transformer_ffn_dim,
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "vessel_feature_dim": self._vessel_feature_dim(),
            "include_global": self.include_global,
            "dropout": self.config.dropout,
            "device": str(self._device()),
            "dtype": str(self._dtype()),
            "parameters": self.parameter_count(),
            "matches_paper": self.config.matches_paper(),
        }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_backbone(backbone: NeuralBackbone, path: str) -> None:
    """
    Save a backbone's state_dict plus the config needed to rebuild it.

    This is ARCHITECTURE-level serialization only — no optimizer state, no
    training step counter, no checkpoints (P10 owns those).

    torch.save is used (pickle-based) rather than safetensors because the
    payload is architecture metadata plus tensors we produced ourselves.
    """
    payload = {
        "config": backbone.config.to_dict(),
        "node_feature_dim": backbone.node_feature_dim,
        "edge_feature_dim": backbone.edge_feature_dim,
        "include_global": backbone.include_global,
        "state_dict": backbone.state_dict(),
        "paper_architecture": PAPER_ARCHITECTURE,
    }
    torch.save(payload, path)


def load_backbone(path: str, map_location: Any = "cpu") -> NeuralBackbone:
    """
    Rebuild a backbone saved by `save_backbone`, with identical parameters.

    Raises
    ------
    ValueError
        If the payload is missing required keys or the stored config is invalid.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Malformed backbone file: expected a dict payload.")
    for key in ("config", "state_dict"):
        if key not in payload:
            raise ValueError(f"Malformed backbone file: missing {key!r}.")

    config = ArchitectureConfig(**payload["config"])
    backbone = NeuralBackbone(
        config=config,
        node_feature_dim=payload.get("node_feature_dim", 2),
        edge_feature_dim=payload.get("edge_feature_dim", 4 + 2 + 100),
        include_global=payload.get("include_global", True),
    )
    backbone.load_state_dict(payload["state_dict"])
    backbone.eval()
    return backbone
