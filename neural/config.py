"""
P7 — Neural architecture configuration.

Separates two kinds of values:

  * PAPER-FROZEN architectural constants (Table 5 of Dutta et al., 2024).
    These MUST NOT be changed silently. `ArchitectureConfig` defaults to them
    and `test_neural_architecture.py` asserts against the paper values.

  * ENGINEERING KNOBS for details the paper does not specify (activation,
    residual placement, feed-forward expansion, dropout). These are marked
    [ENGINEERING DECISION] and documented in docs/P7_FINAL_REPORT.md.

Evidence tags:
  [PAPER]               — value stated in the target paper
  [IMPLEMENTATION]      — chosen implementation of a paper-specified structure
  [ENGINEERING DECISION]— choice made where the paper is silent
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Frozen paper values
# ---------------------------------------------------------------------------

# [PAPER] Table 5, "Hyper-parameter settings for encoder / decoder networks
# and PPO". Every value below is read directly from the paper's table.
#
#   Hidden layer        — number of neurons in the hidden layer    → 512
#   Transformer head    — number of attention heads in transformer → 8
#   Transformer layer   — number of transformer layers             → 3
#   GAT layer           — number of Graph Attention Network layers → 3
#   LSTM layer          — number of LSTM layers                    → 1
PAPER_ARCHITECTURE: Dict[str, int] = {
    "hidden_dim": 512,
    "transformer_heads": 8,
    "transformer_layers": 3,
    "gat_layers": 3,
    "lstm_layers": 1,
}

# [PAPER] Section 2.3 / Appendix A.1: D_v = 11 vessel features per class.
PAPER_VESSEL_FEATURE_DIM: int = 11

# [PAPER] Section 2.1: p_i ∈ R^2 per port node (incoming, outgoing demand).
PAPER_NODE_FEATURE_DIM: int = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchitectureConfig:
    """
    Configuration for the P7 neural backbone.

    Paper-frozen values default to PAPER_ARCHITECTURE. Changing them requires
    an explicit P0 revision: the test-suite asserts the defaults match the
    paper's Table 5.

    Parameters
    ----------
    hidden_dim : int
        [PAPER] Embedding dimension H = 512.
    gat_layers : int
        [PAPER] Number of stacked GAT layers L = 3.
    transformer_layers : int
        [PAPER] Number of Transformer encoder layers = 3.
    transformer_heads : int
        [PAPER] Number of attention heads in the Transformer = 8.
    lstm_layers : int
        [PAPER] Number of LSTM layers in the decoder = 1 (consumed by P9).
    gat_heads : int
        [ENGINEERING DECISION] Number of heads per GAT layer. The paper gives
        a head count only for the Transformer; the GAT head count is not
        specified. Default 1 (single-head, matching the original GAT paper's
        formulation of Eq. 7-8 without a multi-head suffix).
    gat_activation : str
        [ENGINEERING DECISION] Activation applied to each GAT layer output.
        The paper does not state the GAT activation. Default "elu" (the
        original GAT paper's choice).
    gat_edge_features : str
        [ENGINEERING DECISION] How edge features enter GAT attention:
          "additive" — projected onto the attention logit (Gong & Cheng style).
        The paper states only that node features are transformed while edge
        features "remain unchanged" (i.e. they are not re-embedded through the
        stack). They must still reach the attention mechanism, otherwise Eqs.
        7-8 would carry no edge information at all.
    transformer_ffn_multiplier : int
        [ENGINEERING DECISION] Transformer feed-forward expansion ratio.
        Not specified by the paper. Default 4 (standard Transformer).
    transformer_activation : str
        [ENGINEERING DECISION] Transformer FFN activation. Default "relu"
        (the paper uses ReLU for the decoder FF layer, so ReLU is consistent).
    dropout : float
        [ENGINEERING DECISION] Dropout probability. Not specified. Default 0.0
        so that forward passes are exactly reproducible without train/eval
        mode bookkeeping.
    device : str
        [ENGINEERING DECISION] "cpu" (default) or "cuda". CPU is the
         determinism reference; CUDA is exposed but not required by P7.
    dtype : str
        [ENGINEERING DECISION] "float32" (default) or "float64".
    """

    # ---- paper-frozen ----
    hidden_dim: int = PAPER_ARCHITECTURE["hidden_dim"]
    gat_layers: int = PAPER_ARCHITECTURE["gat_layers"]
    transformer_layers: int = PAPER_ARCHITECTURE["transformer_layers"]
    transformer_heads: int = PAPER_ARCHITECTURE["transformer_heads"]
    lstm_layers: int = PAPER_ARCHITECTURE["lstm_layers"]

    # ---- engineering knobs ----
    gat_heads: int = 1
    gat_activation: str = "elu"
    gat_edge_features: str = "additive"
    gat_negative_slope: float = 0.2
    transformer_ffn_multiplier: int = 4
    transformer_activation: str = "relu"
    dropout: float = 0.0
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on an inconsistent configuration."""
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.gat_layers < 1:
            raise ValueError(f"gat_layers must be >= 1, got {self.gat_layers}")
        if self.transformer_layers < 1:
            raise ValueError(
                f"transformer_layers must be >= 1, got {self.transformer_layers}"
            )
        if self.lstm_layers < 1:
            raise ValueError(f"lstm_layers must be >= 1, got {self.lstm_layers}")
        if self.transformer_heads < 1:
            raise ValueError(
                f"transformer_heads must be >= 1, got {self.transformer_heads}"
            )
        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"transformer_heads ({self.transformer_heads})"
            )
        if self.gat_heads < 1:
            raise ValueError(f"gat_heads must be >= 1, got {self.gat_heads}")
        if self.hidden_dim % self.gat_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"gat_heads ({self.gat_heads})"
            )
        if self.transformer_ffn_multiplier < 1:
            raise ValueError(
                "transformer_ffn_multiplier must be >= 1, got "
                f"{self.transformer_ffn_multiplier}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.gat_edge_features not in ("additive", "none"):
            raise ValueError(
                "gat_edge_features must be 'additive' or 'none', got "
                f"{self.gat_edge_features!r}"
            )
        if self.dtype not in ("float32", "float64"):
            raise ValueError(
                f"dtype must be 'float32' or 'float64', got {self.dtype!r}"
            )

    # ---- derived ----

    @property
    def gat_head_dim(self) -> int:
        """Per-head dimension of each GAT layer."""
        return self.hidden_dim // self.gat_heads

    @property
    def transformer_ffn_dim(self) -> int:
        """Feed-forward inner dimension of each Transformer layer."""
        return self.hidden_dim * self.transformer_ffn_multiplier

    def matches_paper(self) -> bool:
        """True if every paper-frozen value equals the paper's Table 5."""
        return all(
            getattr(self, key) == value
            for key, value in PAPER_ARCHITECTURE.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict of all configuration values."""
        return asdict(self)

    @classmethod
    def tiny(cls, **overrides: Any) -> "ArchitectureConfig":
        """
        A small, fast configuration for unit tests.

        NOT the paper architecture — used only to keep tests fast. Tests that
        assert paper dimensions use the default constructor.
        """
        base: Dict[str, Any] = {
            "hidden_dim": 16,
            "gat_layers": 2,
            "transformer_layers": 2,
            "transformer_heads": 2,
            "lstm_layers": 1,
            "gat_heads": 1,
            "transformer_ffn_multiplier": 2,
        }
        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------
# Default instance (paper-faithful)
# ---------------------------------------------------------------------------

def paper_config(**overrides: Any) -> ArchitectureConfig:
    """
    Return a configuration carrying the paper's frozen architecture values.

    Any override that changes a paper-frozen value is applied but recorded in
    `.deviations` by the caller; `matches_paper()` reports whether the result
    still conforms to Table 5.
    """
    return ArchitectureConfig(**overrides)
