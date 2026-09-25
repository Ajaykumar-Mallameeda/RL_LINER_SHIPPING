"""
P5 — Neural state representation package.

Exports the public API for converting P4 environment state into a
GAT/Transformer-ready tensor representation.
"""

from .representation import (
    NeuralState,
    ServiceMembership,
    StateEncoder,
    build_index_mappings,
)

__all__ = [
    "NeuralState",
    "ServiceMembership",
    "StateEncoder",
    "build_index_mappings",
]
