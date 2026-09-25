"""
P13 — Inference / Solver Layer for LSNDP RL Engine.

Builds a production-quality inference layer around the validated P8 (encoder-only)
and P9 (encoder-decoder) policies. Loads a trained checkpoint and repeatedly
constructs services until the environment terminates.

Architecture:
    Checkpoint loader
        ↓
    Model construction + architecture validation
        ↓
    Environment + state encoding
        ↓
    Policy inference (deterministic / stochastic)
        ↓
    Service execution through P4 env
        ↓
    Final MCF evaluation through P3
        ↓
    InferenceResult serialization

Scope boundaries enforced:
  * No training, no PPO, no optimizer updates.
  * Checkpoint compatibility checked — no silent resizing or partial loading.
  * Termination semantics follow P4 exactly (terminated vs truncated distinction).
  * Final objective is always the authoritative P2/P3 formulation.

Evidence tags throughout document paper vs implementation decisions.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver import InferenceSolver
    from .result import InferenceResult

__all__ = [
    "InferenceSolver",
    "InferenceResult",
    "InferenceConfig",
]

from .config import InferenceConfig
from .result import InferenceResult
from .solver import InferenceSolver
from .checkpoint import load_and_validate_checkpoint
