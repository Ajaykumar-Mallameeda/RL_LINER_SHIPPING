"""
P10 — Proximal Policy Optimization (PPO) Training Engine for LSNDP.

Implements the PPO trainer required to train P8 and P9 policies against
the P4 environment. P10 owns:
  - rollout collection
  - trajectory storage
  - reward accumulation
  - return calculation
  - advantage estimation (GAE)
  - PPO policy loss with clipping
  - value function (critic)
  - entropy bonus
  - KL diagnostics
  - optimizer integration
  - minibatching
  - PPO epochs
  - training/update loop
  - checkpointing

This module does NOT own:
  - environment mechanics (P4)
  - MCF evaluation (P3)
  - state representation (P5)
  - service generation (P6)
  - neural backbone architecture (P7)
  - policy architectures (P8, P9)
  - benchmark experiments

Evidence tags document every non-paper-specified choice.
"""

from __future__ import annotations

from .adapter import RolloutBatch, build_rollout_batch
from .buffer import PPOBuffer, TrajectoryStep
from .config import PPOConfig, PaperPPOConfig
from .critic import ValueFunction, build_critic_from_graph
from .trainer import PPOTrainer, PDiagnostics

__all__ = [
    "PPOBuffer",
    "TrajectoryStep",
    "PPOConfig",
    "PaperPPOConfig",
    "ValueFunction",
    "build_critic_from_graph",
    "PPOTrainer",
    "PDiagnostics",
    "RolloutBatch",
    "build_rollout_batch",
]
