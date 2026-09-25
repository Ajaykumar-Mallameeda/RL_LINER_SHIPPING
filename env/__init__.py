"""
P4 — RL Environment for Liner Shipping Network Design Problem (LSNDP).

Implements a Gymnasium-compatible environment that wraps the P3 MCF evaluator.
Each valid action adds one complete liner service to the network.

The environment maintains:
  - Current network state (services, vessel requirements)
  - Remaining demand state
  - Remaining fleet state
  - Profit history (η_0, η_1, ..., η_t)
  - Reward calculation (raw incremental + normalized)

Source-of-truth hierarchy:
  1. docs/PAPER_METHOD_SPECIFICATION.md
  2. docs/PROBLEM_FORMULATION.md (P2 specification)
  3. data.instance (P1 data model)
  4. mcf (P3 MCF evaluator)

Prohibited: no GAT, Transformer, LSTM, PPO, training loops, or learned policies.
"""

from __future__ import annotations

from .environment import LSNDPEnv, ServiceAction

__all__ = ["LSNDPEnv", "ServiceAction"]
