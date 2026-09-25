"""
P3 – Multi-Commodity Flow (MCF) / Network Evaluation Engine.

Implements the paper-faithful greedy MCF heuristic for evaluating a candidate
liner shipping network (set of services) against an instance's demand.

Source hierarchy:
  1. Paper (Dutta et al. 2024, arXiv:2411.09068) – Algorithm 1, Appendix B,
     Figures 3-4, Sections 3 & 5.
  2. P2 mathematical contract – docs/PROBLEM_FORMULATION.md.
  3. P1 LINERLIB data API – data.instance, data.linerlib_loader.

Prohibited: no RL, no Gymnasium, no training, no model code.
"""

from __future__ import annotations

from .expanded_graph import ExpandedGraph, ProxyNode, ServiceDefinition
from .flow_evaluator import evaluate_network
from .flow_solver import FlowSolver
from .result import CommodityResult, MCFResult

__all__ = [
    "ExpandedGraph",
    "ProxyNode",
    "ServiceDefinition",
    "FlowSolver",
    "MCFResult",
    "CommodityResult",
    "evaluate_network",
]
