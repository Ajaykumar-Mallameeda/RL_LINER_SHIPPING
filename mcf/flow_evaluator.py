"""
P3 – High-level flow evaluator wrapper.

Provides the public API ``evaluate_network(...)`` that takes a LINERLIBInstance
and a candidate network (list of services) and returns an MCFResult.
"""

from __future__ import annotations

from typing import Dict, List

from data.instance import LINERLIBInstance

from .expanded_graph import ServiceDefinition
from .flow_solver import FlowSolver
from .result import MCFResult


def evaluate_network(
    instance: LINERLIBInstance,
    services: List[ServiceDefinition],
    vessel_requirements: Dict[str, Dict[str, float]],
) -> MCFResult:
    """
    Evaluate a candidate liner network through the greedy MCF heuristic.

    Parameters
    ----------
    instance :
        A loaded LINERLIBInstance (P1 data foundation). Never mutated.
    services :
        List of ServiceDefinition describing the candidate network.
    vessel_requirements :
        vessel_requirements[service_id][vessel_class] = n_vs
        (fractional vessel count required for each service).

    Returns
    -------
    MCFResult
        Structured result with all cost components and diagnostics.

    Example
    -------
    >>> from mcf import evaluate_network, ServiceDefinition
    >>> result = evaluate_network(instance, [svc], {"svc_0": {"V1": 2.5}})
    >>> print(result.eta)
    """
    solver = FlowSolver(
        instance=instance,
        services=services,
        vessel_requirements=vessel_requirements,
    )
    return solver.solve()
