"""
P3 – Greedy MCF demand routing and flow solver.

Implements the paper's greedy heuristic (Algorithm 1, Appendix B):
  - Process demands in descending revenue order [PAPER]
  - For each demand, find shortest path in expanded graph by marginal handling cost
  - Route min(remaining_demand, bottleneck_capacity) along that path
  - Update residual capacities
  - Accumulate rejected demand

Deterministic tie-breaking:
  - [IMPLEMENTATION] Demand tie-break: secondary sort by (origin, destination).
  - [IMPLEMENTATION] Shortest-path tie-break: heapq pops lexicographically
    smallest (distance, sort_key) where sort_key is a deterministic string.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data.instance import DistanceArc, LINERLIBInstance

from .expanded_graph import ExpandedGraph, ProxyNode, ServiceDefinition
from .result import CommodityResult, MCFResult
from .costs import CostCalculator


# ---------------------------------------------------------------------------
# Helper: deterministic node sort key for Dijkstra tie-breaking
# ---------------------------------------------------------------------------

def _node_sort_key(node) -> str:
    """
    Deterministic string key for tie-breaking in Dijkstra.
    ProxyNodes sort by (port_code, service_id); physical ports by UNLOCODE.
    [IMPLEMENTATION] Resolves OQ-5: deterministic shortest-path tie-break.
    """
    if isinstance(node, ProxyNode):
        return f"{node.port_code}|{node.service_id:06d}"
    return str(node)


def _node_key(node) -> str:
    """Convert a graph node to a string key for flow tracking."""
    if isinstance(node, ProxyNode):
        return f"{node.port_code}[s{node.service_id}]"
    return str(node)


# ---------------------------------------------------------------------------
# Dijkstra with residual capacity constraints
# ---------------------------------------------------------------------------

def _dijkstra_residual(
    source,
    target,
    capacities,
    edge_weights,
    adj,
) -> Optional[Tuple[List, float]]:
    """
    Modified Dijkstra respecting edge residual capacities.

    Only traverses edges with positive residual capacity.
    Returns (path_nodes, total_cost) or None if no feasible path exists.

    Deterministic tie-break via (cost, sort_key) tuple in priority queue.
    """
    dist: Dict = {source: 0.0}
    prev: Dict = {source: None}
    # Priority: (cost, sort_key_string, node)
    pq: List = [(0.0, _node_sort_key(source), source)]
    visited: set = set()

    while pq:
        d, _sk, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, d

        for v, eidx in adj.get(u, []):
            if v in visited:
                continue
            res_cap = capacities.get((u, v), 0.0)
            if res_cap <= 1e-12:
                continue
            w = edge_weights[eidx]
            nd = d + w
            if v not in dist or nd < dist[v] - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, _node_sort_key(v), v))

    return None


# ---------------------------------------------------------------------------
# Flow Solver
# ---------------------------------------------------------------------------

class FlowSolver:
    """
    Greedy MCF solver operating on the expanded graph.

    Implements Algorithm 1 from the paper (Appendix B).
    """

    def __init__(
        self,
        instance: LINERLIBInstance,
        services: List[ServiceDefinition],
        vessel_requirements: Dict[str, Dict[str, float]],
    ) -> None:
        self._instance = instance
        self._services = services
        self._vreqs = vessel_requirements

    def solve(self) -> MCFResult:
        """Execute the full greedy MCF evaluation."""
        t0 = time.perf_counter()

        # Build distance lookup
        dist_by_pair: Dict[Tuple[str, str], DistanceArc] = {}
        for arc in self._instance.distances:
            dist_by_pair[(arc.origin, arc.destination)] = arc

        # Build expanded graph
        eg = ExpandedGraph(
            instance=self._instance,
            services=self._services,
            distances_by_pair=dist_by_pair,
            vessel_types=self._instance.vessel_types,
            vessel_requirements=self._vreqs,
        )
        eg.build()

        # Prepare edge data structures
        edges = eg.edges
        edge_weights = [e.weight for e in edges]
        adj = dict(eg.adj)
        num_edges = len(edges)

        # Residual capacity map: (u, v) -> remaining
        residual: Dict = {}
        for e in edges:
            residual[(e.u, e.v)] = e.capacity

        # Sort commodities by descending revenue [PAPER], deterministic tie-break
        commodities = sorted(
            enumerate(self._instance.demands),
            key=lambda x: (-x[1].revenue, x[1].origin, x[1].destination),
        )

        # Track results per commodity
        routed_per_commodity: Dict[int, float] = {}
        flows: Dict[int, Dict[str, float]] = {}
        commodity_results: List[CommodityResult] = []

        for cmd_idx, dem in commodities:
            remaining = dem.ffe_per_week
            total_satisfied = 0.0
            routes_count = 0
            last_path = []
            last_bottleneck = 0.0

            while remaining > 1e-12:
                path = _dijkstra_residual(
                    source=dem.origin,
                    target=dem.destination,
                    capacities=residual,
                    edge_weights=edge_weights,
                    adj=adj,
                )

                if path is None:
                    break

                path_nodes, _cost = path
                last_path = path_nodes

                # Compute bottleneck
                bottleneck = float("inf")
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    cap = residual.get((u, v), 0.0)
                    bottleneck = min(bottleneck, cap)

                if bottleneck <= 1e-12:
                    break

                # Route flow
                flow = min(remaining, bottleneck)
                if flow <= 1e-12:
                    break

                # Update residual capacities
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    residual[(u, v)] -= flow

                remaining -= flow
                total_satisfied += flow
                routes_count += 1

                # Store flow for cost calculation
                if cmd_idx not in flows:
                    flows[cmd_idx] = {}
                for i in range(len(path_nodes) - 1):
                    key = f"{_node_key(path_nodes[i])}-->{_node_key(path_nodes[i+1])}"
                    flows[cmd_idx][key] = flows[cmd_idx].get(key, 0.0) + flow

                # Guard against infinite loop (shouldn't happen, but safety check)
                if routes_count > len(self._instance.demands) * 100:
                    break

            last_bottleneck = bottleneck if routes_count > 0 else 0.0
            satisfied = total_satisfied
            rejected = max(0.0, dem.ffe_per_week - satisfied)
            routed_per_commodity[cmd_idx] = satisfied

            commodity_results.append(CommodityResult(
                commodity_idx=cmd_idx,
                origin=dem.origin,
                destination=dem.destination,
                demand_ffe=dem.ffe_per_week,
                revenue_per_ffe=dem.revenue,
                satisfied=satisfied,
                rejected=rejected,
                path_nodes=[_node_key(n) for n in last_path],
                path_cost=0.0,  # computed by cost calculator separately
                bottleneck_capacity=last_bottleneck,
                routes_count=routes_count,
            ))

        # Compute all cost components
        timing = time.perf_counter() - t0
        cost_calc = CostCalculator(self._instance, dist_by_pair)
        cost_data = cost_calc.compute_all(
            services=self._services,
            vessel_requirements=self._vreqs,
            flows=flows,
            routed_per_commodity=routed_per_commodity,
            expanded_graph=eg,
        )

        total_demand = sum(d.ffe_per_week for d in self._instance.demands)
        routed_demand = sum(routed_per_commodity.values())

        result = MCFResult(
            eta=cost_data["eta"],
            total_revenue=cost_data["total_revenue"],
            rejected_demand=cost_data["rejected_demand"],
            rejection_cost=cost_data["rejection_cost"],
            handling_cost=cost_data["handling_cost"],
            loading_unloading_cost=cost_data["loading_unloading_cost"],
            transshipment_cost=cost_data["transshipment_cost"],
            service_cost=cost_data["service_cost"],
            unused_vessel_cost=cost_data["unused_vessel_cost"],
            voyage_cost=cost_data["voyage_cost"],
            port_call_cost=cost_data["port_call_cost"],
            sailing_fuel_cost=cost_data["sailing_fuel_cost"],
            idle_fuel_cost=cost_data["idle_fuel_cost"],
            canal_fee_cost=cost_data["canal_fee_cost"],
            total_demand=total_demand,
            routed_demand=routed_demand,
            demand_coverage=routed_demand / total_demand if total_demand > 0 else 0.0,
            commodity_results=commodity_results,
            vessel_requirements=self._vreqs,
            raw_flows=flows,
            expanded_graph_node_count=eg.num_nodes,
            expanded_graph_edge_count=eg.num_edges,
            num_services=len(self._services),
            num_demands=len(self._instance.demands),
            instance_name=self._instance.name,
            timing_seconds=timing,
        )

        return result
