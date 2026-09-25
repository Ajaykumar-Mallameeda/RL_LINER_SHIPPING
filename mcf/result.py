"""
P3 – MCF Result data structures.

Provides a structured, diagnostic-rich result object that exposes every
cost component independently for later debugging and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CommodityResult:
    """Result for a single demand commodity."""

    commodity_idx: int
    origin: str
    destination: str
    demand_ffe: float
    revenue_per_ffe: float
    satisfied: float  # FFE/week actually routed
    rejected: float  # FFE/week not satisfied
    path_nodes: List[str] = field(default_factory=list)
    path_edges: List[tuple] = field(default_factory=list)
    path_cost: float = 0.0  # marginal cost per FFE for this routing
    bottleneck_capacity: float = 0.0
    routes_count: int = 0  # number of augmentation iterations

    @property
    def revenue_contribution(self) -> float:
        return self.revenue_per_ffe * self.satisfied


@dataclass
class MCFResult:
    """
    Complete result of an MCF evaluation run.

    Exposes all cost components separately so P4 can access them without
    recomputation.
    """

    # --- scalar metrics ---
    eta: float = 0.0  # total network profit
    total_revenue: float = 0.0
    rejected_demand: float = 0.0
    rejection_cost: float = 0.0
    handling_cost: float = 0.0
    loading_unloading_cost: float = 0.0
    transshipment_cost: float = 0.0
    service_cost: float = 0.0
    unused_vessel_cost: float = 0.0
    voyage_cost: float = 0.0
    port_call_cost: float = 0.0
    sailing_fuel_cost: float = 0.0
    idle_fuel_cost: float = 0.0
    canal_fee_cost: float = 0.0

    # --- demand coverage ---
    total_demand: float = 0.0
    routed_demand: float = 0.0
    demand_coverage: float = 0.0  # routed / total

    # --- diagnostic fields ---
    commodity_results: List[CommodityResult] = field(default_factory=list)
    vessel_requirements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_flows: Dict[int, Dict[str, float]] = field(default_factory=dict)
    expanded_graph_node_count: int = 0
    expanded_graph_edge_count: int = 0
    num_services: int = 0
    num_demands: int = 0
    instance_name: str = ""
    timing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # --- convenience accessors ---
    def get_commodity_result(self, idx: int) -> Optional[CommodityResult]:
        for cr in self.commodity_results:
            if cr.commodity_idx == idx:
                return cr
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict suitable for logging / JSON serialisation."""
        return {
            "instance": self.instance_name,
            "eta": self.eta,
            "total_revenue": self.total_revenue,
            "rejected_demand": self.rejected_demand,
            "rejection_cost": self.rejection_cost,
            "handling_cost": self.handling_cost,
            "service_cost": self.service_cost,
            "unused_vessel_cost": self.unused_vessel_cost,
            "voyage_cost": self.voyage_cost,
            "port_call_cost": self.port_call_cost,
            "sailing_fuel_cost": self.sailing_fuel_cost,
            "idle_fuel_cost": self.idle_fuel_cost,
            "canal_fee_cost": self.canal_fee_cost,
            "total_demand": self.total_demand,
            "routed_demand": self.routed_demand,
            "demand_coverage": self.demand_coverage,
            "num_services": self.num_services,
            "num_demands": self.num_demands,
            "timing_seconds": self.timing_seconds,
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        return (
            f"MCFResult(instance={self.instance_name!r}, "
            f"eta={self.eta:,.2f}, "
            f"coverage={self.demand_coverage:.1%}, "
            f"services={self.num_services})"
        )
