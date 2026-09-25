"""
P3 – Cost calculation components for the MCF evaluation engine.

All cost terms follow the P2 mathematical contract (docs/PROBLEM_FORMULATION.md).
Evidence tags identify source: [PAPER], [REFERENCE], [INFERENCE], [IMPLEMENTATION].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from data.instance import DistanceArc, FleetEntry, LINERLIBInstance, VesselType


# ---------------------------------------------------------------------------
# Dataclasses for result components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RevenueComponent:
    """Revenue from routed demand."""
    total_revenue: float  # USD, weekly basis


@dataclass(frozen=True)
class RejectionComponent:
    """Rejected demand and its penalty cost."""
    total_rejected: float  # FFE/week
    penalty_cost: float  # USD; C_reject = Y_d * total_rejected


@dataclass(frozen=True)
class HandlingComponent:
    """Loading/unloading + transshipment handling cost."""
    loading_unloading_cost: float  # USD
    transshipment_cost: float  # USD
    total_handling_cost: float  # USD


@dataclass(frozen=True)
class ServiceComponent:
    """Vessel charter cost (C_service)."""
    # [PAPER] Eq. 33 structure with [INFERENCE] weekly factor 7.
    total_service_cost: float  # USD


@dataclass(frozen=True)
class UnusedVesselComponent:
    """Unused/over-utilized vessel economic effect (C_unused)."""
    # [PAPER] Eq. 34 structure with [INFERENCE] weekly factor 7.
    total_unused_cost: float  # USD (can be negative = profit)


@dataclass(frozen=True)
class VoyageComponent:
    """Port call + fuel + canal costs (C_voyage)."""
    port_call_cost: float  # USD
    sailing_fuel_cost: float  # USD
    idle_fuel_cost: float  # USD
    canal_fee_cost: float  # USD
    total_voyage_cost: float  # USD


# ---------------------------------------------------------------------------
# Core cost calculator
# ---------------------------------------------------------------------------

class CostCalculator:
    """
    Compute every cost/revenue component for a candidate network.

    Parameters
    ----------
    instance :
        A loaded LINERLIBInstance (P1 data foundation).
    distances_by_pair :
        Dict mapping (origin, dest) -> DistanceArc for quick lookup.
    """

    def __init__(
        self,
        instance: LINERLIBInstance,
        distances_by_pair: Dict[tuple, DistanceArc],
    ) -> None:
        self._instance = instance
        self._dist = distances_by_pair
        self._Y_d = 1000.0  # [PAPER — CONFIRMED] Appendix A.1

    # ---- public API ----

    def compute_revenue(self, routed_per_commodity: Dict[int, float]) -> RevenueComponent:
        """
        R_total = Σ_d d_R * satisfied_d.

        Parameters
        ----------
        routed_per_commodity :
            Mapping from demand index -> satisfied FFE/week.
        """
        total = 0.0
        for idx, satisfied in routed_per_commodity.items():
            dem = self._instance.demands[idx]
            total += dem.revenue * satisfied
        return RevenueComponent(total_revenue=total)

    def compute_rejection(
        self,
        routed_per_commodity: Dict[int, float],
        total_demand: float,
    ) -> RejectionComponent:
        """
        C_reject = Y_d * Σ_d (d_q - satisfied_d).

        Parameters
        ----------
        routed_per_commodity :
            Mapping from demand index -> satisfied FFE/week.
        total_demand :
            Sum of all d_q across demands.
        """
        total_routed = sum(routed_per_commodity.values())
        rejected = max(0.0, total_demand - total_routed)
        return RejectionComponent(
            total_rejected=rejected,
            penalty_cost=self._Y_d * rejected,
        )

    def compute_handling(
        self,
        flows: Dict[int, Dict[str, float]],
        expanded_graph,
    ) -> HandlingComponent:
        """
        C_handle = loading/unloading + transshipment cost.

        Flows are stored as string keys "u_key-->v_key". We parse these
        back to determine edge types and apply the correct per-FFE cost.
        """
        port_costs = {
            p: port for p, port in self._instance.ports.items()
        }

        load_unload = 0.0
        transship = 0.0

        for _cmd_idx, edge_flows in flows.items():
            for flow_key, flow_val in edge_flows.items():
                if flow_val <= 0:
                    continue
                parts = flow_key.split("-->")
                if len(parts) != 2:
                    continue
                u_str, v_str = parts[0].strip(), parts[1].strip()

                # Determine edge type from node structure
                # Loading: physical_port -> physical_port[sN]
                # Offloading: physical_port[sN] -> physical_port
                # Transshipment: physical_port[sN] -> physical_port[sM]
                # Service: physical_port[sN] -> physical_port[sN] (same service)

                u_is_proxy = "[s" in u_str
                v_is_proxy = "[s" in v_str

                if u_is_proxy and not v_is_proxy:
                    # Offloading: proxy -> physical port
                    # Extract port code from proxy: "PORT[sN]" -> "PORT"
                    pcode = u_str.split("[")[0]
                    pc = port_costs.get(pcode)
                    if pc is not None and pc.cost_per_full is not None:
                        load_unload += flow_val * pc.cost_per_full
                elif not u_is_proxy and v_is_proxy:
                    # Loading: physical port -> proxy
                    pcode = u_str
                    pc = port_costs.get(pcode)
                    if pc is not None and pc.cost_per_full is not None:
                        load_unload += flow_val * pc.cost_per_full
                elif u_is_proxy and v_is_proxy:
                    # Could be transshipment or service transit
                    u_port = u_str.split("[")[0]
                    v_port = v_str.split("[")[0]
                    if u_port == v_port:
                        # Same port, different services -> transshipment
                        pc = port_costs.get(u_port)
                        if pc is not None and pc.cost_per_full_transfer is not None:
                            transship += flow_val * pc.cost_per_full_transfer
                    # else: service transit, weight=0, no handling cost

        return HandlingComponent(
            loading_unloading_cost=load_unload,
            transshipment_cost=transship,
            total_handling_cost=load_unload + transship,
        )

    def compute_service_cost(self, vessel_requirements: Dict[str, Dict[str, float]]) -> ServiceComponent:
        """
        C_service = 7 * Σ_s Σ_v n_{v,s} * v_TC.

        [PAPER] Eq. 33 structure; factor 7 is [INFERENCE / NUMERICAL VERIFICATION].
        """
        vessel_types = self._instance.vessel_types
        total = 0.0
        for _svc_id, reqs in vessel_requirements.items():
            for vclass, n_vs in reqs.items():
                vt = vessel_types.get(vclass)
                if vt is not None:
                    total += n_vs * vt.tc_rate_daily
        return ServiceComponent(total_service_cost=7.0 * total)

    def compute_unused_vessel_cost(
        self,
        vessel_requirements: Dict[str, Dict[str, float]],
    ) -> UnusedVesselComponent:
        """
        C_unused = -7 * Σ_v (v_n - Σ_s n_{v,s}) * v_TC.

        [PAPER] Eq. 34 structure; factor 7 is [INFERENCE / NUMERICAL VERIFICATION].
        Negative sign is deliberate and confirmed by Table 1.
        """
        fleet_map = {e.vessel_class: e.quantity for e in self._instance.fleet}
        vessel_types = self._instance.vessel_types

        usage: Dict[str, float] = {}
        for reqs in vessel_requirements.values():
            for vclass, n_vs in reqs.items():
                usage[vclass] = usage.get(vclass, 0.0) + n_vs

        total = 0.0
        for vclass, used in usage.items():
            available = fleet_map.get(vclass, 0)
            vt = vessel_types.get(vclass)
            if vt is not None:
                total += (available - used) * vt.tc_rate_daily

        return UnusedVesselComponent(total_unused_cost=-7.0 * total)

    def compute_voyage_cost(
        self,
        services,
        vessel_requirements: Dict[str, Dict[str, float]],
    ) -> VoyageComponent:
        """
        C_voyage = C_port + C_fuel + C_idle + C_canal.

        [PAPER] Eq. 35. Canal fees have NO n_{v,s} multiplier [PAPER — CONFIRMED].
        """
        vessel_types = self._instance.vessel_types
        ports = self._instance.ports

        port_call_total = 0.0
        sailing_fuel_total = 0.0
        idle_fuel_total = 0.0
        canal_total = 0.0

        for svc in services:
            vclass = svc.vessel_class
            vt = vessel_types.get(vclass)
            if vt is None:
                continue
            n_vs = vessel_requirements.get(svc.service_id, {}).get(vclass, 0.0)

            # Tour distance
            tour_dist = 0.0
            num_ports = len(svc.port_sequence)
            for i in range(num_ports):
                p_from = svc.port_sequence[i]
                p_to = svc.port_sequence[(i + 1) % num_ports]
                arc = self._dist.get((p_from, p_to))
                if arc is not None:
                    tour_dist += arc.distance_nm
                    if arc.is_suez and vt.suez_fee:
                        canal_total += vt.suez_fee  # per-service, no n_vs multiplier
                    if arc.is_panama and vt.panama_fee:
                        canal_total += vt.panama_fee  # per-service, no n_vs multiplier

            # Port call cost: |s_P| * (p_f + p_v * v_cap) * n_{v,s}
            for p_name in svc.port_sequence:
                p = ports.get(p_name)
                if p is None:
                    continue
                pf = p.port_call_cost_fixed or 0.0
                pv = p.port_call_cost_per_ffe or 0.0
                port_call_total += (pf + pv * vt.capacity_ffe) * n_vs

            # Fuel: (tour_dist / (v_s * 24)) * v_fs * n_{v,s}
            sailing_days = tour_dist / (vt.design_speed * 24.0)
            sailing_fuel_total += sailing_days * vt.bunker_ton_per_day_at_design * n_vs

            # Idle: |s_P| * 1 day * v_fi * n_{v,s}
            idle_fuel_total += num_ports * 1.0 * vt.idle_consumption_ton_per_day * n_vs

        return VoyageComponent(
            port_call_cost=port_call_total,
            sailing_fuel_cost=sailing_fuel_total,
            idle_fuel_cost=idle_fuel_total,
            canal_fee_cost=canal_total,
            total_voyage_cost=port_call_total + sailing_fuel_total + idle_fuel_total + canal_total,
        )

    def compute_all(
        self,
        services,
        vessel_requirements: Dict[str, Dict[str, float]],
        flows: Dict[int, Dict[str, float]],
        routed_per_commodity: Dict[int, float],
        expanded_graph,
    ) -> dict:
        """Compute all cost components and return as a flat dict."""
        total_demand = sum(d.ffe_per_week for d in self._instance.demands)

        revenue = self.compute_revenue(routed_per_commodity)
        rejection = self.compute_rejection(routed_per_commodity, total_demand)
        handling = self.compute_handling(flows, expanded_graph)
        service_comp = self.compute_service_cost(vessel_requirements)
        unused_comp = self.compute_unused_vessel_cost(vessel_requirements)
        voyage = self.compute_voyage_cost(services, vessel_requirements)

        eta = (
            revenue.total_revenue
            - rejection.penalty_cost
            - handling.total_handling_cost
            - service_comp.total_service_cost
            - unused_comp.total_unused_cost
            - voyage.total_voyage_cost
        )

        return {
            "total_revenue": revenue.total_revenue,
            "rejected_demand": rejection.total_rejected,
            "rejection_cost": rejection.penalty_cost,
            "handling_cost": handling.total_handling_cost,
            "loading_unloading_cost": handling.loading_unloading_cost,
            "transshipment_cost": handling.transshipment_cost,
            "service_cost": service_comp.total_service_cost,
            "unused_vessel_cost": unused_comp.total_unused_cost,
            "voyage_cost": voyage.total_voyage_cost,
            "port_call_cost": voyage.port_call_cost,
            "sailing_fuel_cost": voyage.sailing_fuel_cost,
            "idle_fuel_cost": voyage.idle_fuel_cost,
            "canal_fee_cost": voyage.canal_fee_cost,
            "eta": eta,
            "total_demand": total_demand,
            "routed_demand": sum(routed_per_commodity.values()),
        }
