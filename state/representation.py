"""
P5 — Neural State Representation for LSNDP.

Transforms P4 environment state into a structured graph representation
ready for GAT/Transformer consumption.

State at step t: S_t = (S*_t, V_t, p_t, f^s_e, f^d_e)

  - p_t ∈ R^((P+1)×2)       port/node features (incoming + outgoing demand)
  - f^s_e ∈ R^(4×E)         static edge features
  - f^d_e ∈ R^((2+|S|_max)×E)  dynamic edge features
  - v_t ∈ R^(V×11)          vessel features

This module owns DATA REPRESENTATION ONLY. It does NOT implement:
  - GAT layers, Transformer layers, LSTM
  - Policy networks, PPO, training loops
  - Service generation (P6's domain)

Evidence tags throughout document paper vs implementation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.instance import LINERLIBInstance
from mcf.expanded_graph import ServiceDefinition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of services (matches P4 safety cap).
# Used to fix the shape of service-membership dynamic features.
_MAX_SERVICES = 100

# Floating-point tolerance for equality checks.
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Service membership tracking
# ---------------------------------------------------------------------------

@dataclass
class ServiceMembership:
    """
    Tracks which services are currently active and their composition.

    Parameters
    ----------
    service_defs : list[ServiceDefinition]
        Ordered list of services added so far (from P4 env state).
    n_vs : dict[str, dict[str, float]]
        Vessel requirements keyed by service_id -> {vessel_class: n_vs}.
    """

    service_defs: List[ServiceDefinition] = field(default_factory=list)
    n_vs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(self, svc: ServiceDefinition, n_vs_val: Dict[str, float]) -> None:
        """Append a new service and its vessel requirement."""
        sid = str(svc.service_id)
        self.service_defs.append(svc)
        self.n_vs[sid] = n_vs_val

    @property
    def count(self) -> int:
        return len(self.service_defs)

    def get_edge_service_mask(
        self, origin: str, destination: str,
    ) -> np.ndarray:
        """
        Return a binary vector of length |services| indicating whether each
        service includes the directed edge (origin -> destination).
        """
        mask = np.zeros(len(self.service_defs), dtype=np.float32)
        for j, svc in enumerate(self.service_defs):
            seq = svc.port_sequence
            n = len(seq)
            for i in range(n):
                if seq[i] == origin and seq[(i + 1) % n] == destination:
                    mask[j] = 1.0
                    break
        return mask

    def get_port_service_mask(self, port_code: str) -> np.ndarray:
        """Return a binary vector: 1 if service visits this port."""
        mask = np.zeros(len(self.service_defs), dtype=np.float32)
        for j, svc in enumerate(self.service_defs):
            if port_code in svc.visited_ports:
                mask[j] = 1.0
        return mask


# ---------------------------------------------------------------------------
# Deterministic index mappings
# ---------------------------------------------------------------------------

def build_index_mappings(
    instance: LINERLIBInstance,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], Dict[str, int]]:
    """
    Build authoritative deterministic index maps.

    Returns
    -------
    port_to_node_idx : dict[str, int]
        Maps UNLOCODE -> node index (0..P-1). Global node is P.
    od_to_edge_idx : dict[tuple, int]
        Maps (origin, dest) -> edge index (0..E-1).
    vessel_to_vessel_idx : dict[str, int]
        Maps vessel class name -> vessel index (0..V-1).
    """
    # Ports sorted alphabetically for determinism.
    port_codes = sorted(instance.ports.keys())
    port_to_node_idx = {code: i for i, code in enumerate(port_codes)}

    # Edges sorted by (origin, dest) tuple for determinism.
    od_pairs = sorted(
        {(arc.origin, arc.destination) for arc in instance.distances},
    )
    od_to_edge_idx = {pair: i for i, pair in enumerate(od_pairs)}

    # Vessel classes sorted alphabetically.
    vessel_classes = sorted(instance.vessel_types.keys())
    vessel_to_vessel_idx = {vc: i for i, vc in enumerate(vessel_classes)}

    return port_to_node_idx, od_to_edge_idx, vessel_to_vessel_idx


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _fit_min_max(values: List[float]) -> Tuple[float, float]:
    """Fit min-max parameters; handle constant/empty inputs gracefully."""
    if not values:
        return 0.0, 1.0
    lo, hi = float(min(values)), float(max(values))
    if abs(hi - lo) < _TOL:
        return lo, lo + _TOL  # avoid division by zero
    return lo, hi


def _normalize(x: float, lo: float, hi: float) -> float:
    if abs(hi - lo) < _TOL:
        return 0.0
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Feature fit statistics (train-only)
# ---------------------------------------------------------------------------

@dataclass
class _FitStats:
    """Train-only fitted statistics for normalisation."""

    # Port features
    incoming_demand_lo: float = 0.0
    incoming_demand_hi: float = 0.0
    outgoing_demand_lo: float = 0.0
    outgoing_demand_hi: float = 0.0

    # Static edge features
    distance_lo: float = 0.0
    distance_hi: float = 0.0
    revenue_ratio_lo: float = 0.0
    revenue_ratio_hi: float = 0.0

    # Per-vessel-class statistics
    vessel_stats: List[Dict[str, Tuple[float, float]]] = field(default_factory=list)

    @classmethod
    def from_instance(cls, instance: LINERLIBInstance) -> "_FitStats":
        """Compute stats from a single instance (for testing / smoke)."""
        stats = cls()

        # Port-level aggregates from INITIAL demands
        port_incoming: Dict[str, float] = {}
        port_outgoing: Dict[str, float] = {}
        for d in instance.demands:
            port_incoming[d.destination] = port_incoming.get(d.destination, 0.0) + d.ffe_per_week
            port_outgoing[d.origin] = port_outgoing.get(d.origin, 0.0) + d.ffe_per_week

        all_in = [port_incoming.get(p, 0.0) for p in sorted(instance.ports.keys())]
        all_out = [port_outgoing.get(p, 0.0) for p in sorted(instance.ports.keys())]
        stats.incoming_demand_lo, stats.incoming_demand_hi = _fit_min_max(all_in)
        stats.outgoing_demand_lo, stats.outgoing_demand_hi = _fit_min_max(all_out)

        # Distance normalization stats
        dists = [a.distance_nm for a in instance.distances if a.distance_nm > 0]
        stats.distance_lo, stats.distance_hi = _fit_min_max(dists)

        # Revenue-ratio normalization stats.
        rev_ratios = []
        for arc in instance.distances:
            if arc.distance_nm <= 0:
                continue
            total_rev = sum(
                d.revenue * d.ffe_per_week
                for d in instance.demands
                if d.origin == arc.origin and d.destination == arc.destination
            )
            total_ffe = sum(
                d.ffe_per_week
                for d in instance.demands
                if d.origin == arc.origin and d.destination == arc.destination
            )
            ratio = total_rev / total_ffe if total_ffe > 0 else 0.0
            rev_ratios.append(ratio)
        if rev_ratios:
            stats.revenue_ratio_lo, stats.revenue_ratio_hi = _fit_min_max(rev_ratios)
        else:
            stats.revenue_ratio_lo, stats.revenue_ratio_hi = 0.0, 1.0

        # Demand OD-pair normalization stats (for dynamic demand feature).
        od_demands: Dict[Tuple[str, str], float] = {}
        for d in instance.demands:
            key = (d.origin, d.destination)
            od_demands[key] = od_demands.get(key, 0.0) + d.ffe_per_week
        all_od_vals = list(od_demands.values())
        stats.od_demand_lo, stats.od_demand_hi = _fit_min_max(all_od_vals)

        # Edge capacity max (for dynamic capacity feature normalization).
        # Use sum of all fleet capacities as upper bound.
        total_capacity = sum(e.quantity * instance.vessel_types[e.vessel_class].capacity_ffe
                            for e in instance.fleet)
        stats.capacity_max = total_capacity if total_capacity > 0 else 1.0

        # Per-vessel-class statistics
        vessel_classes = sorted(instance.vessel_types.keys())
        vessel_caps = [float(instance.vessel_types[vc].capacity_ffe) for vc in vessel_classes]
        vessel_tcs = [float(instance.vessel_types[vc].tc_rate_daily) for vc in vessel_classes]
        vessel_drafts = [float(instance.vessel_types[vc].draft) for vc in vessel_classes]
        vessel_speeds = [float(instance.vessel_types[vc].design_speed) for vc in vessel_classes]
        vessel_bunker = [float(instance.vessel_types[vc].bunker_ton_per_day_at_design) for vc in vessel_classes]
        vessel_idle = [float(instance.vessel_types[vc].idle_consumption_ton_per_day) for vc in vessel_classes]
        # Panamа/Suez fees: fixed monetary params; normalize across vessels.
        vessel_panama = [float(instance.vessel_types[vc].panama_fee or 0) for vc in vessel_classes]
        vessel_suez = [float(instance.vessel_types[vc].suez_fee or 0) for vc in vessel_classes]

        for i, _vc in enumerate(vessel_classes):
            stats.vessel_stats.append({
                "capacity": (vessel_caps[i], vessel_caps[i]),
                "tc_rate": (vessel_tcs[i], vessel_tcs[i]),
                "draft": (vessel_drafts[i], vessel_drafts[i]),
                "design_speed": (vessel_speeds[i], vessel_speeds[i]),
                "bunker": (vessel_bunker[i], vessel_bunker[i]),
                "idle": (vessel_idle[i], vessel_idle[i]),
                "panama": _fit_min_max(vessel_panama),
                "suez": _fit_min_max(vessel_suez),
            })

        return stats

    def normalize_demand(self, x: float, is_incoming: bool) -> float:
        lo = self.incoming_demand_lo if is_incoming else self.outgoing_demand_lo
        hi = self.incoming_demand_hi if is_incoming else self.outgoing_demand_hi
        return _normalize(x, lo, hi)

    def normalize_distance(self, x: float) -> float:
        return _normalize(x, self.distance_lo, self.distance_hi)

    def normalize_revenue_ratio(self, x: float) -> float:
        return _normalize(x, self.revenue_ratio_lo, self.revenue_ratio_hi)

    def normalize_od_demand(self, x: float) -> float:
        return _normalize(x, self.od_demand_lo, self.od_demand_hi)

    def normalize_capacity(self, x: float) -> float:
        return _normalize(x, 0.0, self.capacity_max)

    def normalize_vessel_feature(
        self, feature_name: str, value: float, vessel_idx: int,
    ) -> float:
        if vessel_idx >= len(self.vessel_stats):
            return 0.0
        stats = self.vessel_stats[vessel_idx].get(feature_name, (0.0, 1.0))
        return _normalize(value, stats[0], stats[1])


# ---------------------------------------------------------------------------
# NeuralState
# ---------------------------------------------------------------------------

@dataclass
class NeuralState:
    """
    Paper-faithful neural state representation for the LSNDP environment.

    Attributes
    ----------
    port_features : ndarray, shape (P+1, 2)
        Row i = [normalized_incoming_demand, normalized_outgoing_demand].
        Last row (global node) is [0, 0].
    static_edge_features : ndarray, shape (4, E)
        Row 0: origin port index (int, cast to float).
        Row 1: destination port index (int, cast to float).
        Row 2: normalized distance.
        Row 3: normalized revenue/total-demand ratio.
    dynamic_edge_features : ndarray, shape (2 + num_services, E)
        Row 0: remaining unsatisfied demand for this OD (normalized).
        Row 1: remaining edge capacity (FFE/week, absolute).
        Rows 2..: service membership indicator vectors (binary).
    vessel_features : ndarray, shape (V, 11)
        Columns correspond to paper Appendix A.1 features 1-11.
        See VESSEL_FEATURE_ORDER for the exact mapping.
    indices : dict
        Deterministic mappings: 'port_to_node', 'od_to_edge', 'vessel_to_vessel'.
    fit_stats : _FitStats
        Train-only fitted normalisation parameters.
    num_services : int
        Number of services currently active.
    instance_name : str
        Instance identifier.
    """

    port_features: np.ndarray
    static_edge_features: np.ndarray
    dynamic_edge_features: np.ndarray
    vessel_features: np.ndarray
    indices: Dict[str, Dict[str, int]]
    fit_stats: _FitStats
    num_services: int
    instance_name: str

    # Canonical column order for vessel_features (paper Appendix A.1, Table 3).
    VESSEL_FEATURE_ORDER = [
        "capacity",      # 1: v_cap   FFE
        "quantity",      # 2: v_n     vessels (remaining fleet)
        "tc_rate",       # 3: v_TC    USD/day
        "draft",         # 4: v_draft m
        "min_speed",     # 5: v_minSpeed knots
        "max_speed",     # 6: v_maxSpeed knots
        "design_speed",  # 7: v_s     knots
        "bunker",        # 8: v_fish  USD/day
        "idle",          # 9: v_fi    USD/day
        "panama",        # 10: v_panama USD
        "suez",          # 11: v_suez  USD
    ]


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """
    Encode a P4 environment step into a NeuralState.

    Parameters
    ----------
    instance : LINERLIBInstance
        The loaded benchmark instance (read-only reference).
    distances_by_pair : dict
        (origin, dest) -> DistanceArc lookup built from instance.distances.
    fit_stats : Optional[_FitStats]
        Pre-computed train-only normalisation stats. If None, computed from
        instance directly (suitable for single-instance testing).
    """

    def __init__(
        self,
        instance: LINERLIBInstance,
        distances_by_pair: Dict[Tuple[str, str], Any],
        fit_stats: Optional[_FitStats] = None,
    ) -> None:
        self._instance = instance
        self._dist = distances_by_pair
        self._fit_stats = fit_stats or _FitStats.from_instance(instance)
        self._port_to_node, self._od_to_edge, self._vessel_to_vessel = \
            build_index_mappings(instance)
        self._n_ports = len(instance.ports)
        # Use all distinct OD pairs from instance.distances for edge count.
        self._n_edges = len({(a.origin, a.destination) for a in instance.distances})
        self._vessel_classes = sorted(instance.vessel_types.keys())

    # ---- public API ----

    def encode(
        self,
        remaining_demand: Dict[int, float],
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
    ) -> NeuralState:
        """
        Produce a NeuralState from the current P4 environment state.

        Parameters
        ----------
        remaining_demand : dict[int, float]
            remaining_demand[idx] = unsatisfied FFE/week for commodity idx.
        fleet_remaining : dict[str, float]
            Remaining vessel count per class (fractional, >= 0).
        membership : ServiceMembership
            Current services and their vessel requirements.
        """
        return NeuralState(
            port_features=self._build_port_features(remaining_demand),
            static_edge_features=self._build_static_edges(),
            dynamic_edge_features=self._build_dynamic_edges(
                remaining_demand, fleet_remaining, membership,
            ),
            vessel_features=self._build_vessel_features(fleet_remaining, membership),
            indices={
                "port_to_node": self._port_to_node,
                "od_to_edge": self._od_to_edge,
                "vessel_to_vessel": self._vessel_to_vessel,
            },
            fit_stats=self._fit_stats,
            num_services=membership.count,
            instance_name=self._instance.name,
        )

    # ---- feature builders ----

    def _build_port_features(
        self, remaining_demand: Dict[int, float],
    ) -> np.ndarray:
        """
        Port features: [Sigma incoming demand, Sigma outgoing demand], normalized.
        Global node (index P) has [0, 0].

        Demand is recomputed at each step from remaining_demand.
        """
        incoming: Dict[str, float] = {}
        outgoing: Dict[str, float] = {}
        for idx, rem in remaining_demand.items():
            dem = self._instance.demands[idx]
            incoming[dem.destination] = incoming.get(dem.destination, 0.0) + rem
            outgoing[dem.origin] = outgoing.get(dem.origin, 0.0) + rem

        n = self._n_ports + 1  # +1 for global node
        feats = np.zeros((n, 2), dtype=np.float32)
        for i, code in enumerate(sorted(self._instance.ports.keys())):
            feats[i, 0] = self._fit_stats.normalize_demand(
                incoming.get(code, 0.0), is_incoming=True,
            )
            feats[i, 1] = self._fit_stats.normalize_demand(
                outgoing.get(code, 0.0), is_incoming=False,
            )
        # Row P (global node) stays [0, 0]
        return feats

    def _build_static_edges(self) -> np.ndarray:
        """
        Static edge features, shape (4, E):
          0: origin port index (float)
          1: destination port index (float)
          2: normalized distance
          3: normalized revenue/unit-demand ratio
        """
        E = self._n_edges
        feats = np.zeros((4, E), dtype=np.float32)

        # Iterate over edges in deterministic (sorted) order.
        for j, (o, d) in enumerate(self._od_to_edge.keys()):
            feats[0, j] = float(self._port_to_node[o])
            feats[1, j] = float(self._port_to_node[d])

            arc = self._dist.get((o, d))
            dist = arc.distance_nm if arc and arc.distance_nm > 0 else 0.0
            feats[2, j] = self._fit_stats.normalize_distance(dist)

            # Revenue per unit demand on this OD pair.
            total_rev = sum(
                dem.revenue * dem.ffe_per_week
                for dem in self._instance.demands
                if dem.origin == o and dem.destination == d
            )
            total_ffe = sum(
                dem.ffe_per_week
                for dem in self._instance.demands
                if dem.origin == o and dem.destination == d
            )
            ratio = total_rev / total_ffe if total_ffe > 0 else 0.0
            feats[3, j] = self._fit_stats.normalize_revenue_ratio(ratio)

        return feats

    def _build_dynamic_edges(
        self,
        remaining_demand: Dict[int, float],
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
    ) -> np.ndarray:
        """
        Dynamic edge features, shape (2 + num_services, E):
          0: remaining unsatisfied demand for this OD (normalized).
          1: remaining edge capacity in FFE/week (absolute).
          2..: binary service membership indicators.
        """
        num_svc = membership.count
        rows = 2 + num_svc
        feats = np.zeros((rows, self._n_edges), dtype=np.float32)

        # Remaining demand per OD pair.
        od_demand: Dict[Tuple[str, str], float] = {}
        for idx, rem in remaining_demand.items():
            dem = self._instance.demands[idx]
            key = (dem.origin, dem.destination)
            od_demand[key] = od_demand.get(key, 0.0) + rem

        # Fit demand normalization on-the-fly for dynamic features.
        # This is local to this step only — no train/test leakage because
        # we normalize within the current tensor, not across episodes.
        all_d_vals = list(od_demand.values())
        d_lo, d_hi = _fit_min_max(all_d_vals) if all_d_vals else (0.0, 1.0)

        for j, (o, d) in enumerate(self._od_to_edge.keys()):
            # Remaining demand (normalized using pre-fit stats).
            val = od_demand.get((o, d), 0.0)
            feats[0, j] = self._fit_stats.normalize_od_demand(val)

            # Edge capacity: sum over services using this edge of n_vs * v_cap.
            cap = 0.0
            for svc in membership.service_defs:
                seq = svc.port_sequence
                n_p = len(seq)
                in_service = any(
                    seq[i] == o and seq[(i + 1) % n_p] == d
                    for i in range(n_p)
                )
                if in_service:
                    sid = str(svc.service_id)
                    n_vs = membership.n_vs.get(sid, {}).get(svc.vessel_class, 0.0)
                    vt = self._instance.vessel_types.get(svc.vessel_class)
                    if vt:
                        cap += n_vs * vt.capacity_ffe
            feats[1, j] = self._fit_stats.normalize_capacity(cap)

            # Service membership indicators.
            mask = membership.get_edge_service_mask(o, d)
            feats[2:, j] = mask

        return feats

    def _build_vessel_features(
        self,
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
    ) -> np.ndarray:
        """
        Vessel features, shape (V, 11).
        Column order matches paper Appendix A.1 features 1-11.
        """
        V = len(self._vessel_classes)
        feats = np.zeros((V, 11), dtype=np.float32)

        for vi, vc in enumerate(self._vessel_classes):
            vt = self._instance.vessel_types[vc]

            # 1: capacity (normalized).
            feats[vi, 0] = self._fit_stats.normalize_vessel_feature(
                "capacity", float(vt.capacity_ffe), vi,
            )
            # 2: quantity (remaining fleet, raw — dynamic).
            feats[vi, 1] = float(fleet_remaining.get(vc, 0.0))
            # 3: tc_rate (normalized).
            feats[vi, 2] = self._fit_stats.normalize_vessel_feature(
                "tc_rate", float(vt.tc_rate_daily), vi,
            )
            # 4: draft (normalized).
            feats[vi, 3] = self._fit_stats.normalize_vessel_feature(
                "draft", float(vt.draft), vi,
            )
            # 5: min_speed (normalized).
            feats[vi, 4] = self._fit_stats.normalize_vessel_feature(
                "min_speed", float(vt.min_speed), vi,
            )
            # 6: max_speed (normalized).
            feats[vi, 5] = self._fit_stats.normalize_vessel_feature(
                "max_speed", float(vt.max_speed), vi,
            )
            # 7: design_speed (normalized).
            feats[vi, 6] = self._fit_stats.normalize_vessel_feature(
                "design_speed", float(vt.design_speed), vi,
            )
            # 8: bunker_ton_per_day_at_design (normalized).
            feats[vi, 7] = self._fit_stats.normalize_vessel_feature(
                "bunker", float(vt.bunker_ton_per_day_at_design), vi,
            )
            # 9: idle_consumption_ton_per_day (normalized).
            feats[vi, 8] = self._fit_stats.normalize_vessel_feature(
                "idle", float(vt.idle_consumption_ton_per_day), vi,
            )
            # 10: panama_fee (normalized across vessel classes).
            feats[vi, 9] = self._fit_stats.normalize_vessel_feature(
                "panama", float(vt.panama_fee or 0), vi,
            )
            # 11: suez_fee (normalized across vessel classes).
            feats[vi, 10] = self._fit_stats.normalize_vessel_feature(
                "suez", float(vt.suez_fee or 0), vi,
            )

        return feats
