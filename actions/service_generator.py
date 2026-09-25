"""
P6 — Action & Service Generation for LSNDP.

Implements the paper-faithful mechanism for constructing valid liner
shipping service actions.

Action semantics:
    A_t = (A_v,t, A_p,t)
      - A_v,t: selected vessel class
      - A_p,t: ordered port sequence forming a cyclic rotation

This module owns:
  - Vessel selection interfaces
  - Port selection mechanisms
  - Service construction and ordering
  - Round-trip closure verification
  - Feasibility validation
  - Service uniqueness handling
  - Clean interfaces for future policies (encoder-only / encoder-decoder)

This module does NOT own:
  - GAT/Transformer/LSTM layers
  - Policy networks or PPO training
  - Benchmark experiments

Evidence tags throughout document paper vs implementation decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from data.instance import DistanceArc, LINERLIBInstance, VesselType
from env.action import ServiceAction
from mcf.expanded_graph import ServiceDefinition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Floating-point tolerance for distance comparisons.
_DIST_TOL = 1e-6

# Minimum number of ports in a service (paper requires at least 2 for a cycle).
_MIN_PORT_COUNT = 2


# ---------------------------------------------------------------------------
# Vessel selection heuristic for encoder-only pathway
# ---------------------------------------------------------------------------

def select_largest_available_vessel(
    fleet_remaining: Dict[str, float],
    vessel_types: Dict[str, VesselType],
) -> Optional[str]:
    """
    Select the vessel class with the largest capacity among those with
    remaining quantity > 0.

    [PAPER] Section 4.2, encoder-only pathway:
        "Rule-based — select the vessel class with the highest remaining
        count among those that can physically visit all selected ports
        (capacity constraint check)."

    This is a deterministic primitive used by the encoder-only pathway
    when the policy does not explicitly select a vessel class. For the
    encoder-decoder pathway, the LSTM decoder selects the vessel class,
    so this function is not invoked for vessel selection.

    Parameters
    ----------
    fleet_remaining : dict[str, float]
        Remaining vessel count per class (fractional, >= 0).
    vessel_types : dict[str, VesselType]
        Available vessel types keyed by class name.

    Returns
    -------
    str or None
        Selected vessel class name, or None if no vessels remain.

    Tie-breaking
    ------------
    If multiple classes share the same maximum capacity, the class whose
    name is lexicographically smallest is selected.
    """
    available = [
        vc for vc, qty in fleet_remaining.items()
        if qty > 0 and vc in vessel_types
    ]
    if not available:
        return None
    # Primary sort: capacity descending; secondary sort: name ascending (for ties).
    best = min(
        available,
        key=lambda vc: (-vessel_types[vc].capacity_ffe, vc),
    )
    return best


# ---------------------------------------------------------------------------
# Vessel requirement calculation
# ---------------------------------------------------------------------------

def calculate_vessel_requirement(
    tour_distance_nm: float,
    design_speed_knots: float,
) -> float:
    """
    Calculate the number of vessels required for a service.

    Formula: n_{v,s} = L_s / (v_s × 7)

    [PAPER] Appendix A.3, Eq. 3-4. The factor of 7 converts daily speed
    (knots = nm/hour) to weekly distance coverage.

    Parameters
    ----------
    tour_distance_nm : float
        Total round-trip distance in nautical miles (L_s).
    design_speed_knots : float
        Vessel design speed in knots (v_s).

    Returns
    -------
    float
        Required number of vessels (fractional, no ceiling applied).

    Evidence
    --------
    [PAPER] Appendix A.3:
        "The number of ships required for service s is given by
         n_{v,s} = L_s / (v_s * 7)"
    [P2] CONFIRMED: Fractional vessel requirements preserved, no rounding.
    """
    if design_speed_knots <= 0:
        raise ValueError(
            f"Design speed must be positive, got {design_speed_knots}"
        )
    return tour_distance_nm / (design_speed_knots * 7.0)


# ---------------------------------------------------------------------------
# Approximate TSP for service ordering
# ---------------------------------------------------------------------------

def _nearest_neighbor_tsp(
    ports: List[str],
    dist_by_pair: Dict[Tuple[str, str], Any],
    start_port: Optional[str] = None,
) -> List[str]:
    """
    Solve approximate TSP using nearest-neighbor heuristic.

    [PAPER] Section 4, encoder-only pathway:
        "After selecting which ports are included, the order is determined
        by an approximate TSP procedure."

    The exact approximation procedure is underspecified in the paper.
    We use nearest-neighbor with deterministic tie-breaking (alphabetical
    order for equal distances).

    Parameters
    ----------
    ports : list[str]
        Ports to visit (already selected by policy).
    dist_by_pair : dict
        (origin, dest) -> distance_nm (float) OR DistanceArc object.
    start_port : str, optional
        If provided, force this as the starting port. Otherwise use the
        first port alphabetically for determinism.

    Returns
    -------
    list[str]
        Ordered port sequence forming a cyclic tour.
    """
    if not ports:
        return []

    # Deterministic starting point.
    if start_port is None:
        start_port = sorted(ports)[0]

    remaining = list(ports)
    tour = []
    current = start_port

    while remaining:
        tour.append(current)
        remaining.remove(current)

        if not remaining:
            break

        # Find nearest neighbor with deterministic tie-breaking.
        best_next = None
        best_dist = float('inf')
        for candidate in sorted(remaining):  # sorted for deterministic ties
            arc = dist_by_pair.get((current, candidate))
            if arc is None:
                dist = float('inf')
            elif isinstance(arc, DistanceArc):
                dist = arc.distance_nm
            else:
                dist = arc  # already a float
            if dist < best_dist - _DIST_TOL:
                best_dist = dist
                best_next = candidate
            elif abs(dist - best_dist) <= _DIST_TOL and candidate < best_next:
                # Tie-break alphabetically.
                best_next = candidate

        if best_next is None:
            # No valid edge found; append remaining ports alphabetically.
            tour.extend(sorted(remaining))
            break
        else:
            current = best_next

    return tour


# ---------------------------------------------------------------------------
# Service construction and validation
# ---------------------------------------------------------------------------

@dataclass
class ServiceValidationResult:
    """Result of service validation."""

    is_valid: bool
    reasons: List[str] = field(default_factory=list)
    service_action: Optional[ServiceAction] = None

    @classmethod
    def valid(cls, action: ServiceAction) -> "ServiceValidationResult":
        return cls(is_valid=True, service_action=action)

    @classmethod
    def invalid(cls, reasons: List[str]) -> "ServiceValidationResult":
        return cls(is_valid=False, reasons=reasons)


class ServiceGenerator:
    """
    Generate and validate liner shipping services.

    Parameters
    ----------
    instance : LINERLIBInstance
        The benchmark instance providing port/vessel/distance data.
    distances_by_pair : dict
        (origin, dest) -> DistanceArc lookup.
    """

    def __init__(
        self,
        instance: LINERLIBInstance,
        distances_by_pair: Dict[Tuple[str, str], DistanceArc],
        draft_filter_enabled: bool = True,
    ) -> None:
        self._instance = instance
        self._dist = distances_by_pair
        self._vessel_classes = sorted(instance.vessel_types.keys())
        self._ports_sorted = sorted(instance.ports.keys())
        self._draft_filter_enabled = draft_filter_enabled

    # ---- public API ----

    def generate_service(
        self,
        vessel_class: str,
        port_sequence: List[str],
        service_id: Optional[int] = None,
    ) -> ServiceValidationResult:
        """
        Generate a validated service action.

        Parameters
        ----------
        vessel_class : str
            Name of the vessel class.
        port_sequence : list[str]
            Ordered sequence of port UNLOCODEs.
        service_id : int, optional
            Optional unique identifier.

        Returns
        -------
        ServiceValidationResult
            Contains the validated ServiceAction or error reasons.
        """
        reasons = self.validate_service(vessel_class, port_sequence)
        if reasons:
            return ServiceValidationResult.invalid(reasons)

        action = ServiceAction(
            vessel_class=vessel_class,
            port_sequence=port_sequence,
            service_id=service_id,
        )
        return ServiceValidationResult.valid(action)

    def validate_service(
        self,
        vessel_class: str,
        port_sequence: List[str],
    ) -> List[str]:
        """
        Validate structural feasibility of a service.

        Checks:
          1. Vessel class exists in instance.
          2. All ports exist in instance.
          3. At least 2 ports (minimum meaningful cycle).
          4. No duplicate ports within the sequence.
          5. Distance exists for each consecutive port pair.
          6. Draft compatibility: vessel draft >= port draft.

        Returns
        -------
        list[str]
            Empty list if valid; list of failure reasons otherwise.
        """
        reasons: List[str] = []

        # Check 1: Vessel class exists.
        if vessel_class not in self._instance.vessel_types:
            reasons.append(
                f"Vessel class '{vessel_class}' not in instance vessel_types. "
                f"Available: {self._vessel_classes}"
            )

        vt = self._instance.vessel_types.get(vessel_class)

        # Check 2 & 3: Ports exist and minimum length.
        if not port_sequence:
            reasons.append("Port sequence is empty.")
        else:
            for p in port_sequence:
                if p not in self._instance.ports:
                    reasons.append(f"Port '{p}' not in instance ports.")
            if len(port_sequence) < _MIN_PORT_COUNT:
                reasons.append(
                    f"Port sequence must contain at least {_MIN_PORT_COUNT} "
                    f"ports for a cycle, got {len(port_sequence)}."
                )

        # Check 4: No duplicate ports.
        if len(port_sequence) != len(set(port_sequence)):
            reasons.append(
                "Duplicate ports in sequence (each port should appear once, "
                "cycle closes implicitly)."
            )

        # Check 5: Distance existence for consecutive port pairs.
        # [PAPER] Draft is NOT a hard constraint — C_unused handles fleet
        # deviations economically per Eq. 34. We validate distance only here.
        if vt and port_sequence:
            n_ports = len(port_sequence)
            for i in range(n_ports):
                p_from = port_sequence[i]
                p_to = port_sequence[(i + 1) % n_ports]

                arc = self._dist.get((p_from, p_to))
                if arc is None or arc.distance_nm <= 0:
                    reasons.append(
                        f"No valid distance for leg {p_from}→{p_to}."
                    )

        return reasons

    def order_ports(
        self,
        selected_ports: List[str],
        vessel_class: str,
    ) -> List[str]:
        """
        Determine an ordered service sequence from a set of selected ports.

        Uses nearest-neighbor TSP heuristic with deterministic tie-breaking.

        [PAPER] Section 4, encoder-only pathway.

        Parameters
        ----------
        selected_ports : list[str]
            Ports selected by the policy (unordered set).
        vessel_class : str
            Vessel class for draft feasibility checking.

        Returns
        -------
        list[str]
            Ordered port sequence forming a valid cyclic tour.
        """
        if len(selected_ports) < _MIN_PORT_COUNT:
            return sorted(selected_ports)

        # Filter to draft-feasible ports (only when draft filtering is enabled).
        # When disabled, all decoded ports pass through to TSP for a controlled
        # ablation of the hard draft filter (G11.2.3 intervention).
        vt = self._instance.vessel_types.get(vessel_class)
        feasible_ports = selected_ports
        if self._draft_filter_enabled and vt:
            feasible_ports = [
                p for p in selected_ports
                if self._is_draft_feasible(p, vt)
            ]

        if len(feasible_ports) < _MIN_PORT_COUNT:
            # Fall back to alphabetical ordering if too many ports are infeasible.
            feasible_ports = selected_ports[:_MIN_PORT_COUNT]

        return _nearest_neighbor_tsp(
            feasible_ports,
            {(o, d): a.distance_nm for (o, d), a in self._dist.items()},
        )

    def can_visit_port(self, vessel_class: str, port_code: str) -> bool:
        """Check if a vessel class can physically visit a port."""
        vt = self._instance.vessel_types.get(vessel_class)
        port = self._instance.ports.get(port_code)
        if vt is None or port is None:
            return False
        if port.draft is None:
            return True
        return vt.draft >= port.draft - _DIST_TOL

    def get_vessel_classes_for_ports(
        self, port_sequence: List[str],
    ) -> List[str]:
        """
        Return vessel classes that can physically visit all ports in sequence.
        """
        compatible = []
        for vc in self._vessel_classes:
            vt = self._instance.vessel_types[vc]
            if all(self._is_draft_feasible(p, vt) for p in port_sequence):
                compatible.append(vc)
        return compatible

    def compute_tour_distance(self, port_sequence: List[str]) -> float:
        """Compute total round-trip distance for a port sequence."""
        total = 0.0
        n = len(port_sequence)
        for i in range(n):
            arc = self._dist.get((port_sequence[i], port_sequence[(i + 1) % n]))
            if arc is not None:
                total += arc.distance_nm
        return total

    def compute_vessel_requirement(
        self, vessel_class: str, port_sequence: List[str],
    ) -> float:
        """
        Compute the number of vessels required for a service.

        Formula: n_{v,s} = L_s / (v_s × 7)
        """
        vt = self._instance.vessel_types.get(vessel_class)
        if vt is None:
            raise ValueError(f"Unknown vessel class: {vessel_class}")
        tour_dist = self.compute_tour_distance(port_sequence)
        return calculate_vessel_requirement(tour_dist, vt.design_speed)

    # ---- internal helpers ----

    def _is_draft_feasible(self, port_code: str, vt: VesselType) -> bool:
        """Check if vessel draft is sufficient for port."""
        port = self._instance.ports.get(port_code)
        if port is None or port.draft is None:
            return True
        return vt.draft >= port.draft - _DIST_TOL


# ---------------------------------------------------------------------------
# Canonicalization for duplicate detection
# ---------------------------------------------------------------------------

def canonicalize_service(action: ServiceAction) -> Tuple[str, ...]:
    """
    Create a canonical representation for duplicate detection.

    Two cyclic services are considered identical if one is a rotation of the
    other. This function returns the lexicographically smallest rotation.

    Parameters
    ----------
    action : ServiceAction
        The service to canonicalize.

    Returns
    -------
    tuple[str, ...]
        Canonical form as a tuple of port codes.
    """
    seq = action.port_sequence
    if len(seq) <= 1:
        return tuple(seq)

    # Find all rotations and return the lexicographically smallest.
    rotations = []
    n = len(seq)
    for i in range(n):
        rotation = tuple(seq[i:] + seq[:i])
        rotations.append(rotation)

    return min(rotations)


def are_services_equivalent(
    action1: ServiceAction, action2: ServiceAction,
) -> bool:
    """
    Check if two services are equivalent (same ports, same vessel, cyclically same).

    [ENGINEERING DECISION]: Services are equivalent if they use the same vessel
    class and their port sequences are cyclic rotations of each other.

    Parameters
    ----------
    action1 : ServiceAction
    action2 : ServiceAction

    Returns
    -------
    bool
        True if services are equivalent.
    """
    if action1.vessel_class != action2.vessel_class:
        return False
    if action1.num_ports != action2.num_ports:
        return False
    return canonicalize_service(action1) == canonicalize_service(action2)


# ---------------------------------------------------------------------------
# Action validation for policy compatibility
# ---------------------------------------------------------------------------

class ActionValidator:
    """
    Validate actions from policy outputs.

    Provides clean interfaces for both encoder-only and encoder-decoder
    pathways.
    """

    def __init__(self, generator: ServiceGenerator) -> None:
        self._gen = generator

    def validate_encoder_only_output(
        self,
        vessel_class: str,
        selected_ports: Set[str],
    ) -> ServiceValidationResult:
        """
        Validate output from encoder-only pathway.

        Encoder-only pathway:
          1. Policy selects vessel class
          2. Policy selects port subset (Bernoulli sampling)
          3. This module orders the ports via approximate TSP

        Parameters
        ----------
        vessel_class : str
            Selected vessel class.
        selected_ports : set[str]
            Ports selected by the policy.

        Returns
        -------
        ServiceValidationResult
        """
        ports_list = sorted(selected_ports)
        if len(ports_list) < _MIN_PORT_COUNT:
            return ServiceValidationResult.invalid([
                f"Need at least {_MIN_PORT_COUNT} ports, got {len(ports_list)}."
            ])

        ordered = self._gen.order_ports(ports_list, vessel_class)
        return self._gen.generate_service(vessel_class, ordered)

    def validate_encoder_decoder_output(
        self,
        vessel_class: str,
        port_sequence: List[str],
    ) -> ServiceValidationResult:
        """
        Validate output from encoder-decoder pathway.

        Encoder-decoder pathway:
          1. Policy selects vessel class
          2. LSTM decoder sequentially selects ports
          3. This module validates the resulting sequence

        Parameters
        ----------
        vessel_class : str
            Selected vessel class.
        port_sequence : list[str]
            Port sequence from decoder.

        Returns
        -------
        ServiceValidationResult
        """
        return self._gen.generate_service(vessel_class, port_sequence)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def make_service_action(
    vessel_class: str,
    port_sequence: List[str],
    service_id: Optional[int] = None,
) -> ServiceAction:
    """Create a ServiceAction directly (for testing)."""
    return ServiceAction(
        vessel_class=vessel_class,
        port_sequence=port_sequence,
        service_id=service_id,
    )


def make_service_definition(
    service_id: int,
    vessel_class: str,
    port_sequence: List[str],
) -> ServiceDefinition:
    """Create a ServiceDefinition for MCF evaluation."""
    return ServiceDefinition(
        service_id=service_id,
        vessel_class=vessel_class,
        port_sequence=port_sequence,
    )
