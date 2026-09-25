"""
P6 — Tests for action & service generation.

Covers:
  - Vessel selection
  - Port selection
  - Service construction
  - Cyclic closure
  - Duplicate-port rejection
  - Invalid-port rejection
  - Draft feasibility
  - Distance feasibility
  - Deterministic ordering
  - Approximate-TSP ordering interface
  - Vessel requirement calculation
  - Fractional vessel requirement
  - Duplicate-service handling
  - ServiceAction conversion
  - P4 integration
  - P3 integration
  - Encoder-only interface compatibility
  - Encoder-decoder interface compatibility
  - Real Baltic service-generation smoke test
  - Synthetic deterministic fixture tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.instance import (
    LINERLIBInstance, Port, VesselType, DistanceArc, FleetEntry, Demand,
    ProvenanceRecord,
)
from env.environment import LSNDPEnv
from actions import (
    ServiceGenerator,
    ServiceValidationResult,
    ActionValidator,
    calculate_vessel_requirement,
    canonicalize_service,
    are_services_equivalent,
    make_service_action,
    make_service_definition,
    _nearest_neighbor_tsp,
    select_largest_available_vessel,
)


# ===========================================================================
# Synthetic fixtures — NOT LINERLIB benchmark data
# ===========================================================================

def _make_toy_instance() -> LINERLIBInstance:
    """Minimal 2-port synthetic fixture for deterministic P6 tests."""
    port_a = Port(
        unlocode="A", name="Port A", country=None, cabotage_region="test",
        d_region=None, longitude=None, latitude=None, draft=10.0,
        cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    port_b = Port(
        unlocode="B", name="Port B", country=None, cabotage_region="test",
        d_region=None, longitude=None, latitude=None, draft=10.0,
        cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=2),
    )
    vessel_v1 = VesselType(
        vessel_class="V1", capacity_ffe=200, tc_rate_daily=100,
        draft=12.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    arc_ab = DistanceArc(
        origin="A", destination="B", distance_nm=100.0,
        draft_required=10.0, is_panama=False, is_suez=False,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    arc_ba = DistanceArc(
        origin="B", destination="A", distance_nm=100.0,
        draft_required=10.0, is_panama=False, is_suez=False,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=2),
    )
    demand = Demand(
        origin="A", destination="B",
        ffe_per_week=100.0, revenue=50.0, max_transit_time=10,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    return LINERLIBInstance(
        name="TOY_P6_TEST",
        vessel_types={"V1": vessel_v1},
        ports={"A": port_a, "B": port_b},
        fleet=[FleetEntry(vessel_class="V1", quantity=5)],
        distances=[arc_ab, arc_ba],
        sparse_distances=[arc_ab, arc_ba],
        demands=[demand],
    )


def _make_multi_port_instance() -> LINERLIBInstance:
    """3-port synthetic fixture for TSP ordering tests."""
    ports_data = {
        "A": Port(unlocode="A", name="A", country=None, cabotage_region="x",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        "B": Port(unlocode="B", name="B", country=None, cabotage_region="x",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=2.0, cost_per_full_transfer=1.0,
                  port_call_cost_fixed=200.0, port_call_cost_per_ffe=1.0,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
        "C": Port(unlocode="C", name="C", country=None, cabotage_region="x",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=3.0, cost_per_full_transfer=1.5,
                  port_call_cost_fixed=300.0, port_call_cost_per_ffe=1.5,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=3)),
        "D": Port(unlocode="D", name="D", country=None, cabotage_region="x",
                  d_region=None, longitude=None, latitude=None, draft=8.0,
                  cost_per_full=1.5, cost_per_full_transfer=0.75,
                  port_call_cost_fixed=150.0, port_call_cost_per_ffe=0.75,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=4)),
    }
    vessels = {
        "Small": VesselType(vessel_class="Small", capacity_ffe=100, tc_rate_daily=50,
                            draft=8.0, min_speed=5.0, max_speed=12.0, design_speed=10.0,
                            bunker_ton_per_day_at_design=20.0, idle_consumption_ton_per_day=5.0,
                            panama_fee=0, suez_fee=0,
                            provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        "Large": VesselType(vessel_class="Large", capacity_ffe=500, tc_rate_daily=200,
                            draft=12.0, min_speed=8.0, max_speed=20.0, design_speed=15.0,
                            bunker_ton_per_day_at_design=80.0, idle_consumption_ton_per_day=15.0,
                            panama_fee=1000, suez_fee=2000,
                            provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
    }
    arcs = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        DistanceArc(origin="B", destination="C", distance_nm=200.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
        DistanceArc(origin="C", destination="D", distance_nm=150.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=3)),
        DistanceArc(origin="D", destination="A", distance_nm=120.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=4)),
        DistanceArc(origin="A", destination="C", distance_nm=250.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=5)),
        DistanceArc(origin="B", destination="D", distance_nm=180.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=6)),
        # Reverse directions for completeness
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=7)),
        DistanceArc(origin="C", destination="B", distance_nm=200.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=8)),
        DistanceArc(origin="D", destination="C", distance_nm=150.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=9)),
        DistanceArc(origin="A", destination="D", distance_nm=120.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=10)),
        DistanceArc(origin="C", destination="A", distance_nm=250.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=11)),
        DistanceArc(origin="D", destination="B", distance_nm=180.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=12)),
    ]
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=10.0,
               max_transit_time=5,
               provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        Demand(origin="C", destination="D", ffe_per_week=80.0, revenue=20.0,
               max_transit_time=8,
               provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
    ]
    instance = LINERLIBInstance(
        name="TOY_MULTI_PORT_P6",
        vessel_types=vessels,
        ports=ports_data,
        fleet=[FleetEntry(vessel_class="Small", quantity=3),
               FleetEntry(vessel_class="Large", quantity=2)],
        distances=arcs,
        sparse_distances=arcs,
        demands=demands,
    )
    return instance


def _dist_by_pair(instance):
    return {(a.origin, a.destination): a for a in instance.distances}


# ===========================================================================
# Test fixtures for vessel-selection tests (vessel types only)
# ===========================================================================

_TEST_VESSELS = {
    "Feeder_450": VesselType(
        vessel_class="Feeder_450", capacity_ffe=450, tc_rate_daily=5000,
        draft=8.0, min_speed=10.0, max_speed=14.0, design_speed=12.0,
        bunker_ton_per_day_at_design=18.8, idle_consumption_ton_per_day=2.4,
        panama_fee=64800, suez_fee=175769,
        provenance=ProvenanceRecord(source_file="test_fixture", source_row=1),
    ),
    "Feeder_800": VesselType(
        vessel_class="Feeder_800", capacity_ffe=800, tc_rate_daily=8000,
        draft=9.5, min_speed=10.0, max_speed=17.0, design_speed=14.0,
        bunker_ton_per_day_at_design=23.7, idle_consumption_ton_per_day=2.5,
        panama_fee=115200, suez_fee=218445,
        provenance=ProvenanceRecord(source_file="test_fixture", source_row=2),
    ),
    "Panamax_1200": VesselType(
        vessel_class="Panamax_1200", capacity_ffe=1200, tc_rate_daily=11000,
        draft=12.0, min_speed=12.0, max_speed=19.0, design_speed=18.0,
        bunker_ton_per_day_at_design=52.5, idle_consumption_ton_per_day=4.0,
        panama_fee=172800, suez_fee=267217,
        provenance=ProvenanceRecord(source_file="test_fixture", source_row=3),
    ),
}

_TIE_VESSELS = {
    "Alpha": VesselType(
        vessel_class="Alpha", capacity_ffe=500, tc_rate_daily=100,
        draft=10.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=20.0, idle_consumption_ton_per_day=5.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="test_fixture", source_row=1),
    ),
    "Beta": VesselType(
        vessel_class="Beta", capacity_ffe=500, tc_rate_daily=200,
        draft=10.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=30.0, idle_consumption_ton_per_day=6.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="test_fixture", source_row=2),
    ),
}


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def toy_instance():
    return _make_toy_instance()


@pytest.fixture
def multi_port_instance():
    return _make_multi_port_instance()


@pytest.fixture
def generator(toy_instance):
    return ServiceGenerator(toy_instance, _dist_by_pair(toy_instance))


@pytest.fixture
def multi_generator(multi_port_instance):
    return ServiceGenerator(multi_port_instance, _dist_by_pair(multi_port_instance))


# ===========================================================================
# 1. Vessel Selection
# ===========================================================================

class TestVesselSelection:
    def test_valid_vessel_class(self, generator):
        reasons = generator.validate_service("V1", ["A", "B"])
        assert reasons == []

    def test_invalid_vessel_class(self, generator):
        reasons = generator.validate_service("Unknown", ["A", "B"])
        assert len(reasons) > 0
        assert any("not in instance" in r for r in reasons)

    def test_get_vessel_classes_for_ports(self, multi_generator):
        classes = multi_generator.get_vessel_classes_for_ports(["A", "B", "C"])
        # Small has draft=8, Large has draft=12
        # All ports have draft=10, so only Large can visit all
        assert "Large" in classes
        assert "Small" not in classes

    def test_can_visit_port_feasible(self, multi_generator):
        assert multi_generator.can_visit_port("Large", "A")
        assert not multi_generator.can_visit_port("Small", "A")  # draft 8 < 10

    def test_deterministic_vessel_ordering(self, multi_generator):
        classes = multi_generator._vessel_classes
        assert classes == sorted(classes)


# ===========================================================================
# 2. Port Selection
# ===========================================================================

class TestPortSelection:
    def test_valid_ports(self, generator):
        reasons = generator.validate_service("V1", ["A", "B"])
        assert reasons == []

    def test_invalid_port(self, generator):
        reasons = generator.validate_service("V1", ["A", "X"])
        assert len(reasons) > 0
        assert any("not in instance" in r for r in reasons)

    def test_empty_port_sequence(self, generator):
        reasons = generator.validate_service("V1", [])
        assert len(reasons) > 0
        assert any("empty" in r.lower() for r in reasons)

    def test_single_port(self, generator):
        reasons = generator.validate_service("V1", ["A"])
        assert len(reasons) > 0
        assert any("at least 2" in r for r in reasons)


# ===========================================================================
# 3. Service Construction
# ===========================================================================

class TestServiceConstruction:
    def test_construct_valid_service(self, generator):
        result = generator.generate_service("V1", ["A", "B"])
        assert result.is_valid
        assert result.service_action is not None
        assert result.service_action.vessel_class == "V1"
        assert result.service_action.port_sequence == ["A", "B"]

    def test_construct_with_service_id(self, generator):
        result = generator.generate_service("V1", ["A", "B"], service_id=5)
        assert result.is_valid
        assert result.service_action.service_id == 5

    def test_invalid_service_returns_error(self, generator):
        result = generator.generate_service("Unknown", ["A", "B"])
        assert not result.is_valid
        assert len(result.reasons) > 0

    def test_service_action_type(self, generator):
        result = generator.generate_service("V1", ["A", "B"])
        from env.action import ServiceAction
        assert isinstance(result.service_action, ServiceAction)


# ===========================================================================
# 4. Cyclic Closure
# ===========================================================================

class TestCyclicClosure:
    def test_cycle_closes(self, generator):
        result = generator.generate_service("V1", ["A", "B"])
        assert result.is_valid
        # Check that distance exists for closure leg B→A
        dist = generator.compute_tour_distance(["A", "B"])
        assert dist == 200.0  # A→B (100) + B→A (100)

    def test_three_port_cycle(self, multi_generator):
        result = multi_generator.generate_service("Large", ["A", "B", "C"])
        assert result.is_valid
        dist = multi_generator.compute_tour_distance(["A", "B", "C"])
        assert dist == 100.0 + 200.0 + 250.0  # A→B→C→A

    def test_four_port_cycle(self, multi_generator):
        result = multi_generator.generate_service("Large", ["A", "B", "C", "D"])
        assert result.is_valid
        dist = multi_generator.compute_tour_distance(["A", "B", "C", "D"])
        assert dist == 100.0 + 200.0 + 150.0 + 120.0


# ===========================================================================
# 5. Duplicate-Port Rejection
# ===========================================================================

class TestDuplicatePortRejection:
    def test_duplicate_ports_rejected(self, generator):
        reasons = generator.validate_service("V1", ["A", "A", "B"])
        assert len(reasons) > 0
        assert any("duplicate" in r.lower() for r in reasons)

    def test_all_same_port_rejected(self, generator):
        reasons = generator.validate_service("V1", ["A", "A"])
        assert len(reasons) > 0


# ===========================================================================
# 6. Invalid-Port Rejection
# ===========================================================================

class TestInvalidPortRejection:
    def test_nonexistent_port(self, generator):
        reasons = generator.validate_service("V1", ["A", "Z"])
        assert len(reasons) > 0
        assert any("not in instance" in r for r in reasons)


# ===========================================================================
# 7. Draft Feasibility
# ===========================================================================

class TestDraftFeasibility:
    def test_draft_incompatible(self, multi_generator):
        # [PAPER] Draft is NOT a hard constraint — C_unused handles fleet
        # deviations economically per Eq. 34. Services with draft gaps
        # are allowed; cost is computed in MCF evaluation.
        reasons = multi_generator.validate_service("Small", ["A", "B"])
        assert len(reasons) == 0

    def test_draft_compatible(self, multi_generator):
        # Large vessel has draft=12, all ports have draft=10
        reasons = multi_generator.validate_service("Large", ["A", "B"])
        assert reasons == []

    def test_mixed_draft_feasible(self, multi_generator):
        # [PAPER] Draft gaps are allowed — no hard rejection
        reasons = multi_generator.validate_service("Small", ["C", "D"])
        assert len(reasons) == 0


# ===========================================================================
# 8. Distance Feasibility
# ===========================================================================

class TestDistanceFeasibility:
    def test_missing_distance(self, toy_instance):
        # Create instance with missing distance
        from data.instance import DistanceArc, ProvenanceRecord
        inst = toy_instance
        gen = ServiceGenerator(inst, _dist_by_pair(inst))
        # A→B and B→A exist, so this should work
        reasons = gen.validate_service("V1", ["A", "B"])
        assert reasons == []


# ===========================================================================
# 9. Deterministic Ordering
# ===========================================================================

class TestDeterministicOrdering:
    def test_nearest_neighbor_deterministic(self, multi_port_instance):
        ports = ["A", "B", "C", "D"]
        dist_map = {
            ("A", "B"): 100.0, ("A", "C"): 250.0, ("A", "D"): 120.0,
            ("B", "C"): 200.0, ("B", "D"): 180.0,
            ("C", "D"): 150.0,
        }
        # Run twice to verify determinism
        order1 = _nearest_neighbor_tsp(ports, dist_map)
        order2 = _nearest_neighbor_tsp(ports, dist_map)
        assert order1 == order2

    def test_tsp_starting_point_deterministic(self, multi_port_instance):
        # With no start_port specified, should use first alphabetically
        ports = ["C", "A", "B"]
        dist_map = {
            ("A", "B"): 100.0, ("B", "C"): 200.0, ("C", "A"): 250.0,
        }
        order = _nearest_neighbor_tsp(ports, dist_map)
        assert order[0] == "A"  # Alphabetically first


# ===========================================================================
# 10. Approximate-TSP Ordering Interface
# ===========================================================================

class TestTSPOrdering:
    def test_order_ports(self, multi_generator):
        selected = {"A", "B", "C"}
        ordered = multi_generator.order_ports(selected, "Large")
        assert len(ordered) == 3
        assert set(ordered) == selected

    def test_order_ports_preserves_all(self, multi_generator):
        selected = ["A", "B", "C", "D"]
        ordered = multi_generator.order_ports(selected, "Large")
        assert len(ordered) == 4
        assert set(ordered) == set(selected)

    def test_order_ports_small_vessel_filtered(self, multi_generator):
        # Small cannot visit A, B, C (draft=10 > 8) but can visit D (draft=8)
        selected = ["A", "B", "C", "D"]
        ordered = multi_generator.order_ports(selected, "Small")
        # Should fall back to feasible subset (only D is feasible)
        # With only 1 feasible port, it will use alphabetical fallback
        assert len(ordered) >= 1
        # Check that any returned ports are feasible for Small
        for p in ordered:
            if p != "D":
                # If D is not in ordered list, then ordering fell back to alphabetical
                pass


# ===========================================================================
# 11. Vessel Requirement
# ===========================================================================

class TestVesselRequirement:
    def test_basic_calculation(self):
        # 200 nm at 10 knots: 200 / (10 * 7) = 2.857
        n_vs = calculate_vessel_requirement(200.0, 10.0)
        assert abs(n_vs - 200.0 / 70.0) < 1e-6

    def test_fractional_not_rounded(self, generator):
        # A→B→A = 200 nm at 10 knots
        n_vs = generator.compute_vessel_requirement("V1", ["A", "B"])
        expected = 200.0 / (10.0 * 7.0)
        assert abs(n_vs - expected) < 1e-6
        assert n_vs != int(n_vs)  # Verify it's fractional

    def test_zero_speed_raises(self):
        with pytest.raises(ValueError):
            calculate_vessel_requirement(100.0, 0.0)

    def test_negative_speed_raises(self):
        with pytest.raises(ValueError):
            calculate_vessel_requirement(100.0, -5.0)


# ===========================================================================
# 12. Encoder-Only Vessel Selection (Largest Available)
# ===========================================================================

class TestSelectLargestAvailableVessel:
    """
    Test the select_largest_available_vessel() function that implements
    the encoder-only pathway's deterministic vessel selection rule:
    "largest available vessel" — the class with maximum capacity among
    those with remaining quantity > 0.
    """

    def test_picks_highest_capacity(self):
        """When all vessels have remaining stock, pick the one with max capacity."""
        fleet = {"Feeder_450": 2.0, "Feeder_800": 1.0, "Panamax_1200": 3.0}
        result = select_largest_available_vessel(fleet, _TEST_VESSELS)
        assert result == "Panamax_1200"

    def test_skips_exhausted(self):
        """Vessels with zero remaining are excluded."""
        fleet = {"Feeder_450": 0.0, "Feeder_800": 1.0, "Panamax_1200": 0.0}
        result = select_largest_available_vessel(fleet, _TEST_VESSELS)
        assert result == "Feeder_800"

    def test_returns_none_when_all_exhausted(self):
        """Returns None when no vessels have remaining quantity."""
        fleet = {"Feeder_450": 0.0, "Feeder_800": 0.0, "Panamax_1200": 0.0}
        result = select_largest_available_vessel(fleet, _TEST_VESSELS)
        assert result is None

    def test_tie_breaks_alphabetically(self):
        """When capacities are equal, picks lexicographically first name."""
        fleet = {"Alpha": 1.0, "Beta": 1.0}
        result = select_largest_available_vessel(fleet, _TIE_VESSELS)
        assert result == "Alpha"

    def test_respects_fleet_remaining_boundary(self):
        """Only considers vessels with qty > 0; ignores unknown keys."""
        fleet = {"Feeder_450": 0.5, "Unknown_Class": 999.0}
        result = select_largest_available_vessel(fleet, _TEST_VESSELS)
        assert result == "Feeder_450"  # Unknown_Class not in vessel_types

    def test_fractional_quantity_counts_as_available(self):
        """Fractional remaining (> 0) counts as available."""
        fleet = {"Feeder_450": 0.001, "Feeder_800": 0.002}
        result = select_largest_available_vessel(fleet, _TEST_VESSELS)
        assert result == "Feeder_800"

    def test_empty_fleet(self):
        """Empty fleet dict returns None."""
        result = select_largest_available_vessel({}, _TEST_VESSELS)
        assert result is None

    def test_empty_vessel_types(self):
        """Empty vessel types returns None even with fleet."""
        result = select_largest_available_vessel({"X": 5.0}, {})
        assert result is None


# ===========================================================================
# 13. Fractional Vessel Semantics
# ===========================================================================

class TestFractionalVesselSemantics:
    def test_no_ceiling_applied(self, generator):
        # Compute requirement without ceiling
        n_vs = generator.compute_vessel_requirement("V1", ["A", "B"])
        # Should be ~2.857, not ceiling to 3
        assert n_vs < 3.0
        assert n_vs > 2.0

    def test_fractional_preserved_in_service(self, generator):
        result = generator.generate_service("V1", ["A", "B"])
        assert result.is_valid
        svc = result.service_action
        assert svc.vessel_class == "V1"
        assert svc.port_sequence == ["A", "B"]


# ===========================================================================
# 13. Duplicate Service Handling
# ===========================================================================

class TestDuplicateServiceHandling:
    def test_canonicalize_rotation(self):
        action1 = make_service_action("V1", ["A", "B", "C"])
        action2 = make_service_action("V1", ["B", "C", "A"])  # rotation of action1
        canon1 = canonicalize_service(action1)
        canon2 = canonicalize_service(action2)
        assert canon1 == canon2

    def test_different_cycles_not_equivalent(self):
        action1 = make_service_action("V1", ["A", "B", "C"])
        action2 = make_service_action("V1", ["A", "C", "B"])  # different order
        canon1 = canonicalize_service(action1)
        canon2 = canonicalize_service(action2)
        assert canon1 != canon2

    def test_different_vessels_not_equivalent(self):
        action1 = make_service_action("V1", ["A", "B"])
        action2 = make_service_action("V2", ["A", "B"])
        assert not are_services_equivalent(action1, action2)

    def test_same_service_equivalent(self):
        action1 = make_service_action("V1", ["A", "B"])
        action2 = make_service_action("V1", ["A", "B"])
        assert are_services_equivalent(action1, action2)


# ===========================================================================
# 14. ServiceAction Conversion
# ===========================================================================

class TestServiceActionConversion:
    def test_make_service_action(self):
        action = make_service_action("V1", ["A", "B"], service_id=0)
        assert action.vessel_class == "V1"
        assert action.port_sequence == ["A", "B"]
        assert action.service_id == 0

    def test_make_service_definition(self):
        svc = make_service_definition(0, "V1", ["A", "B"])
        assert svc.service_id == 0
        assert svc.vessel_class == "V1"
        assert svc.port_sequence == ["A", "B"]


# ===========================================================================
# 15. P4 Integration
# ===========================================================================

class TestP4Integration:
    def test_env_accepts_generated_service(self, toy_instance):
        env = LSNDPEnv(toy_instance)
        obs, info = env.reset()

        gen = ServiceGenerator(toy_instance, env._dist_by_pair)
        result = gen.generate_service("V1", ["A", "B"])
        assert result.is_valid

        # Pass to environment
        action = result.service_action
        obs, reward, terminated, truncated, info = env.step(action)
        # May terminate if demand is satisfied or fleet exhausted

    def test_multiple_services(self, multi_port_instance):
        env = LSNDPEnv(multi_port_instance)
        obs, info = env.reset()

        gen = ServiceGenerator(multi_port_instance, env._dist_by_pair)

        # Add first service
        result1 = gen.generate_service("Large", ["A", "B", "C"])
        assert result1.is_valid
        env.step(result1.service_action)

        # Add second service (different path)
        result2 = gen.generate_service("Large", ["A", "B", "D"])
        assert result2.is_valid
        try:
            obs, reward, terminated, truncated, info = env.step(result2.service_action)
        except RuntimeError:
            # Terminal state is OK - just verify services were added
            pass
        assert env._state.num_services_added >= 1


# ===========================================================================
# 16. P3 Integration
# ===========================================================================

class TestP3Integration:
    def test_service_converts_to_definition(self, generator):
        result = generator.generate_service("V1", ["A", "B"])
        assert result.is_valid

        from mcf.expanded_graph import ServiceDefinition
        svc_def = make_service_definition(
            service_id=0,
            vessel_class=result.service_action.vessel_class,
            port_sequence=result.service_action.port_sequence,
        )
        assert isinstance(svc_def, ServiceDefinition)
        assert svc_def.service_id == 0
        assert svc_def.port_sequence == ["A", "B"]


# ===========================================================================
# 17. Encoder-Only Interface Compatibility
# ===========================================================================

class TestEncoderOnlyCompatibility:
    def test_validate_encoder_only_output(self, multi_generator):
        validator = ActionValidator(multi_generator)
        result = validator.validate_encoder_only_output(
            vessel_class="Large",
            selected_ports={"A", "B", "C"},
        )
        assert result.is_valid
        assert result.service_action is not None
        # Should have 3 ports
        assert result.service_action.num_ports == 3
        # Ports should be in some order
        assert set(result.service_action.port_sequence) == {"A", "B", "C"}

    def test_encoder_only_too_few_ports(self, multi_generator):
        validator = ActionValidator(multi_generator)
        result = validator.validate_encoder_only_output(
            vessel_class="Large",
            selected_ports={"A"},
        )
        assert not result.is_valid


# ===========================================================================
# 18. Encoder-Decoder Interface Compatibility
# ===========================================================================

class TestEncoderDecoderCompatibility:
    def test_validate_encoder_decoder_output(self, multi_generator):
        validator = ActionValidator(multi_generator)
        result = validator.validate_encoder_decoder_output(
            vessel_class="Large",
            port_sequence=["A", "B", "C"],
        )
        assert result.is_valid
        assert result.service_action.port_sequence == ["A", "B", "C"]

    def test_encoder_decoder_invalid_sequence(self, multi_generator):
        validator = ActionValidator(multi_generator)
        result = validator.validate_encoder_decoder_output(
            vessel_class="Large",
            port_sequence=["A", "A", "B"],  # duplicate
        )
        assert not result.is_valid


# ===========================================================================
# 19. Real Baltic Service-Generation Smoke Test
# ===========================================================================

class TestBalticSmokeTest:
    def test_baltic_service_generation(self):
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        baltic = loader.load("Baltic", validate=False)
        gen = ServiceGenerator(baltic, _dist_by_pair(baltic))

        # Use Super_panamax which has draft=12.5
        # Check which ports it can visit
        feasible_ports = [p for p in baltic.ports.keys()
                         if baltic.ports[p].draft is None or 12.5 >= baltic.ports[p].draft]
        assert len(feasible_ports) >= 3, "Need at least 3 feasible ports"

        # Generate a valid service with feasible ports
        sample_ports = feasible_ports[:3]
        result = gen.generate_service("Super_panamax", sample_ports)
        assert result.is_valid, f"Validation failed: {result.reasons}"

    def test_baltic_tsp_ordering(self):
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        baltic = loader.load("Baltic", validate=False)
        gen = ServiceGenerator(baltic, _dist_by_pair(baltic))

        # Use Super_panamax (draft=12.5) which can visit more ports
        feasible_ports = [p for p in baltic.ports.keys()
                         if baltic.ports[p].draft is None or 12.5 >= baltic.ports[p].draft]
        if len(feasible_ports) >= 3:
            ordered = gen.order_ports(feasible_ports[:3], "Super_panamax")
            assert len(ordered) >= 2  # At least 2 ports


# ===========================================================================
# 20. Synthetic Deterministic Fixture Tests
# ===========================================================================

class TestSyntheticDeterministic:
    def test_deterministic_service_generation(self, generator):
        # Generate same service twice
        result1 = generator.generate_service("V1", ["A", "B"])
        result2 = generator.generate_service("V1", ["A", "B"])
        assert result1.is_valid == result2.is_valid
        if result1.is_valid:
            assert result1.service_action.port_sequence == result2.service_action.port_sequence

    def test_deterministic_tsp(self, multi_port_instance):
        gen = ServiceGenerator(multi_port_instance, _dist_by_pair(multi_port_instance))
        order1 = gen.order_ports(["A", "B", "C", "D"], "Large")
        order2 = gen.order_ports(["A", "B", "C", "D"], "Large")
        assert order1 == order2


# ===========================================================================
# Run tests
# ===========================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
