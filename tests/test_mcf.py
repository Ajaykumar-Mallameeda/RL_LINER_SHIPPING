"""
P3 – MCF / Network Evaluation Engine unit tests.

Covers the 10+ required test scenarios plus Baltic numerical validation
and a real LINERLIB smoke test.

SYNTHETIC fixtures are used for deterministic unit tests; real LINERLIB
data is only used for the integration smoke test (clearly labeled).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.instance import Demand, DistanceArc, FleetEntry, InstanceMetadata
from data.instance import LINERLIBInstance, Port, ProvenanceRecord, VesselType, DatasetProvenance
from data.linerlib_loader import LINERLIBLoader
from mcf import ServiceDefinition, evaluate_network, FlowSolver
from tests.fixtures.toy_p2_fixture import make_toy_2port_instance, compute_toy_expected_profit


# ===========================================================================
# Helpers
# ===========================================================================

def _make_simple_instance():
    """Minimal 2-port instance for basic routing tests."""
    ports = {
        "A": Port(
            unlocode="A", name="Port A", country="X", cabotage_region="X",
            d_region=None, longitude=0.0, latitude=0.0, draft=10.0,
            cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.1,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
        "B": Port(
            unlocode="B", name="Port B", country="Y", cabotage_region="Y",
            d_region=None, longitude=1.0, latitude=1.0, draft=10.0,
            cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.1,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=2),
        ),
        "C": Port(
            unlocode="C", name="Port C", country="Z", cabotage_region="Z",
            d_region=None, longitude=2.0, latitude=2.0, draft=10.0,
            cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.1,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=3),
        ),
    }
    vessels = {
        "V1": VesselType(
            vessel_class="V1", capacity_ffe=100, tc_rate_daily=10,
            draft=10.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
            bunker_ton_per_day_at_design=5.0, idle_consumption_ton_per_day=1.0,
            panama_fee=0, suez_fee=0,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
    }
    return ports, vessels


def _build_instance(ports, vessels, demands, distances, fleet, name="TEST"):
    metadata = InstanceMetadata(
        name=name, active_port_count=len(ports),
        vessel_type_count=len(vessels), total_vessels=sum(e.quantity for e in fleet),
        demand_count=len(demands), distance_arc_count=len(distances),
    )
    return LINERLIBInstance(
        name=name, ports=ports, vessel_types=vessels, demands=demands,
        distances=distances, fleet=fleet, metadata=metadata,
        provenance=DatasetProvenance(source_root="[P3 TEST]"),
    )


def _service(instance, sid, vclass, ports_seq):
    return ServiceDefinition(service_id=sid, vessel_class=vclass, port_sequence=ports_seq)


def _vreq(services, instance):
    """Compute vessel requirements for given services."""
    vreqs = {}
    dist_by_pair = {(a.origin, a.destination): a for a in instance.distances}
    for svc in services:
        tour_dist = 0.0
        for i in range(len(svc.port_sequence)):
            p_from = svc.port_sequence[i]
            p_to = svc.port_sequence[(i + 1) % len(svc.port_sequence)]
            arc = dist_by_pair.get((p_from, p_to))
            if arc is None:
                raise ValueError(f"No distance for {p_from}->{p_to}")
            tour_dist += arc.distance_nm
        vt = instance.vessel_types[svc.vessel_class]
        n_vs = tour_dist / (vt.design_speed * 7)
        if svc.service_id not in vreqs:
            vreqs[svc.service_id] = {}
        vreqs[svc.service_id][svc.vessel_class] = n_vs
    return vreqs


# ===========================================================================
# Test 1 — No feasible path
# ===========================================================================

def test_no_feasible_path():
    """Demand exists but no service connects origin to destination."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    # No distances at all
    distances = []
    fleet = [FleetEntry(vessel_class="V1", quantity=2)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    result = evaluate_network(inst, [], {})

    assert result.routed_demand == 0.0
    assert result.rejected_demand == pytest.approx(50.0, abs=1e-6)
    assert result.demand_coverage == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# Test 2 — Fully routable demand
# ===========================================================================

def test_fully_routable():
    """Single demand fully satisfied within capacity."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=10)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    assert result.rejected_demand == 0.0
    assert result.routed_demand == pytest.approx(10.0, abs=1e-6)
    assert result.demand_coverage == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# Test 3 — Partial capacity (demand > available path capacity)
# ===========================================================================

def test_partial_capacity():
    """Demand exceeds path bottleneck; some rejected."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=500.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    # n_vs = 200/(10*7) = 2.857; capacity per edge = 2.857 * 100 = 285.71 FFE/week
    expected_cap = (200.0 / (10.0 * 7.0)) * 100.0
    assert result.routed_demand == pytest.approx(expected_cap, abs=1e-6)
    assert result.rejected_demand == pytest.approx(500.0 - expected_cap, abs=1e-6)
    assert 0 < result.demand_coverage < 1.0


# ===========================================================================
# Test 4 — Bottleneck edge determines path capacity
# ===========================================================================

def test_bottleneck_edge():
    """Two-edge path where one edge is the bottleneck."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="C", ffe_per_week=1000.0, revenue=100.0,
               max_transit_time=10, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
        DistanceArc(origin="B", destination="C", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 3)),
        DistanceArc(origin="C", destination="B", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 4)),
        DistanceArc(origin="C", destination="A", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 5)),
        DistanceArc(origin="A", destination="C", distance_nm=50.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 6)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    # Service A -> B -> C -> A: tour_dist = 150, n_vs = 150/70 = 2.143
    svc = _service(inst, "svc_0", "V1", ["A", "B", "C"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    n_vs = 150.0 / (10.0 * 7.0)
    expected_cap = n_vs * 100.0
    assert result.routed_demand == pytest.approx(expected_cap, abs=1e-6)
    assert result.rejected_demand == pytest.approx(1000.0 - expected_cap, abs=1e-6)


# ===========================================================================
# Test 5 — Capacity update after routing
# ===========================================================================

def test_capacity_update():
    """After routing flow, remaining capacity reflects the subtraction."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=30.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
        Demand(origin="A", destination="B", ffe_per_week=30.0, revenue=80.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 2)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    # Both demands can be fully routed (total 60 < capacity 285.7)
    assert result.routed_demand == pytest.approx(60.0, abs=1e-6)
    assert result.rejected_demand == 0.0

    crs = sorted(result.commodity_results, key=lambda x: -x.revenue_per_ffe)
    assert crs[0].satisfied == pytest.approx(30.0, abs=1e-6)
    assert crs[1].satisfied == pytest.approx(30.0, abs=1e-6)


# ===========================================================================
# Test 6 — Multiple commodities, greedy revenue ordering
# ===========================================================================

def test_multiple_commodities_greedy_ordering():
    """Higher-revenue demand gets served first; lower may be partially rejected."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=200.0, revenue=200.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
        Demand(origin="A", destination="B", ffe_per_week=200.0, revenue=50.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 2)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    n_vs = 200.0 / (10.0 * 7.0)
    cap = n_vs * 100.0  # ~285.71

    # First commodity (rev=200) gets min(200, 285.71) = 200
    # Second commodity (rev=50) gets min(200, 85.71) = 85.71
    crs = sorted(result.commodity_results, key=lambda x: -x.revenue_per_ffe)
    assert crs[0].satisfied == pytest.approx(200.0, abs=1e-6)
    assert crs[0].rejected == 0.0
    assert crs[1].satisfied == pytest.approx(cap - 200.0, abs=1e-6)
    assert crs[1].rejected == pytest.approx(200.0 - (cap - 200.0), abs=1e-6)
    assert result.routed_demand == pytest.approx(cap, abs=1e-6)
    assert result.rejected_demand == pytest.approx(400.0 - cap, abs=1e-6)


# ===========================================================================
# Test 7 — Equal-revenue commodities, deterministic tie-breaking
# ===========================================================================

def test_equal_revenue_deterministic_tiebreak():
    """Equal-revenue demands must produce identical results across runs."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
        Demand(origin="A", destination="C", ffe_per_week=50.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 2)),
    ]
    # All pairwise distances needed for service A->B->C
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
        DistanceArc(origin="A", destination="C", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 3)),
        DistanceArc(origin="C", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 4)),
        DistanceArc(origin="B", destination="C", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 5)),
        DistanceArc(origin="C", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 6)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=10)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B", "C"])
    vreqs = _vreq([svc], inst)

    results = [evaluate_network(inst, [svc], vreqs) for _ in range(3)]

    for i in range(1, 3):
        assert math.isclose(results[0].eta, results[i].eta, abs_tol=1e-9)
        assert math.isclose(results[0].total_revenue, results[i].total_revenue, abs_tol=1e-9)
        assert math.isclose(results[0].rejected_demand, results[i].rejected_demand, abs_tol=1e-9)
        assert results[0].routed_demand == results[i].routed_demand
        assert results[0].num_services == results[i].num_services


# ===========================================================================
# Test 8 — No negative capacity (sanity check)
# ===========================================================================

def test_no_negative_capacity():
    """After all routing, no edge residual capacity should be negative."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    solver = FlowSolver(inst, [svc], vreqs)
    result = solver.solve()

    for cr in result.commodity_results:
        assert cr.satisfied >= -1e-9
        assert cr.rejected >= -1e-9

    assert math.isclose(result.routed_demand + result.rejected_demand,
                        result.total_demand, abs_tol=1e-6)


# ===========================================================================
# Test 9 — Cost decomposition integrity
# ===========================================================================

def test_cost_decomposition():
    """R_total - C_reject - C_handle - C_service - C_unused - C_voyage = eta."""
    inst = make_toy_2port_instance(profitable=False)
    svc = _service(inst, "svc_0", "Toy_V1", ["TOY_A", "TOY_B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    computed_eta = (
        result.total_revenue
        - result.rejection_cost
        - result.handling_cost
        - result.service_cost
        - result.unused_vessel_cost
        - result.voyage_cost
    )

    assert math.isclose(computed_eta, result.eta, abs_tol=1e-6), (
        f"Decomposition mismatch: computed={computed_eta:.6f}, stored={result.eta:.6f}"
    )


# ===========================================================================
# Test 10 — P2 toy fixture validation
# ===========================================================================

@pytest.mark.parametrize("profitable", [False, True])
def test_p2_toy_fixture(profitable: bool):
    """Reproduce the approved P2 toy result exactly."""
    inst = make_toy_2port_instance(profitable=profitable)
    svc = _service(inst, "svc_0", "Toy_V1", ["TOY_A", "TOY_B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)
    expected = compute_toy_expected_profit(profitable)

    assert math.isclose(result.eta, expected["eta"], abs_tol=1e-6), (
        f"eta mismatch: got {result.eta:.6f}, expected {expected['eta']:.6f}"
    )
    assert math.isclose(result.total_revenue, expected["R_total"], abs_tol=1e-6)
    assert math.isclose(result.rejection_cost, expected["C_reject"], abs_tol=1e-6)
    assert math.isclose(result.handling_cost, expected["C_handle"], abs_tol=1e-6)
    assert math.isclose(result.service_cost, expected["C_service"], abs_tol=1e-6)
    assert math.isclose(result.unused_vessel_cost, expected["C_unused"], abs_tol=1e-6)
    assert math.isclose(result.voyage_cost, expected["C_voyage"], abs_tol=1e-6)


# ===========================================================================
# Test 11 — Zero demand
# ===========================================================================

def test_zero_demand():
    """Instance with zero total demand produces zero revenue, zero routing."""
    ports, vessels = _make_simple_instance()
    demands = []
    distances = []
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    result = evaluate_network(inst, [], {})

    assert result.total_demand == 0.0
    assert result.routed_demand == 0.0
    assert result.rejected_demand == 0.0
    assert result.total_revenue == 0.0


# ===========================================================================
# Test 12 — Disconnected network
# ===========================================================================

def test_disconnected_network():
    """Two disconnected components; demand between them cannot be routed."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="C", ffe_per_week=50.0, revenue=100.0,
               max_transit_time=10, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=10)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    assert result.routed_demand == 0.0
    assert result.rejected_demand == pytest.approx(50.0, abs=1e-6)


# ===========================================================================
# Test 13 — Transshipment between services
# ===========================================================================

def test_transshipment_between_services():
    """Demand A->C routed via transshipment at B across two services."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="C", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=10, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
        DistanceArc(origin="B", destination="C", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 3)),
        DistanceArc(origin="C", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 4)),
        DistanceArc(origin="A", destination="C", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 5)),
        DistanceArc(origin="C", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 6)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=10)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc1 = _service(inst, "svc_0", "V1", ["A", "B"])
    svc2 = _service(inst, "svc_1", "V1", ["B", "C"])
    vreqs = _vreq([svc1, svc2], inst)

    result = evaluate_network(inst, [svc1, svc2], vreqs)

    # Direct path A->C exists (distance provided); should route directly
    assert result.routed_demand == pytest.approx(10.0, abs=1e-6)
    assert result.rejected_demand == 0.0


# ===========================================================================
# Test 14 — Multiple parallel service edges
# ===========================================================================

def test_multiple_parallel_services():
    """Two services on same route; combined capacity serves more demand."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=150.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=10)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc1 = _service(inst, "svc_0", "V1", ["A", "B"])
    svc2 = _service(inst, "svc_1", "V1", ["A", "B"])
    vreqs = _vreq([svc1, svc2], inst)

    result = evaluate_network(inst, [svc1, svc2], vreqs)

    # Combined capacity ~571 FFE/week; demand 150 should be fully met
    assert result.routed_demand == pytest.approx(150.0, abs=1e-6)
    assert result.rejected_demand == 0.0


# ===========================================================================
# Test 15 — Determinism across repeated runs
# ===========================================================================

def test_determinism():
    """Three consecutive runs must produce identical results."""
    inst = make_toy_2port_instance(profitable=True)
    svc = _service(inst, "svc_0", "Toy_V1", ["TOY_A", "TOY_B"])
    vreqs = _vreq([svc], inst)

    results = [evaluate_network(inst, [svc], vreqs) for _ in range(3)]

    for i in range(1, 3):
        assert math.isclose(results[0].eta, results[i].eta, abs_tol=1e-9)
        assert math.isclose(results[0].total_revenue, results[i].total_revenue, abs_tol=1e-9)
        assert math.isclose(results[0].rejected_demand, results[i].rejected_demand, abs_tol=1e-9)
        assert results[0].routed_demand == results[i].routed_demand
        assert results[0].num_services == results[i].num_services


# ===========================================================================
# Test 16 — Fractional vessel requirement
# ===========================================================================

def test_fractional_vessel_requirement():
    """n_vs should be fractional, not rounded."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    expected_n = 200.0 / (10.0 * 7.0)
    actual_n = vreqs["svc_0"]["V1"]
    assert math.isclose(actual_n, expected_n, abs_tol=1e-9)
    assert actual_n != round(actual_n)  # confirmed fractional


# ===========================================================================
# Test 17 — Unused fleet (under-utilization produces profit)
# ===========================================================================

def test_unused_fleet_profit():
    """Under-utilized fleet: C_unused should be negative (profit)."""
    # Create an instance where fleet greatly exceeds service need
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    # 100 vessels available; service needs only ~2.86 -> under-utilized
    fleet = [FleetEntry(vessel_class="V1", quantity=100)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    # Under-utilization: C_unused should be negative (profit contribution)
    assert result.unused_vessel_cost < 0, (
        f"Expected negative C_unused for under-utilized fleet, got {result.unused_vessel_cost}"
    )
    # The negative C_unused increases eta (subtracted in the formula)
    # eta = R - C_reject - C_handle - C_service - C_unused - C_voyage


# ===========================================================================
# Test 18 — Fleet over-utilization produces cost
# ===========================================================================

def test_fleet_over_utilization_cost():
    """Over-utilized fleet: C_unused should be positive (cost)."""
    # Create instance with very small fleet
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    # Only 1 vessel available, but service needs 2.857
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    assert result.unused_vessel_cost > 0, (
        f"Expected positive C_unused for over-utilized fleet, got {result.unused_vessel_cost}"
    )


# ===========================================================================
# Test 19 — Baltic numerical validation (component-level formulas)
# ===========================================================================

def test_baltic_component_validation():
    """Validate cost formulas against known Baltic paper values."""
    loader = LINERLIBLoader("data")
    inst = loader.load("Baltic")

    feeder_450 = inst.vessel_types["Feeder_450"]
    feeder_800 = inst.vessel_types["Feeder_800"]

    # LINERLIB solution from Table 1: n_450=3.58, n_800=2.14
    n_450, n_800 = 3.58, 2.14

    c_service = 7.0 * (n_450 * feeder_450.tc_rate_daily + n_800 * feeder_800.tc_rate_daily)
    assert math.isclose(c_service, 245140.0, abs_tol=1.0), (
        f"C_service expected ~245140, got {c_service}"
    )

    fleet_450, fleet_800 = 4, 2
    c_unused = -7.0 * ((fleet_450 - n_450) * feeder_450.tc_rate_daily +
                       (fleet_800 - n_800) * feeder_800.tc_rate_daily)
    assert math.isclose(c_unused, -6860.0, abs_tol=10.0), (
        f"C_unused expected ~-6860, got {c_unused}"
    )

    # RL solution: n_450=4.31, n_800=2.03
    n_450_rl, n_800_rl = 4.31, 2.03
    c_service_rl = 7.0 * (n_450_rl * feeder_450.tc_rate_daily + n_800_rl * feeder_800.tc_rate_daily)
    assert math.isclose(c_service_rl, 264530.0, abs_tol=10.0)

    c_unused_rl = -7.0 * ((fleet_450 - n_450_rl) * feeder_450.tc_rate_daily +
                          (fleet_800 - n_800_rl) * feeder_800.tc_rate_daily)
    assert math.isclose(c_unused_rl, 12530.0, abs_tol=10.0)


# ===========================================================================
# Test 20 — Real LINERLIB smoke test (integration, not benchmark)
# ===========================================================================

def test_real_linerlib_smoke():
    """
    Integration smoke test against real Baltic LINERLIB data.
    NOT a benchmark — validates that the engine runs end-to-end with real data.
    """
    loader = LINERLIBLoader("data")
    inst = loader.load("Baltic")

    # Use a triangle of connected ports: DEBRV -> DKAAR -> FIKTK -> DEBRV
    svc1 = _service(inst, "svc_0", "Feeder_450", ["DEBRV", "DKAAR", "FIKTK"])
    svc2 = _service(inst, "svc_1", "Feeder_800", ["DEBRV", "FIKTK", "DKAAR"])

    vreqs = _vreq([svc1, svc2], inst)

    for sid, reqs in vreqs.items():
        for vc, n in reqs.items():
            assert n > 0, f"Non-positive vessel requirement for {sid}/{vc}: {n}"
            assert math.isfinite(n), f"Non-finite vessel requirement for {sid}/{vc}: {n}"

    result = evaluate_network(inst, [svc1, svc2], vreqs)

    assert math.isfinite(result.eta), f"Non-finite eta: {result.eta}"
    assert result.total_revenue >= 0
    assert result.rejected_demand >= 0
    assert result.service_cost >= 0
    assert result.voyage_cost >= 0
    assert result.num_demands == len(inst.demands)
    assert result.num_services == 2
    assert result.timing_seconds >= 0

    print(f"\n--- Baltic Smoke Test ---")
    print(f"  Services: {result.num_services}")
    print(f"  Demands: {result.num_demands}")
    print(f"  Routed: {result.routed_demand:.1f} / {result.total_demand:.1f} FFE/week")
    print(f"  Coverage: {result.demand_coverage:.1%}")
    print(f"  Revenue: {result.total_revenue:,.0f} USD")
    print(f"  Rejected penalty: {result.rejection_cost:,.0f} USD")
    print(f"  Handling: {result.handling_cost:,.0f} USD")
    print(f"  Service: {result.service_cost:,.0f} USD")
    print(f"  Unused: {result.unused_vessel_cost:,.0f} USD")
    print(f"  Voyage: {result.voyage_cost:,.0f} USD")
    print(f"  ETA: {result.eta:,.0f} USD")
    print(f"  Timing: {result.timing_seconds:.4f}s")
    print(f"-------------------------")


# ===========================================================================
# Test 21 — Large demand
# ===========================================================================

def test_large_demand():
    """Very large demand with limited capacity should reject most of it."""
    ports, vessels = _make_simple_instance()
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=10000.0, revenue=100.0,
               max_transit_time=5, provenance=ProvenanceRecord("test", 1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=5.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord("test", 2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=1)]
    inst = _build_instance(ports, vessels, demands, distances, fleet)

    svc = _service(inst, "svc_0", "V1", ["A", "B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    n_vs = 200.0 / (10.0 * 7.0)
    cap = n_vs * 100.0
    assert result.routed_demand == pytest.approx(cap, abs=1e-6)
    assert result.rejected_demand == pytest.approx(10000.0 - cap, abs=1e-6)
    # Rejection cost should dominate
    assert result.rejection_cost > result.total_revenue


# ===========================================================================
# Test 22 — Service cost sign sanity
# ===========================================================================

def test_service_cost_positive():
    """Service cost should always be non-negative."""
    inst = make_toy_2port_instance(profitable=True)
    svc = _service(inst, "svc_0", "Toy_V1", ["TOY_A", "TOY_B"])
    vreqs = _vreq([svc], inst)

    result = evaluate_network(inst, [svc], vreqs)

    assert result.service_cost > 0
    assert math.isfinite(result.service_cost)


# ===========================================================================
# Test 23 — Empty services list
# ===========================================================================

def test_empty_services_list():
    """Evaluate with no services should route nothing."""
    inst = make_toy_2port_instance(profitable=True)

    result = evaluate_network(inst, [], {})

    assert result.routed_demand == 0.0
    assert result.rejected_demand == pytest.approx(100.0, abs=1e-6)
    assert result.total_revenue == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
