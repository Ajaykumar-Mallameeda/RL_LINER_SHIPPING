"""
P1 – Edge-case tests using synthetic fixtures.

SYNTHETIC TEST FIXTURES only — never mixed with real LINERLIB data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.validation import Validator, Severity
from data.instance import (
    Demand, DistanceArc, FleetEntry, InstanceMetadata,
    LINERLIBInstance, Port, ProvenanceRecord, VesselType,
    DatasetProvenance,
)
from tests.fixtures.synthetic_test_fixture import (
    make_2port_test_instance,
    make_duplicate_od_instance,
)


# ===========================================================================
# Empty / missing-data edge cases
# ===========================================================================

def test_empty_instance_validates():
    """An empty instance (no demands, no distances) should validate cleanly."""
    inst = LINERLIBInstance(name="EMPTY")
    report = Validator().validate(inst)
    assert not report.has_errors()


def test_missing_port_referenced_in_demand():
    """Demand referencing a port absent from the catalogue should ERROR."""
    ports = {
        "AAA": Port(
            unlocode="AAA", name="A", country="X", cabotage_region="X",
            d_region=None, longitude=0.0, latitude=0.0, draft=None,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
            provenance=ProvenanceRecord(source_file="ports.csv", source_row=1),
        ),
    }
    inst = LINERLIBInstance(
        name="MISSING_REF",
        ports=ports,
        demands=[
            Demand(
                origin="AAA", destination="ZZZ_UNKNOWN",
                ffe_per_week=10.0, revenue=100.0, max_transit_time=5,
                provenance=ProvenanceRecord(source_file="test", source_row=1),
            ),
        ],
        metadata=InstanceMetadata(name="MISSING_REF", active_port_count=1,
                                  vessel_type_count=0, total_vessels=0,
                                  demand_count=1, distance_arc_count=0),
    )
    report = Validator().validate(inst)
    unknown_dests = [f for f in report.findings if f.code == "DEM_UNKNOWN_DEST"]
    assert len(unknown_dests) == 1
    assert unknown_dests[0].severity is Severity.ERROR


def test_zero_ffe_detected():
    """Demand with FFEPerWeek == 0 should be an ERROR."""
    inst = make_2port_test_instance()
    inst.demands.append(Demand(
        origin="TESTA", destination="TESTB",
        ffe_per_week=0.0, revenue=100.0, max_transit_time=5,
        provenance=ProvenanceRecord(source_file="synthetic", source_row=99),
    ))
    report = Validator().validate(inst, global_ports=inst.ports)
    zero_ffes = [f for f in report.findings if f.code == "DEM_ZERO_OR_NEG_FFE"]
    assert len(zero_ffes) == 1


def test_negative_revenue_detected():
    inst = make_2port_test_instance()
    inst.demands.append(Demand(
        origin="TESTA", destination="TESTB",
        ffe_per_week=10.0, revenue=-5.0, max_transit_time=5,
        provenance=ProvenanceRecord(source_file="synthetic", source_row=99),
    ))
    report = Validator().validate(inst, global_ports=inst.ports)
    neg_rev = [f for f in report.findings if f.code == "DEM_NEG_REVENUE"]
    assert len(neg_rev) == 1


def test_duplicate_od_warning():
    """Two demands on the same OD pair should generate a WARNING, not ERROR."""
    inst = make_duplicate_od_instance()
    report = Validator().validate(inst, global_ports=inst.ports)
    dup_warnings = [f for f in report.warnings() if f.code == "DEM_DUP_OD_PAIR"]
    assert len(dup_warnings) == 1


def test_invalid_latitude():
    """Port with latitude outside [-90, 90] should ERROR."""
    ports = {
        "BADLAT": Port(
            unlocode="BADLAT", name="Bad Lat", country="X",
            cabotage_region="X", d_region=None,
            longitude=0.0, latitude=100.0, draft=5.0,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
            provenance=ProvenanceRecord(source_file="ports.csv", source_row=1),
        ),
    }
    inst = LINERLIBInstance(name="BAD_LAT", ports=ports)
    report = Validator().validate(inst)
    bad_lats = [f for f in report.findings if f.code == "PORT_INVALID_LAT"]
    assert len(bad_lats) == 1


def test_invalid_longitude():
    ports = {
        "BADLON": Port(
            unlocode="BADLON", name="Bad Lon", country="X",
            cabotage_region="X", d_region=None,
            longitude=-200.0, latitude=0.0, draft=5.0,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
            provenance=ProvenanceRecord(source_file="ports.csv", source_row=1),
        ),
    }
    inst = LINERLIBInstance(name="BAD_LON", ports=ports)
    report = Validator().validate(inst)
    bad_lons = [f for f in report.findings if f.code == "PORT_INVALID_LON"]
    assert len(bad_lons) == 1


def test_negative_distance():
    ports = {
        "NDA": Port(unlocode="NDA", name="NDA", country="X", cabotage_region="X",
                     d_region=None, longitude=0.0, latitude=0.0, draft=5.0,
                     cost_per_full=100.0, cost_per_full_transfer=10.0,
                     port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
                     provenance=ProvenanceRecord(source_file="ports.csv", source_row=1)),
        "NDB": Port(unlocode="NDB", name="NDB", country="Y", cabotage_region="Y",
                     d_region=None, longitude=1.0, latitude=1.0, draft=6.0,
                     cost_per_full=120.0, cost_per_full_transfer=12.0,
                     port_call_cost_fixed=600.0, port_call_cost_per_ffe=3.0,
                     provenance=ProvenanceRecord(source_file="ports.csv", source_row=2)),
    }
    inst = LINERLIBInstance(
        name="NEG_DIST",
        ports=ports,
        distances=[
            DistanceArc(origin="NDA", destination="NDB", distance_nm=-50.0,
                        draft_required=None, is_panama=False, is_suez=False,
                        provenance=ProvenanceRecord(source_file="dist_dense.csv", source_row=1)),
        ],
        metadata=InstanceMetadata(name="NEG_DIST", active_port_count=2,
                                   vessel_type_count=0, total_vessels=0,
                                   demand_count=0, distance_arc_count=1),
    )
    report = Validator().validate(inst)
    neg_dists = [f for f in report.findings if f.code == "DIST_NEGATIVE"]
    assert len(neg_dists) == 1


def test_unknown_fleet_vessel_class():
    """Fleet entry referencing an unknown vessel class should ERROR."""
    inst = make_2port_test_instance()
    inst.fleet.append(FleetEntry(vessel_class="GHOST_VESSEL", quantity=1))
    report = Validator().validate(inst)
    unknown_vessel = [f for f in report.findings if f.code == "FLEET_UNKNOWN_VESSEL_TYPE"]
    assert len(unknown_vessel) == 1


def test_self_loop_distance_not_flagged_as_error():
    """Self-loops in distance are preserved but not flagged as errors."""
    ports = {
        "SLA": Port(unlocode="SLA", name="SLA", country="X", cabotage_region="X",
                     d_region=None, longitude=0.0, latitude=0.0, draft=5.0,
                     cost_per_full=100.0, cost_per_full_transfer=10.0,
                     port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
                     provenance=ProvenanceRecord(source_file="ports.csv", source_row=1)),
    }
    inst = LINERLIBInstance(
        name="SELF_LOOP",
        ports=ports,
        distances=[
            DistanceArc(origin="SLA", destination="SLA", distance_nm=0.0,
                        draft_required=None, is_panama=False, is_suez=False,
                        provenance=ProvenanceRecord(source_file="dist_dense.csv", source_row=1)),
        ],
        metadata=InstanceMetadata(name="SELF_LOOP", active_port_count=1,
                                   vessel_type_count=0, total_vessels=0,
                                   demand_count=0, distance_arc_count=1),
    )
    report = Validator().validate(inst)
    # Self-loops are not validated against anything; no ERROR should fire.
    self_loop_errors = [f for f in report.errors() if "self" in f.code.lower()]
    assert len(self_loop_errors) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
