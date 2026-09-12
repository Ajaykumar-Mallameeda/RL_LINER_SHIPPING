"""
P1 – Tests for validation logic.

Covers:
  - Structural checks
  - Port coordinate validity
  - Vessel capacity validity
  - Distance referential integrity
  - Demand referential integrity
  - Duplicate detection
  - Numeric validity
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.linerlib_loader import LINERLIBLoader
from data.validation import Validator, Severity, DataQualityReport


DATA_ROOT = ROOT / "data" / "LINERLIB-master (1)" / "LINERLIB-master" / "data"


def _loader():
    return LINERLIBLoader(root=str(DATA_ROOT), strict_validation=False)


# ===========================================================================
# Integration tests with real LINERLIB data
# ===========================================================================

def test_baltic_validates_cleanly():
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    report = Validator().validate(inst)
    # Baltic should have no ERROR findings (may have INFO about missing coords).
    errors = report.errors()
    for e in errors:
        print(f"  ERROR: {e.message}")
    assert not any(e.code.startswith(("DIST_UNKNOWN", "DEM_UNKNOWN", "STRUCT_PORT_DUP"))
                   for e in errors), f"Unexpected structural errors in Baltic: {errors}"


def test_worldlarge_duplicate_od_warning():
    """WorldLarge has 7 duplicate OD pairs; validator should flag them as WARNING."""
    loader = _loader()
    inst = loader.load("WorldLarge", validate=False)
    report = Validator().validate(inst)
    dup_warnings = [f for f in report.warnings() if f.code == "DEM_DUP_OD_PAIR"]
    assert len(dup_warnings) == 7, f"Expected 7 duplicate-OD warnings, got {len(dup_warnings)}"


def test_waf_instance_loaded_correctly():
    """WAF has 20 ports and 37 demands (per readme v1.2)."""
    loader = _loader()
    inst = loader.load("WAF", validate=False)
    assert inst.metadata.active_port_count == 20
    assert inst.metadata.demand_count == 37


def test_transittime_revision_applied():
    """If revision exists, transit times are overridden."""
    loader = _loader()
    inst = loader.load("WorldSmall", validate=False)
    # The fixed demand file + transittime revision should produce known TT values.
    # Row 35 (SAJED->AEJEA) originally had TT=6, revised to TT=11.
    tt_for_sajed_aejea = [
        d.max_transit_time
        for d in inst.demands
        if d.origin == "SAJED" and d.destination == "AEJEA"
    ]
    assert 11 in tt_for_sajed_aejea, "Transittime revision not applied for SAJED->AEJEA"


def test_transittime_revision_not_applied_for_baltic():
    """Baltic has no transittime_revision file; original TT is preserved."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    firau_debrv_tts = [
        d.max_transit_time
        for d in inst.demands
        if d.origin == "FIRAU" and d.destination == "DEBRV"
    ]
    assert 16 in firau_debrv_tts


def test_dist_dense_subset_size():
    """Distance count scales with instance size."""
    loader = _loader()
    inst_b = loader.load("Baltic", validate=False)
    inst_ws = loader.load("WorldSmall", validate=False)
    # WorldSmall has more ports → more dense distance arcs.
    assert len(inst_ws.distances) > len(inst_b.distances)


def test_worldsmall_fixed_default():
    """WorldSmall loads the Fixed_Sep variant by default (FFEPerWeek as int)."""
    loader = _loader()
    inst = loader.load("WorldSmall", validate=False)
    # Check that no FFE value is suspiciously small (< 100 but known to be large).
    small_ffes = [d.ffe_per_week for d in inst.demands if 0 < d.ffe_per_week < 100]
    # A few legitimately small demands exist; the corrupted ones (1.86, 1.605...)
    # should all be >= 100 now.
    assert all(d.ffe_per_week >= 100 or d.ffe_per_week == 1.0
               for d in inst.demands
               if d.origin == "CNSHA" and d.destination == "DEBRV"), \
        "WorldSmall original corruption still present"


def test_worldsmall_original_available():
    """Original WorldSmall demand file can still be loaded via parameter.

    The original has 7 rows with decimal FFE (e.g. 1.86 instead of 1860).
    We identify them by checking that specific known-corrupted pairs have
    FFE well below what the fixed version carries.
    """
    loader = _loader()
    orig = loader.load("WorldSmall", validate=False, use_fixed_worldsmall=False)
    fixed = loader.load("WorldSmall", validate=False, use_fixed_worldsmall=True)

    # Compare the 7 known-corrupted OD pairs between the two versions.
    orig_map = {(d.origin, d.destination): d.ffe_per_week for d in orig.demands}
    fixed_map = {(d.origin, d.destination): d.ffe_per_week for d in fixed.demands}

    corrupted_pairs = [
        ("CNSHA", "DEBRV"),
        ("KRPUS", "DEBRV"),
        ("MYTPP", "DEBRV"),
        ("CNSHA", "NLRTM"),
        ("CNYTN", "NLRTM"),
        ("CNSHA", "USLAX"),
        ("CNYTN", "USLAX"),
    ]
    diffs = 0
    for pair in corrupted_pairs:
        o_val = orig_map.get(pair, float("nan"))
        f_val = fixed_map.get(pair, float("nan"))
        if o_val != f_val and o_val < f_val:
            diffs += 1
    assert diffs == 7, f"Expected 7 corrupted pairs in original vs fixed, got {diffs}"


# ===========================================================================
# Synthetic fixture tests (isolated error conditions)
# ===========================================================================

def _make_synthetic_ports():
    from data.instance import Port, ProvenanceRecord
    return {
        "AAA": Port(
            unlocode="AAA", name="Alpha", country="X", cabotage_region="X",
            d_region=None, longitude=0.0, latitude=0.0, draft=None,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=1000.0, port_call_cost_per_ffe=5.0,
            provenance=ProvenanceRecord(source_file="ports.csv", source_row=1),
        ),
        "BBB": Port(
            unlocode="BBB", name="Bravo", country="Y", cabotage_region="Y",
            d_region=None, longitude=0.0, latitude=0.0, draft=10.0,
            cost_per_full=200.0, cost_per_full_transfer=20.0,
            port_call_cost_fixed=2000.0, port_call_cost_per_ffe=6.0,
            provenance=ProvenanceRecord(source_file="ports.csv", source_row=2),
        ),
    }


def _make_synthetic_vessels():
    from data.instance import VesselType, ProvenanceRecord
    return {
        "Tiny_100": VesselType(
            vessel_class="Tiny_100", capacity_ffe=100, tc_rate_daily=1000,
            draft=5.0, min_speed=8.0, max_speed=12.0, design_speed=10.0,
            bunker_ton_per_day_at_design=5.0, idle_consumption_ton_per_day=1.0,
            panama_fee=None, suez_fee=None,
            provenance=ProvenanceRecord(source_file="fleet_data.csv", source_row=1),
        ),
    }


def _make_bad_instance():
    from data.instance import (
        LINERLIBInstance, FleetEntry, DistanceArc, Demand, InstanceMetadata,
        DatasetProvenance, ProvenanceRecord,
    )
    ports = _make_synthetic_ports()
    vessels = _make_synthetic_vessels()
    return LINERLIBInstance(
        name="SYNTHETIC_BAD",
        ports=ports,
        vessel_types=vessels,
        fleet=[FleetEntry(vessel_class="Tiny_100", quantity=1)],
        distances=[
            DistanceArc(
                origin="AAA", destination="XXX_NOT_REAL", distance_nm=100.0,
                draft_required=None, is_panama=False, is_suez=False,
                provenance=ProvenanceRecord(source_file="dist_dense.csv", source_row=1),
            ),
        ],
        demands=[
            Demand(
                origin="AAA", destination="ZZZ_NOWHERE",
                ffe_per_week=-5.0, revenue=100.0, max_transit_time=10,
                provenance=ProvenanceRecord(source_file="Demand_test.csv", source_row=1),
            ),
            Demand(
                origin="AAA", destination="AAA",  # self-loop
                ffe_per_week=10.0, revenue=50.0, max_transit_time=-1,
                provenance=ProvenanceRecord(source_file="Demand_test.csv", source_row=2),
            ),
        ],
        metadata=InstanceMetadata(
            name="SYNTHETIC_BAD", active_port_count=2,
            vessel_type_count=1, total_vessels=1,
            demand_count=2, distance_arc_count=1,
        ),
        provenance=DatasetProvenance(source_root=str(DATA_ROOT)),
    )


def test_validation_catches_unknown_demand_dest():
    inst = _make_bad_instance()
    report = Validator().validate(inst, global_ports=_make_synthetic_ports())
    unknown_dests = [f for f in report.findings if f.code == "DEM_UNKNOWN_DEST"]
    assert len(unknown_dests) == 1


def test_validation_catches_negative_ffe():
    inst = _make_bad_instance()
    report = Validator().validate(inst, global_ports=_make_synthetic_ports())
    neg_ffes = [f for f in report.findings if f.code == "DEM_ZERO_OR_NEG_FFE"]
    assert len(neg_ffes) == 1


def test_validation_catches_invalid_transit_time():
    inst = _make_bad_instance()
    report = Validator().validate(inst, global_ports=_make_synthetic_ports())
    bad_tt = [f for f in report.findings if f.code == "DEM_BAD_TRANSIT_TIME"]
    assert len(bad_tt) == 1


def test_validation_strict_mode_raises():
    loader = LINERLIBLoader(root=str(DATA_ROOT), strict_validation=True)
    # Construct a deliberately broken instance using real ports but bad demand
    from data.instance import Demand, ProvenanceRecord, LINERLIBInstance, InstanceMetadata, DatasetProvenance
    inst = loader.load("Baltic", validate=False)
    # Inject a bad demand
    inst.demands.append(Demand(
        origin="NONEXIST", destination="DEBRV",
        ffe_per_week=10.0, revenue=100.0, max_transit_time=5,
        provenance=ProvenanceRecord(source_file="test", source_row=999),
    ))
    try:
        # Reload with strict validation to trigger raise
        inst2 = loader.load("Baltic", validate=True)
        inst2.demands.append(Demand(
            origin="NONEXIST", destination="DEBRV",
            ffe_per_week=10.0, revenue=100.0, max_transit_time=5,
            provenance=ProvenanceRecord(source_file="test", source_row=999),
        ))
        from data.validation import Validator
        report = Validator(fail_fast=True).validate(inst2)
        report.raise_errors()
        assert False, "Should have raised"
    except ValueError as exc:
        assert "Validation failed" in str(exc)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
