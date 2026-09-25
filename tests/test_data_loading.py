"""
P1 – Tests for data loading (LINERLIBLoader).

Covers:
  1. File discovery / available instances
  2. Parsing correctness per file type
  3. Schema / column presence
  4. Deterministic loading (same source → same instance)
  5. Instance selection (all verified instances)
  6. No raw-file mutation
"""

import csv
import os
import sys
from pathlib import Path

# Add parent to path so ``data`` package is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.linerlib_loader import LINERLIBLoader
from data.schema import INSTANCE_DEFS


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = ROOT / "data"


def _loader() -> LINERLIBLoader:
    return LINERLIBLoader(root=str(DATA_ROOT))


# ===========================================================================
# Test 1 — File discovery
# ===========================================================================

def test_available_instances():
    loader = _loader()
    names = loader.available_instances()
    assert names == sorted(INSTANCE_DEFS.keys()), f"Expected {sorted(INSTANCE_DEFS.keys())}, got {names}"
    assert len(names) == 7


def test_all_instance_files_exist():
    """Every instance definition must reference an existing file."""
    for name, defs in INSTANCE_DEFS.items():
        df = DATA_ROOT / defs["demand_file"]
        ff = DATA_ROOT / defs["fleet_file"]
        assert df.exists(), f"{name}: missing demand file {df}"
        assert ff.exists(), f"{name}: missing fleet file {ff}"


# ===========================================================================
# Test 2 — Parsing
# ===========================================================================

def test_port_parsing():
    """ports.csv parses without error and yields expected columns."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    assert len(inst.ports) == 12
    assert "DEBRV" in inst.ports
    p = inst.ports["DEBRV"]
    assert p.name == "Bremerhaven"
    assert p.cabotage_region == "Germany"
    assert p.cost_per_full == 199.0


def test_fleet_parsing():
    """fleet_Baltic.csv parses correctly."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    assert len(inst.fleet) == 2
    classes = {e.vessel_class for e in inst.fleet}
    assert classes == {"Feeder_450", "Feeder_800"}
    qty_map = {e.vessel_class: e.quantity for e in inst.fleet}
    assert qty_map["Feeder_450"] == 4
    assert qty_map["Feeder_800"] == 2


def test_demand_parsing():
    """Demand rows parse with correct numeric types."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    assert len(inst.demands) == 22
    d = inst.demands[0]
    assert d.origin == "FIRAU"
    assert d.destination == "DEBRV"
    assert d.ffe_per_week == 77.0
    assert d.revenue == 1120.0
    assert d.max_transit_time == 16


def test_distance_parsing():
    """dist_dense subset for Baltic has correct structure."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    baltic_ports = set(inst.ports.keys())
    assert len(inst.distances) > 0
    for arc in inst.distances:
        assert arc.origin in baltic_ports
        assert arc.destination in baltic_ports
        assert arc.distance_nm >= 0


def test_sparse_distance_parsing():
    """dist_sparse subset loads without header confusion."""
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    assert len(inst.sparse_distances) >= 0  # may be small for Baltic
    for arc in inst.sparse_distances:
        assert arc.origin in inst.ports
        assert arc.destination in inst.ports


# ===========================================================================
# Test 3 — Schema
# ===========================================================================

def test_schema_columns_present():
    """All required schema columns are present in loaded objects."""
    loader = _loader()
    inst = loader.load("WorldSmall", validate=False)
    # Ports
    for p in inst.ports.values():
        assert hasattr(p, "unlocode")
        assert hasattr(p, "name")
        assert hasattr(p, "latitude")
        assert hasattr(p, "cost_per_full")
    # Vessels
    for vt in inst.vessel_types.values():
        assert hasattr(vt, "capacity_ffe")
        assert hasattr(vt, "design_speed")
    # Demands
    for dem in inst.demands:
        assert hasattr(dem, "ffe_per_week")
        assert hasattr(dem, "revenue")
        assert hasattr(dem, "max_transit_time")
    # Distances
    for arc in inst.distances:
        assert hasattr(arc, "distance_nm")
        assert hasattr(arc, "is_panama")


# ===========================================================================
# Test 4 — Deterministic loading
# ===========================================================================

def test_deterministic_loading():
    """Loading the same instance twice yields equivalent instances."""
    loader = _loader()
    a = loader.load("Mediterranean", validate=False)
    b = loader.load("Mediterranean", validate=False)
    assert a == b
    assert a.metadata.active_port_count == b.metadata.active_port_count
    assert a.metadata.demand_count == b.metadata.demand_count


# ===========================================================================
# Test 5 — Instance selection
# ===========================================================================

def test_load_all_instances():
    """Every verified instance loads without error."""
    loader = _loader()
    for name in loader.available_instances():
        inst = loader.load(name, validate=False)
        assert inst.name == name
        assert inst.metadata is not None
        assert inst.metadata.active_port_count > 0
        assert inst.metadata.demand_count > 0
        assert len(inst.fleet) > 0


# ===========================================================================
# Test 6 — No raw mutation
# ===========================================================================

def test_no_raw_mutation():
    """Loading must not alter raw source files on disk."""
    loader = _loader()
    hashes_before = {}
    for p in sorted(DATA_ROOT.rglob("*")):
        if p.is_file():
            hashes_before[str(p.relative_to(DATA_ROOT))] = p.stat().st_size

    loader.load("Pacific", validate=False)
    loader.load("WorldLarge", validate=False)

    for p in sorted(DATA_ROOT.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(DATA_ROOT))
            assert hashes_before.get(rel) == p.stat().st_size, (
                f"Raw file mutated: {rel}"
            )


# ===========================================================================
# Test 7 — Deterministic ordering
# ===========================================================================

def test_deterministic_ordering():
    """Ports and demands should have stable ordering across loads."""
    loader = _loader()
    a = loader.load("WAF", validate=False)
    b = loader.load("WAF", validate=False)
    port_keys_a = list(a.ports.keys())
    port_keys_b = list(b.ports.keys())
    assert port_keys_a == port_keys_b, "Port ordering is non-deterministic"
    dem_orig_a = [d.origin for d in a.demands]
    dem_orig_b = [d.origin for d in b.demands]
    assert dem_orig_a == dem_orig_b, "Demand ordering is non-deterministic"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
