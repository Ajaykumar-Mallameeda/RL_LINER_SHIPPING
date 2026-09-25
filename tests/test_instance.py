"""
P1 – Tests for the canonical instance model.

Covers:
  - Instance equality
  - active_ports() method
  - Metadata consistency
  - Immutability guarantees (deep copy on transform)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.linerlib_loader import LINERLIBLoader


DATA_ROOT = ROOT / "data"


def _loader():
    return LINERLIBLoader(root=str(DATA_ROOT))


# ===========================================================================
# Instance equality
# ===========================================================================

def test_instance_equality_symmetric():
    loader = _loader()
    a = loader.load("Baltic", validate=False)
    b = loader.load("Baltic", validate=False)
    assert a == b
    assert b == a


def test_different_instances_not_equal():
    loader = _loader()
    baltic = loader.load("Baltic", validate=False)
    waf = loader.load("WAF", validate=False)
    assert baltic != waf


def test_equality_is_reflexive():
    loader = _loader()
    inst = loader.load("Pacific", validate=False)
    assert inst == inst


# ===========================================================================
# active_ports()
# ===========================================================================

def test_active_ports_returns_set():
    loader = _loader()
    inst = loader.load("WAF", validate=False)
    ports = inst.active_ports()
    assert isinstance(ports, set)
    assert len(ports) == 20
    assert "DJJIB" in ports  # Djibouti (DJJIB) is in WAF


def test_active_ports_matches_dict_keys():
    loader = _loader()
    inst = loader.load("Mediterranean", validate=False)
    assert inst.active_ports() == set(inst.ports.keys())


# ===========================================================================
# Metadata consistency
# ===========================================================================

def test_metadata_port_count_matches():
    loader = _loader()
    for name in loader.available_instances():
        inst = loader.load(name, validate=False)
        assert inst.metadata.active_port_count == len(inst.ports), \
            f"{name}: metadata.port_count {inst.metadata.active_port_count} != len(ports) {len(inst.ports)}"


def test_metadata_demand_count_matches():
    loader = _loader()
    for name in loader.available_instances():
        inst = loader.load(name, validate=False)
        assert inst.metadata.demand_count == len(inst.demands), \
            f"{name}: metadata.demand_count mismatch"


def test_metadata_total_vessels_matches_fleet():
    loader = _loader()
    for name in loader.available_instances():
        inst = loader.load(name, validate=False)
        expected = sum(e.quantity for e in inst.fleet)
        assert inst.metadata.total_vessels == expected, \
            f"{name}: metadata.total_vessels {inst.metadata.total_vessels} != fleet sum {expected}"


# ===========================================================================
# Known instance sizes
# ===========================================================================

EXPECTED_SIZES = {
    "Baltic":       {"ports": 12, "demands": 22, "vessels": 6},
    "WAF":          {"ports": 20, "demands": 37, "vessels": 42},
    "Mediterranean":{"ports": 39, "demands": 365, "vessels": 20},
    "Pacific":      {"ports": 45, "demands": 722, "vessels": 100},
    "WorldSmall":   {"ports": 47, "demands": 1764, "vessels": 263},
    "WorldLarge":   {"ports": 201, "demands": 9622, "vessels": 501},
    "EuropeAsia":   {"ports": 114, "demands": 4000, "vessels": 176},
}


def test_known_instance_sizes():
    loader = _loader()
    for name, expected in EXPECTED_SIZES.items():
        inst = loader.load(name, validate=False)
        assert inst.metadata.active_port_count == expected["ports"], \
            f"{name}: expected {expected['ports']} ports, got {inst.metadata.active_port_count}"
        assert inst.metadata.demand_count == expected["demands"], \
            f"{name}: expected {expected['demands']} demands, got {inst.metadata.demand_count}"
        assert inst.metadata.total_vessels == expected["vessels"], \
            f"{name}: expected {expected['vessels']} vessels, got {inst.metadata.total_vessels}"


# ===========================================================================
# Deep-copy semantics (no mutation of originals)
# ===========================================================================

def test_transform_does_not_mutate_source():
    from data.normalization import Normalizer
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    original_demands = list(inst.demands)

    norm = Normalizer()
    norm.fit([inst])
    normed = norm.transform(inst)

    # Original must be untouched
    for od, nd in zip(inst.demands, original_demands):
        assert od.ffe_per_week == nd.ffe_per_week
        assert od.revenue == nd.revenue

    # Normalized values should differ from raw (Baltic FFEs range 6–1215,
    # so the min_max transform produces non-trivial outputs in (0, 1)).
    assert normed.instance.demands[0].ffe_per_week != inst.demands[0].ffe_per_week
    # And every normalized FFE should be finite and in a reasonable range.
    for d in normed.instance.demands:
        assert float("-inf") < d.ffe_per_week < float("inf")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
