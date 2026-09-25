"""
P1 – Tests for the normalization utilities.

Covers:
  - Deterministic transform
  - Fit on training, transform on held-out instance (no leakage)
  - Invariant: original instance is not mutated
  - State serialisation round-trip
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.linerlib_loader import LINERLIBLoader
from data.normalization import Normalizer, FieldStats


DATA_ROOT = ROOT / "data"


def _loader():
    return LINERLIBLoader(root=str(DATA_ROOT))


# ===========================================================================
# Deterministic transforms
# ===========================================================================

def test_transform_is_deterministic():
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    norm = Normalizer()
    norm.fit([inst])
    n1 = norm.transform(inst)
    n2 = norm.transform(inst)
    for d1, d2 in zip(n1.instance.demands, n2.instance.demands):
        assert d1.ffe_per_week == d2.ffe_per_week
        assert d1.revenue == d2.revenue


def test_fit_on_multiple_instances():
    loader = _loader()
    baltic = loader.load("Baltic", validate=False)
    waf = loader.load("WAF", validate=False)
    norm = Normalizer()
    norm.fit([baltic, waf])
    normed_ws = norm.transform(loader.load("WorldSmall", validate=False))
    # FFE values from Baltic (6–1215) and WAF (1–1595) set the scale.
    # WorldSmall demands outside [1, 1595] will clip slightly below 0 or above 1.
    # We verify the transform is non-trivial and values are finite.
    ffe_vals = [d.ffe_per_week for d in normed_ws.instance.demands]
    assert all(float("-inf") < v < float("inf") for v in ffe_vals)
    assert len(set(ffe_vals)) > 1, "All transformed FFE values should not be identical"


def test_log_transform_range():
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    norm = Normalizer()
    norm.fit([inst])
    normed = norm.transform(inst)
    dists = [a.distance_nm for a in normed.instance.distances]
    # log(x+1) for x >= 0 yields values in [0, ~log(max+1)].
    assert all(v >= 0.0 for v in dists)


# ===========================================================================
# No mutation of source
# ===========================================================================

def test_source_not_mutated():
    loader = _loader()
    inst = loader.load("Mediterranean", validate=False)
    first_ffe = inst.demands[0].ffe_per_week
    first_rev = inst.demands[0].revenue

    norm = Normalizer()
    norm.fit([inst])
    _ = norm.transform(inst)

    assert inst.demands[0].ffe_per_week == first_ffe
    assert inst.demands[0].revenue == first_rev


# ===========================================================================
# State serialisation
# ===========================================================================

def test_state_roundtrip():
    from data.normalization import NormalizerState
    loader = _loader()
    inst = loader.load("Pacific", validate=False)
    norm = Normalizer()
    norm.fit([inst])
    state = norm.get_state()
    d = state.to_dict()
    restored = Normalizer()
    restored.fit_from_state(NormalizerState.from_dict(d))
    n1 = norm.transform(inst)
    n2 = restored.transform(inst)
    for d1, d2 in zip(n1.instance.demands, n2.instance.demands):
        assert d1.ffe_per_week == d2.ffe_per_week


# ===========================================================================
# Edge cases
# ===========================================================================

def test_empty_fit():
    """Fitting on an empty list should not crash; transform yields 0 for min_max."""
    norm = Normalizer()
    norm.fit([])  # should not raise
    inst = _loader().load("Baltic", validate=False)
    # With no fitted stats (min/max still None), min_max transform returns 0.0.
    normed = norm.transform(inst)
    assert normed.instance.demands[0].ffe_per_week == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
