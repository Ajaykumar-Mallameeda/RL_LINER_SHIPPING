"""
P1 – Deterministic, read-only normalisation utilities.

Normalization is applied **after** loading and **never** mutates the original
LINERLIBInstance.  A separate fitted state object stores the statistics so that
train/validation/test splits can share a single Normalizer without leaking
test-set information into the fit statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FieldStats:
    method: str  # "min_max" | "log" | "none"
    # min_max parameters
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    # log parameters
    base: float = 1.0  # ln when base=1.0 (natural log); log10 when base=10.0
    offset: float = 0.0  # added before transform (e.g. +1 for log(x+1))

    def fit_from_values(self, values: List[float]) -> None:
        if self.method == "min_max" and values:
            self.min_val = min(values)
            self.max_val = max(values)
        elif self.method == "log" and values:
            # For log, min/max aren't needed at transform time but we record them
            # for reproducibility.
            pass

    def transform(self, x: float) -> float:
        if self.method == "none":
            return x
        if self.method == "min_max":
            if self.max_val is None or self.max_val == self.min_val:
                return 0.0
            return (x - self.min_val) / (self.max_val - self.min_val)
        if self.method == "log":
            import math
            return math.log(x + self.offset)
        return x

    def to_dict(self) -> dict:
        d = {"method": self.method}
        if self.min_val is not None:
            d["min"] = self.min_val
        if self.max_val is not None:
            d["max"] = self.max_val
        d["base"] = self.base
        d["offset"] = self.offset
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FieldStats":
        return cls(
            method=d["method"],
            min_val=d.get("min"),
            max_val=d.get("max"),
            base=d.get("base", 1.0),
            offset=d.get("offset", 0.0),
        )


# Default per-field-category stats.
DEFAULT_STATS: Dict[str, FieldStats] = {
    "ffe_per_week": FieldStats(method="min_max", offset=0.0),
    "revenue":      FieldStats(method="min_max", offset=0.0),
    "distance_nm":  FieldStats(method="log",    offset=1.0),
    "vessel_capacity": FieldStats(method="log", offset=1.0),
    "vessel_tc_rate":  FieldStats(method="log", offset=1.0),
    "cost_per_full":   FieldStats(method="min_max", offset=0.0),
}


@dataclass
class NormalizerState:
    """Serializable statistics produced by fit()."""
    fields: Dict[str, FieldStats] = field(default_factory=dict)
    fit_source_instances: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "fit_source_instances": list(self.fit_source_instances),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizerState":
        return cls(
            fields={k: FieldStats.from_dict(v) for k, v in d.get("fields", {}).items()},
            fit_source_instances=d.get("fit_source_instances", []),
        )


class Normalizer:
    """
    Fits normalisation statistics on a collection of training instances and
    applies deterministic transforms to any instance.

    Example
    -------
    >>> norm = Normalizer()
    >>> norm.fit([inst_baltic, inst_waf])          # trains on these only
    >>> normed_ws = norm.transform(inst_worldsmall) # test instance, no leak
    """

    def __init__(
        self,
        field_stats: Optional[Dict[str, FieldStats]] = None,
    ) -> None:
        # Deep-copy default stats so that subsequent fit()/transform() calls
        # never mutate the module-level DEFAULT_STATS object.
        src = field_stats if field_stats is not None else DEFAULT_STATS
        self._stats: Dict[str, FieldStats] = {
            k: FieldStats.from_dict(v.to_dict()) for k, v in src.items()
        }
        self._state = NormalizerState(
            fields={k: FieldStats.from_dict(v.to_dict()) for k, v in self._stats.items()},
        )

    # ---- fit ----

    def fit(self, instances) -> "Normalizer":
        """
        Compute statistics from *instances*.

        Calling ``fit`` again overwrites previous state.  Callers are expected
        to pass ONLY training-set instances so that validation/test leakage
        cannot occur through accidental refitting.
        """
        # Gather per-field value lists from each instance.
        agg: Dict[str, List[float]] = {k: [] for k in self._stats}

        for inst in instances:
            for dem in inst.demands:
                agg["ffe_per_week"].append(dem.ffe_per_week)
                agg["revenue"].append(dem.revenue)
            for arc in inst.distances:
                agg["distance_nm"].append(arc.distance_nm)
            for _, vt in inst.vessel_types.items():
                agg["vessel_capacity"].append(float(vt.capacity_ffe))
                agg["vessel_tc_rate"].append(float(vt.tc_rate_daily))
            for _, port in inst.ports.items():
                agg["cost_per_full"].append(port.cost_per_full)

        for name, vals in agg.items():
            if name in self._stats:
                self._stats[name].fit_from_values(vals)

        self._state = NormalizerState(
            fields={k: FieldStats.from_dict(v.to_dict()) for k, v in self._stats.items()},
            fit_source_instances=[inst.name for inst in instances],
        )
        return self

    def fit_from_state(self, state: NormalizerState) -> "Normalizer":
        """Restore state from a previously-serialized NormalizerState."""
        # Values may be either FieldStats objects (from in-memory round-trip)
        # or plain dicts (from JSON deserialization).
        self._stats = {
            k: v if isinstance(v, FieldStats) else FieldStats.from_dict(v)
            for k, v in state.fields.items()
        }
        self._state = state
        return self

    # ---- transform ----

    def transform(self, instance) -> "NormalizedInstance":
        """
        Return a NEW NormalizedInstance with numeric fields scaled.
        The original *instance* is never mutated.
        """
        import copy
        new_instance = copy.deepcopy(instance)

        # Demands
        for dem in new_instance.demands:
            dem.ffe_per_week = self._stats["ffe_per_week"].transform(dem.ffe_per_week)
            dem.revenue = self._stats["revenue"].transform(dem.revenue)

        # Distances
        for arc in new_instance.distances:
            arc.distance_nm = self._stats["distance_nm"].transform(arc.distance_nm)

        # Vessel types
        for _, vt in new_instance.vessel_types.items():
            vt.capacity_ffe = self._stats["vessel_capacity"].transform(float(vt.capacity_ffe))
            vt.tc_rate_daily = self._stats["vessel_tc_rate"].transform(float(vt.tc_rate_daily))

        # Ports
        for _, port in new_instance.ports.items():
            port.cost_per_full = self._stats["cost_per_full"].transform(port.cost_per_full)

        # Copy tags so caller can track provenance
        new_instance.tags["normalized_by"] = self._state.fit_source_instances

        return NormalizedInstance(
            instance=new_instance,
            state=self._state,
        )

    def get_state(self) -> NormalizerState:
        return self._state


@dataclass
class NormalizedInstance:
    """
    Thin wrapper around a LINERLIBInstance that also exposes the fitted state
    so callers know which instances were used for fitting.
    """
    instance: object  # LINERLIBInstance (deep-copied, never the original)
    state: NormalizerState
