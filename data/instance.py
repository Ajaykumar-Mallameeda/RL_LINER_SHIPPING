"""
P1 – LINERLIB benchmark data foundation.

Canonical internal representation for the LINERLIB master-port catalogue,
vessel-types, distances, instance-specific fleets and demands.

DO NOT import anything from P2+ (RL, Gymnasium, torch, …) here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRecord:
    """Traceability metadata attached to every loaded record."""

    source_file: str
    source_row: int  # 1-indexed
    dataset_version: str = "1.2"
    instance: Optional[str] = None


@dataclass
class DatasetProvenance:
    """High-level provenance for an entire instance."""

    source_root: str
    schema_version: str = "1.0"
    files: Dict[str, str] = field(default_factory=dict)  # name -> sha256 hex
    raw_data_unmodified: bool = True
    source_version_note: str = "SOURCE VERSION NOT AVAILABLE"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

@dataclass
class Port:
    unlocode: str
    name: str
    country: Optional[str]
    cabotage_region: str
    d_region: Optional[str]
    longitude: Optional[float]
    latitude: Optional[float]
    draft: Optional[float]
    cost_per_full: Optional[float]
    cost_per_full_transfer: Optional[float]
    port_call_cost_fixed: float
    port_call_cost_per_ffe: float
    provenance: ProvenanceRecord


@dataclass
class VesselType:
    vessel_class: str
    capacity_ffe: int
    tc_rate_daily: int
    draft: float
    min_speed: float
    max_speed: float
    design_speed: float
    bunker_ton_per_day_at_design: float
    idle_consumption_ton_per_day: float
    panama_fee: Optional[int]
    suez_fee: Optional[int]
    provenance: ProvenanceRecord


@dataclass
class FleetEntry:
    vessel_class: str
    quantity: int


@dataclass
class DistanceArc:
    origin: str
    destination: str
    distance_nm: float
    draft_required: Optional[float]
    is_panama: bool
    is_suez: bool
    provenance: ProvenanceRecord


@dataclass
class Demand:
    origin: str
    destination: str
    ffe_per_week: float
    revenue: float
    max_transit_time: int
    provenance: ProvenanceRecord


@dataclass
class InstanceMetadata:
    """Lightweight summary of what was loaded for this instance."""

    name: str
    active_port_count: int
    vessel_type_count: int
    total_vessels: int
    demand_count: int
    distance_arc_count: int
    sparse_arc_count: int = 0


# ---------------------------------------------------------------------------
# Canonical instance
# ---------------------------------------------------------------------------

@dataclass
class LINERLIBInstance:
    """
    The canonical, validated, deterministic representation of a single
    LINERLIB benchmark instance.
    """

    name: str
    schema_version: str = "1.0"

    # Global registry of all vessel types (shared across instances).
    vessel_types: Dict[str, VesselType] = field(default_factory=dict)

    # Active ports for THIS instance, keyed by UNLOCODE.
    ports: Dict[str, Port] = field(default_factory=dict)

    # Per-instance fleet assignment (subset of vessel_types).
    fleet: List[FleetEntry] = field(default_factory=list)

    # All-pairs distances restricted to active ports (directed arcs).
    distances: List[DistanceArc] = field(default_factory=list)

    # Sparse adjacency list (subset used by this instance).
    sparse_distances: List[DistanceArc] = field(default_factory=list)

    # Instance-specific demand records.
    demands: List[Demand] = field(default_factory=list)

    # Metadata summary.
    metadata: Optional[InstanceMetadata] = None

    # Full provenance.
    provenance: Optional[DatasetProvenance] = None

    # Arbitrary user-defined tags (for later phases).
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = InstanceMetadata(
                name=self.name,
                active_port_count=len(self.ports),
                vessel_type_count=len(self.vessel_types),
                total_vessels=sum(e.quantity for e in self.fleet),
                demand_count=len(self.demands),
                distance_arc_count=len(self.distances),
                sparse_arc_count=len(self.sparse_distances),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LINERLIBInstance):
            return NotImplemented
        return (
            self.name == other.name
            and self.ports == other.ports
            and self.vessel_types == other.vessel_types
            and self.fleet == other.fleet
            and self.distances == other.distances
            and self.sparse_distances == other.sparse_distances
            and self.demands == other.demands
        )

    def active_ports(self) -> set:
        """Return the set of UNLOCODEs active in this instance."""
        return set(self.ports.keys())
