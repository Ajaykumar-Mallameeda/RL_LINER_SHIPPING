"""
P1 – LINERLIB benchmark loader.

Deterministically loads a verified LINERLIB instance from raw CSV/JSON files,
builds the canonical LINERLIBInstance representation, and computes SHA-256
provenance hashes for every source file read.

Raw data is NEVER modified.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .instance import (
    Demand,
    DistanceArc,
    FleetEntry,
    InstanceMetadata,
    LINERLIBInstance,
    Port,
    ProvenanceRecord,
    DatasetProvenance,
    VesselType,
)
from .schema import (
    INSTANCE_DEFS,
    WORLDSMALL_FIXED_DEFAULT,
    TRANSTIME_APPLIED_TO,
    TRANSTIME_REVISION_DIR,
)
from .provenance import sha256_of


# ---------------------------------------------------------------------------
# Internal loader helpers
# ---------------------------------------------------------------------------

def _read_tsv(path: Path, has_header: bool = True):
    """Yield rows as dicts from a tab-separated file."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t") if has_header else csv.reader(f, delimiter="\t")
        for row in reader:
            yield row


def _safe_float(val: str, default: Optional[float] = None) -> Optional[float]:
    """Parse float, returning default on empty, NaN-string, or bad input."""
    s = val.strip()
    if s == "" or s.upper() in ("NULL", "N/A", "NA", ""):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _safe_int(val: str, default: Optional[int] = None) -> Optional[int]:
    s = val.strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return default


# ---------------------------------------------------------------------------
# LINERLIBLoader
# ---------------------------------------------------------------------------

class LINERLIBLoader:
    """
    Load and construct canonical LINERLIB benchmark instances.

    Parameters
    ----------
    root :
        Path to the LINERLIB ``data/`` directory (the one containing ports.csv,
        demand files, fleet files, dist_dense.csv, etc.).
    apply_transittime_revision :
        If True (default), replace TransitTime values with those from the
        transittime_revision/ directory when one exists for the requested
        instance.
    strict_validation :
        If True, load() raises on ERROR-level validation findings.
    """

    def __init__(
        self,
        root: str,
        apply_transittime_revision: bool = True,
        strict_validation: bool = False,
    ) -> None:
        self._root = Path(root).resolve()
        self._apply_tt_revision = apply_transittime_revision
        self._strict = strict_validation
        self._file_hashes: Dict[str, str] = {}
        self._global_ports: Optional[Dict[str, Port]] = None
        self._global_vessel_types: Optional[Dict[str, VesselType]] = None

    # ---- public API ----

    @property
    def root(self) -> Path:
        return self._root

    def available_instances(self) -> List[str]:
        """Return instance names sorted alphabetically."""
        return sorted(INSTANCE_DEFS.keys())

    def load(
        self,
        name: str,
        validate: bool = True,
        use_fixed_worldsmall: Optional[bool] = None,
    ):
        """
        Load and return a validated LINERLIBInstance for *name*.

        Parameters
        ----------
        name :
            One of the verified instance names (Baltic, WAF, Mediterranean,
            Pacific, WorldSmall, WorldLarge, EuropeAsia).
        validate :
            Run Validator after loading; raise on ERROR if ``strict_validation``
            is also enabled at construction time.
        use_fixed_worldsmall :
            Override the default fixed/orig choice for WorldSmall.
        """
        if name not in INSTANCE_DEFS:
            raise ValueError(
                f"Unknown instance '{name}'. Available: {self.available_instances()}"
            )

        # Ensure global shared files are loaded once.
        self._ensure_globals()

        defs = INSTANCE_DEFS[name]
        demand_file = defs["demand_file"]
        fleet_file = defs["fleet_file"]

        # WorldSmall: honour user override or default to fixed variant.
        if name == "WorldSmall" and use_fixed_worldsmall is False:
            demand_file = "Demand_WorldSmall.csv"

        # Compute file hashes before reading.
        demand_path = self._root / demand_file
        fleet_path = self._root / fleet_file
        self._file_hashes[demand_file] = sha256_of(demand_path)
        self._file_hashes[fleet_file] = sha256_of(fleet_path)

        # --- Build active port set from demand ---
        active_ports: Set[str] = set()
        raw_demands = []  # list of (row_dict, row_number)
        with demand_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                o = row["Origin"].strip()
                d = row["Destination"].strip()
                active_ports.add(o)
                active_ports.add(d)
                raw_demands.append((row, i))

        # --- Assemble Port objects (only for active ports) ---
        ports: Dict[str, Port] = {}
        for code in sorted(active_ports):
            gp = self._global_ports.get(code)
            if gp is None:
                # Should not happen for well-formed data, but guard anyway.
                continue
            ports[code] = gp

        # --- Vessel types (global, shared across all instances) ---
        vessel_types = dict(self._global_vessel_types)

        # --- Fleet entries (per-instance) ---
        fleet: List[FleetEntry] = []
        with fleet_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                vc = row["Vessel class"].strip()
                qty = int(row["Quantity"].strip())
                fleet.append(FleetEntry(vessel_class=vc, quantity=qty))

        # --- Distances (dense, restricted to active ports) ---
        distances: List[DistanceArc] = []
        dist_path = self._root / "dist_dense.csv"
        self._file_hashes["dist_dense.csv"] = sha256_of(dist_path)
        with dist_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                o = row["fromUNLOCODe"].strip()
                d = row["ToUNLOCODE"].strip()
                if o in active_ports and d in active_ports:
                    dist = _safe_float(row["Distance"])
                    draft = _safe_float(row["Draft"])
                    distances.append(DistanceArc(
                        origin=o,
                        destination=d,
                        distance_nm=dist if dist is not None else 0.0,
                        draft_required=draft,
                        is_panama=bool(int(row["IsPanama"].strip()) if row["IsPanama"].strip() else 0),
                        is_suez=bool(int(row["IsSuez"].strip()) if row["IsSuez"].strip() else 0),
                        provenance=ProvenanceRecord(
                            source_file="dist_dense.csv",
                            source_row=i,
                            instance=name,
                        ),
                    ))

        # --- Sparse distances (subset for active ports) ---
        sparse_distances: List[DistanceArc] = []
        sparse_path = self._root / "dist_sparse.csv"
        self._file_hashes["dist_sparse.csv"] = sha256_of(sparse_path)
        with sparse_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                if len(row) < 3:
                    continue
                o, d = row[0].strip(), row[1].strip()
                dist = _safe_float(row[2])
                if o in active_ports and d in active_ports and dist is not None:
                    sparse_distances.append(DistanceArc(
                        origin=o,
                        destination=d,
                        distance_nm=dist,
                        draft_required=None,
                        is_panama=False,
                        is_suez=False,
                        provenance=ProvenanceRecord(
                            source_file="dist_sparse.csv",
                            source_row=i,
                            instance=name,
                        ),
                    ))

        # --- Demands (with optional transittime revision) ---
        demands: List[Demand] = []
        tt_override: Dict[Tuple[str, str], int] = {}

        if self._apply_tt_revision and name in TRANSTIME_APPLIED_TO:
            rev_path = self._root / TRANSTIME_REVISION_DIR / f"Demand_{name}_tt.csv"
            if rev_path.exists():
                self._file_hashes[f"{TRANSTIME_REVISION_DIR}/Demand_{name}_tt.csv"] = sha256_of(rev_path)
                with rev_path.open("r", encoding="utf-8", newline="") as rf:
                    rreader = csv.DictReader(rf, delimiter="\t")
                    for ttr in rreader:
                        key = (ttr["Origin"].strip(), ttr["Destination"].strip())
                        tt_override[key] = int(ttr["TransitTime"].strip())

        for row, row_num in raw_demands:
            key = (row["Origin"].strip(), row["Destination"].strip())
            tt = int(row["TransitTime"].strip())
            if key in tt_override:
                tt = tt_override[key]
            demands.append(Demand(
                origin=row["Origin"].strip(),
                destination=row["Destination"].strip(),
                ffe_per_week=float(row["FFEPerWeek"].strip()),
                revenue=float(row["Revenue_1"].strip()),
                max_transit_time=tt,
                provenance=ProvenanceRecord(
                    source_file=demand_file,
                    source_row=row_num,
                    instance=name,
                ),
            ))

        # --- Assemble instance ---
        metadata = InstanceMetadata(
            name=name,
            active_port_count=len(ports),
            vessel_type_count=len(vessel_types),
            total_vessels=sum(e.quantity for e in fleet),
            demand_count=len(demands),
            distance_arc_count=len(distances),
            sparse_arc_count=len(sparse_distances),
        )

        provenance = DatasetProvenance(
            source_root=str(self._root),
            files=dict(self._file_hashes),
        )

        instance = LINERLIBInstance(
            name=name,
            vessel_types=vessel_types,
            ports=ports,
            fleet=fleet,
            distances=distances,
            sparse_distances=sparse_distances,
            demands=demands,
            metadata=metadata,
            provenance=provenance,
        )

        # --- Validation ---
        if validate:
            from .validation import Validator
            v = Validator(fail_fast=self._strict)
            report = v.validate(instance, global_ports=self._global_ports)
            if self._strict and report.has_errors():
                report.raise_errors()

        return instance

    # ---- internals ----

    def _ensure_globals(self) -> None:
        """Lazy-load shared files (ports + fleet_data) once."""
        if self._global_ports is not None:
            return

        # --- Global ports catalogue ---
        ports_path = self._root / "ports.csv"
        self._file_hashes["ports.csv"] = sha256_of(ports_path)
        ports: Dict[str, Port] = {}
        with ports_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                code = row["UNLocode"].strip()
                ports[code] = Port(
                    unlocode=code,
                    name=row["name"].strip(),
                    country=row["Country"].strip() if row["Country"].strip() else None,
                    cabotage_region=row["Cabotage_Region"].strip(),
                    d_region=row["D_Region"].strip() if row["D_Region"].strip() else None,
                    longitude=_safe_float(row["Longitude"]),
                    latitude=_safe_float(row["Latitude"]),
                    draft=_safe_float(row["Draft"]),
                    cost_per_full=_safe_float(row["CostPerFULL"]),
                    cost_per_full_transfer=_safe_float(row["CostPerFULLTrnsf"]),
                    port_call_cost_fixed=_safe_float(row["PortCallCostFixed"], 0.0),
                    port_call_cost_per_ffe=_safe_float(row["PortCallCostPerFFE"], 0.0),
                    provenance=ProvenanceRecord(
                        source_file="ports.csv",
                        source_row=i,
                    ),
                )
        self._global_ports = ports

        # --- Global vessel types ---
        fleet_path = self._root / "fleet_data.csv"
        self._file_hashes["fleet_data.csv"] = sha256_of(fleet_path)
        vessels: Dict[str, VesselType] = {}
        with fleet_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader, start=1):
                vessels[row["Vessel class"].strip()] = VesselType(
                    vessel_class=row["Vessel class"].strip(),
                    capacity_ffe=int(row["Capacity FFE"].strip()),
                    tc_rate_daily=int(row["TC rate daily (fixed Cost)"].strip()),
                    draft=float(row["draft"].strip()),
                    min_speed=float(row["minSpeed"].strip()),
                    max_speed=float(row["maxSpeed"].strip()),
                    design_speed=float(row["designSpeed"].strip()),
                    bunker_ton_per_day_at_design=float(row["Bunker ton per day at designSpeed"].strip()),
                    idle_consumption_ton_per_day=float(row["Idle Consumption ton/day"].strip()),
                    panama_fee=_safe_int(row["panamaFee"]),
                    suez_fee=_safe_int(row["suezFee"]),
                    provenance=ProvenanceRecord(
                        source_file="fleet_data.csv",
                        source_row=i,
                    ),
                )
        self._global_vessel_types = vessels

    def file_hashes(self) -> Dict[str, str]:
        """Return the mapping of loaded-file relative-path -> sha256 hex."""
        return dict(self._file_hashes)
