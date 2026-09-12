"""
P1 – LINERLIB data validation.

Severity levels
---------------
ERROR   – hard failure: canonical instance should not be used until resolved.
WARNING – soft failure: instance is loadable but something suspicious was found.
INFO    – purely informational; does not affect downstream consumers.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class Severity(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass
class Finding:
    severity: Severity
    code: str                # machine-readable identifier
    message: str
    file: Optional[str] = None
    row: Optional[int] = None
    details: Optional[dict] = None


@dataclass
class DataQualityReport:
    instance_name: str
    findings: List[Finding] = field(default_factory=list)

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    def infos(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.INFO]

    def has_errors(self) -> bool:
        return bool(self.errors())

    def raise_errors(self) -> None:
        errs = self.errors()
        if errs:
            lines = [f"- [{f.code}] {f.message}" for f in errs]
            raise ValueError(
                f"Validation failed for instance with {len(errs)} error(s):\n"
                + "\n".join(lines)
            )

    def summary(self) -> str:
        lines = [
            f"DATA QUALITY REPORT",
            f"===================",
            f"Instance : {self.instance_name}",
            f"Errors   : {len(self.errors())}",
            f"Warnings : {len(self.warnings())}",
            f"Info     : {len(self.infos())}",
            "",
        ]
        for f in self.errors():
            lines.append(f"  ERROR [{f.code}]: {f.message}")
        for f in self.warnings():
            lines.append(f"  WARN  [{f.code}]: {f.message}")
        for f in self.infos():
            lines.append(f"  INFO  [{f.code}]: {f.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class Validator:
    """
    Validate a LINERLIBInstance against structural, referential, numeric and
    domain rules.  Never mutates the instance — produces a DataQualityReport.
    """

    def __init__(self, fail_fast: bool = False) -> None:
        self.fail_fast = fail_fast

    # ---- public API ----

    def validate(
        self,
        instance,
        global_ports: Optional[Dict] = None,
    ) -> DataQualityReport:
        """
        Validate *instance* (a LINERLIBInstance from data.instance).

        Parameters
        ----------
        global_ports :
            Full ports catalogue (UNLOCODE -> Port) keyed by UNLOCODE.
            If None, uses instance.ports as the universe.
        """
        report = DataQualityReport(instance_name=instance.name)
        all_ports = global_ports or instance.ports

        self._check_structural(report, instance)
        self._check_ports(report, instance, all_ports)
        self._check_vessels(report, instance)
        self._check_distances(report, instance, all_ports)
        self._check_demands(report, instance, all_ports)
        self._check_referential_integrity(report, instance, all_ports)

        return report

    # ---- internal checks ----

    def _check_structural(self, report: DataQualityReport, inst) -> None:
        # ports.csv unique UNLocode check happens at load time; here we verify
        # the instance's own port dict has unique keys.
        codes = list(inst.ports.keys())
        dupes = [c for c, n in Counter(codes).items() if n > 1]
        for d in dupes:
            report.add(Finding(
                Severity.ERROR, "STRUCT_PORT_DUP",
                f"Duplicate port key '{d}' in instance ports dict.",
                file="ports.csv",
            ))

    def _check_ports(self, report: DataQualityReport, inst, all_ports: Dict) -> None:
        for code, port in inst.ports.items():
            # Missing coordinates
            if port.latitude is None or port.longitude is None:
                report.add(Finding(
                    Severity.INFO, "PORT_MISSING_COORD",
                    f"Port {code} ({port.name}) missing latitude/longitude.",
                    file="ports.csv",
                    row=port.provenance.source_row,
                ))
            else:
                lat, lon = port.latitude, port.longitude
                if not (-90 <= lat <= 90):
                    report.add(Finding(
                        Severity.ERROR, "PORT_INVALID_LAT",
                        f"Port {code} has invalid latitude {lat}.",
                        file="ports.csv",
                        row=port.provenance.source_row,
                    ))
                if not (-180 <= lon <= 180):
                    report.add(Finding(
                        Severity.ERROR, "PORT_INVALID_LON",
                        f"Port {code} has invalid longitude {lon}.",
                        file="ports.csv",
                        row=port.provenance.source_row,
                    ))

            # Missing draft
            if port.draft is None:
                report.add(Finding(
                    Severity.INFO, "PORT_MISSING_DRAFT",
                    f"Port {code} ({port.name}) missing draft value.",
                    file="ports.csv",
                    row=port.provenance.source_row,
                ))

    def _check_vessels(self, report: DataQualityReport, inst) -> None:
        for vc, vt in inst.vessel_types.items():
            if vt.panama_fee is None:
                report.add(Finding(
                    Severity.WARNING, "VESSEL_MISSING_PANAMA_FEE",
                    f"Vessel type '{vc}' has no panamaFee defined.",
                    file="fleet_data.csv",
                    row=vt.provenance.source_row,
                ))
            if vt.suez_fee is None:
                report.add(Finding(
                    Severity.WARNING, "VESSEL_MISSING_SUEZ_FEE",
                    f"Vessel type '{vc}' has no suezFee defined.",
                    file="fleet_data.csv",
                    row=vt.provenance.source_row,
                ))
            if vt.capacity_ffe <= 0:
                report.add(Finding(
                    Severity.ERROR, "VESSEL_BAD_CAPACITY",
                    f"Vessel type '{vc}' has non-positive capacity: {vt.capacity_ffe}.",
                    file="fleet_data.csv",
                    row=vt.provenance.source_row,
                ))

    def _check_distances(self, report: DataQualityReport, inst, all_ports: Dict) -> None:
        # Track seen forward arcs for asymmetry detection
        forward_dist: Dict[Tuple[str, str], float] = {}

        for arc in inst.distances:
            if arc.origin not in all_ports:
                report.add(Finding(
                    Severity.ERROR, "DIST_UNKNOWN_ORIGIN",
                    f"Distance arc origin '{arc.origin}' not in ports catalogue.",
                    file="dist_dense.csv",
                    row=arc.provenance.source_row,
                ))
            if arc.destination not in all_ports:
                report.add(Finding(
                    Severity.ERROR, "DIST_UNKNOWN_DEST",
                    f"Distance arc destination '{arc.destination}' not in ports catalogue.",
                    file="dist_dense.csv",
                    row=arc.provenance.source_row,
                ))
            if arc.distance_nm < 0:
                report.add(Finding(
                    Severity.ERROR, "DIST_NEGATIVE",
                    f"Negative distance {arc.distance_nm} between {arc.origin} and {arc.destination}.",
                    file="dist_dense.csv",
                    row=arc.provenance.source_row,
                ))
            if arc.draft_required is None:
                report.add(Finding(
                    Severity.INFO, "DIST_MISSING_DRAFT",
                    f"Empty draft for distance arc {arc.origin}->{arc.destination}.",
                    file="dist_dense.csv",
                    row=arc.provenance.source_row,
                ))

            pair = (arc.origin, arc.destination)
            forward_dist[pair] = arc.distance_nm

        # Asymmetry warning: if both (a,b) and (b,a) exist with different values
        warned: set = set()
        for (o, d), v in forward_dist.items():
            rev = (d, o)
            if rev in forward_dist and rev not in warned:
                rv = forward_dist[rev]
                if abs(v - rv) > 1.0:
                    report.add(Finding(
                        Severity.WARNING, "DIST_ASYMMETRIC",
                        f"Asymmetric distance: {o}->{d}={v}, {d}->{o}={rv} "
                        f"(diff={abs(v-rv):.1f} nm).",
                        file="dist_dense.csv",
                        details={"origin": o, "destination": d, "forward": v, "reverse": rv},
                    ))
                    warned.add(pair)
                    warned.add(rev)

    def _check_demands(self, report: DataQualityReport, inst, all_ports: Dict) -> None:
        od_pairs: Dict[Tuple[str, str], int] = Counter()

        for dem in inst.demands:
            if dem.origin not in all_ports:
                report.add(Finding(
                    Severity.ERROR, "DEM_UNKNOWN_ORIGIN",
                    f"Demand origin '{dem.origin}' not in ports catalogue.",
                    file=f"Demand_{inst.name}.csv",
                    row=dem.provenance.source_row,
                ))
            if dem.destination not in all_ports:
                report.add(Finding(
                    Severity.ERROR, "DEM_UNKNOWN_DEST",
                    f"Demand destination '{dem.destination}' not in ports catalogue.",
                    file=f"Demand_{inst.name}.csv",
                    row=dem.provenance.source_row,
                ))
            if dem.ffe_per_week <= 0:
                report.add(Finding(
                    Severity.ERROR, "DEM_ZERO_OR_NEG_FFE",
                    f"Demand {dem.origin}->{dem.destination} has non-positive FFE: {dem.ffe_per_week}.",
                    file=f"Demand_{inst.name}.csv",
                    row=dem.provenance.source_row,
                ))
            if dem.revenue < 0:
                report.add(Finding(
                    Severity.ERROR, "DEM_NEG_REVENUE",
                    f"Demand {dem.origin}->{dem.destination} has negative revenue: {dem.revenue}.",
                    file=f"Demand_{inst.name}.csv",
                    row=dem.provenance.source_row,
                ))
            if dem.max_transit_time <= 0:
                report.add(Finding(
                    Severity.ERROR, "DEM_BAD_TRANSIT_TIME",
                    f"Demand {dem.origin}->{dem.destination} has invalid transit time: {dem.max_transit_time}.",
                    file=f"Demand_{inst.name}.csv",
                    row=dem.provenance.source_row,
                ))
            od_pairs[(dem.origin, dem.destination)] += 1

        # Duplicate OD pairs
        for (o, d), cnt in od_pairs.items():
            if cnt > 1:
                report.add(Finding(
                    Severity.WARNING, "DEM_DUP_OD_PAIR",
                    f"Duplicate OD pair {o}->{d} appears {cnt} times (different FFE/revenue values).",
                    file=f"Demand_{inst.name}.csv",
                    details={"origin": o, "destination": d, "count": cnt},
                ))

    def _check_referential_integrity(
        self, report: DataQualityReport, inst, all_ports: Dict
    ) -> None:
        # Fleet vessel classes must exist in vessel_types
        fleet_classes = {e.vessel_class for e in inst.fleet}
        known_classes = set(inst.vessel_types.keys())
        unknown_fleet = fleet_classes - known_classes
        for vc in sorted(unknown_fleet):
            report.add(Finding(
                Severity.ERROR, "FLEET_UNKNOWN_VESSEL_TYPE",
                f"Fleet entry references unknown vessel class '{vc}'.",
                file=f"fleet_{inst.name}.csv",
            ))

        # All demand endpoints must be in the active port set
        demand_ports = set()
        for dem in inst.demands:
            demand_ports.add(dem.origin)
            demand_ports.add(dem.destination)
        inst_ports = set(inst.ports.keys())
        extra_demand = demand_ports - inst_ports
        for p in sorted(extra_demand):
            report.add(Finding(
                Severity.WARNING, "DEM_PORT_NOT_IN_INSTANCE",
                f"Demand endpoint '{p}' not found in instance ports (may be a global-catalogue quirk).",
                file=f"Demand_{inst.name}.csv",
            ))
