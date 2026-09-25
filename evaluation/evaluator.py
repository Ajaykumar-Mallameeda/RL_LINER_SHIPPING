"""
P15 — Common Evaluator.

Evaluates any CandidateSolution using P3 MCF + P2 objective.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data.instance import LINERLIBInstance
from mcf import evaluate_network
from mcf.expanded_graph import ServiceDefinition

from .candidate import CandidateService, CandidateSolution

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete result of evaluating a candidate solution."""

    method: str
    dataset: str
    instance: str
    objective_eta: float = 0.0
    revenue: float = 0.0
    C_reject: float = 0.0
    C_handle: float = 0.0
    C_service: float = 0.0
    C_unused: float = 0.0
    C_voyage: float = 0.0
    routed_demand: float = 0.0
    rejected_demand: float = 0.0
    fleet_usage: Dict[str, float] = field(default_factory=dict)
    fleet_deviation: Dict[str, float] = field(default_factory=dict)
    service_count: int = 0
    runtime_seconds: float = 0.0
    structural_feasibility: bool = True
    mcf_status: str = "pending"
    evaluator_version: str = "P15-v1"
    data_hash: str = ""
    seed: Optional[int] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        return self.structural_feasibility and len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "instance": self.instance,
            "objective_eta": self.objective_eta,
            "revenue": self.revenue,
            "C_reject": self.C_reject,
            "C_handle": self.C_handle,
            "C_service": self.C_service,
            "C_unused": self.C_unused,
            "C_voyage": self.C_voyage,
            "routed_demand": self.routed_demand,
            "rejected_demand": self.rejected_demand,
            "fleet_usage": self.fleet_usage,
            "fleet_deviation": self.fleet_deviation,
            "service_count": self.service_count,
            "runtime_seconds": self.runtime_seconds,
            "structural_feasibility": self.structural_feasibility,
            "mcf_status": self.mcf_status,
            "evaluator_version": self.evaluator_version,
            "data_hash": self.data_hash,
            "seed": self.seed,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class CommonEvaluator:
    """
    Method-agnostic evaluator for LSNDP candidate solutions.
    """

    def __init__(self, instance: LINERLIBInstance, data_hash: str = "") -> None:
        self._instance = instance
        self._data_hash = data_hash
        self._dist_by_pair = {
            (a.origin, a.destination): a for a in instance.distances
        }

    def evaluate(
        self,
        candidate: CandidateSolution,
        compute_fleet_deviations: bool = True,
    ) -> EvaluationResult:
        start_time = time.time()
        warnings: List[str] = []
        errors: List[str] = []

        # Validate structure.
        structural_ok, struct_warnings = self._validate_structure(candidate)
        warnings.extend(struct_warnings)

        # Build services and vessel requirements.
        services = []
        vessel_requirements: Dict[str, Dict[str, float]] = {}
        for svc in candidate.services:
            svc_def = svc.to_service_definition()
            services.append(svc_def)
            n_vs = svc.n_vs
            if n_vs is None:
                n_vs = self._compute_n_vs(svc)
                if n_vs is None:
                    errors.append(f"Cannot compute n_vs for service {svc.service_id}")
                    continue
            vessel_requirements[svc.service_id] = {svc.vessel_class: n_vs}

        if errors:
            return EvaluationResult(
                method=candidate.method,
                dataset=candidate.dataset,
                instance=candidate.instance,
                structural_feasibility=False,
                mcf_status="failed",
                errors=errors,
                data_hash=self._data_hash,
                seed=candidate.seed,
                warnings=warnings,
            )

        # Run MCF.
        try:
            mcf_result = evaluate_network(
                instance=self._instance,
                services=services,
                vessel_requirements=vessel_requirements,
            )
        except Exception as e:
            errors.append(f"MCF failed: {type(e).__name__}: {e}")
            return EvaluationResult(
                method=candidate.method,
                dataset=candidate.dataset,
                instance=candidate.instance,
                structural_feasibility=structural_ok,
                mcf_status="failed",
                errors=errors,
                data_hash=self._data_hash,
                seed=candidate.seed,
                warnings=warnings,
            )

        # Fleet metrics.
        fleet_usage: Dict[str, float] = {}
        fleet_deviation: Dict[str, float] = {}
        if compute_fleet_deviations:
            fleet_usage, fleet_deviation = self._compute_fleet_metrics(vessel_requirements)

        runtime = time.time() - start_time

        return EvaluationResult(
            method=candidate.method,
            dataset=candidate.dataset,
            instance=candidate.instance,
            objective_eta=mcf_result.eta,
            revenue=mcf_result.total_revenue,
            C_reject=mcf_result.rejection_cost,
            C_handle=mcf_result.handling_cost,
            C_service=mcf_result.service_cost,
            C_unused=mcf_result.unused_vessel_cost,
            C_voyage=mcf_result.voyage_cost,
            routed_demand=mcf_result.routed_demand,
            rejected_demand=mcf_result.rejected_demand,
            fleet_usage=fleet_usage,
            fleet_deviation=fleet_deviation,
            service_count=mcf_result.num_services,
            runtime_seconds=runtime,
            structural_feasibility=structural_ok,
            mcf_status="success",
            data_hash=self._data_hash,
            seed=candidate.seed,
            warnings=warnings,
        )

    def _validate_structure(self, candidate: CandidateSolution) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        ok = True
        for svc in candidate.services:
            if svc.vessel_class not in self._instance.vessel_types:
                warnings.append(f"Service {svc.service_id}: unknown vessel '{svc.vessel_class}'")
                ok = False
            for p in svc.port_sequence:
                if p not in self._instance.ports:
                    warnings.append(f"Service {svc.service_id}: unknown port '{p}'")
                    ok = False
            if len(svc.port_sequence) < 2:
                warnings.append(f"Service {svc.service_id}: needs >=2 ports")
                ok = False
        return ok, warnings

    def _compute_n_vs(self, svc: CandidateService) -> Optional[float]:
        vt = self._instance.vessel_types.get(svc.vessel_class)
        if vt is None:
            return None
        tour_dist = 0.0
        n = len(svc.port_sequence)
        for i in range(n):
            arc = self._dist_by_pair.get((svc.port_sequence[i], svc.port_sequence[(i + 1) % n]))
            if arc and arc.distance_nm > 0:
                tour_dist += arc.distance_nm
        if tour_dist <= 0:
            return None
        return tour_dist / (vt.design_speed * 24.0 * 7.0)

    def _compute_fleet_metrics(
        self, vessel_requirements: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        fleet_usage: Dict[str, float] = {}
        fleet_initial: Dict[str, float] = {
            e.vessel_class: float(e.quantity) for e in self._instance.fleet
        }
        for sid, vreq in vessel_requirements.items():
            for vc, n_vs in vreq.items():
                fleet_usage[vc] = fleet_usage.get(vc, 0.0) + n_vs
        fleet_deviation = {
            vc: fleet_usage.get(vc, 0.0) - initial
            for vc, initial in fleet_initial.items()
        }
        return fleet_usage, fleet_deviation
