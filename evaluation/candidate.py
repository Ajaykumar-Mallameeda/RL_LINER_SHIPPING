"""
P15 — Candidate Solution Schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcf.expanded_graph import ServiceDefinition


@dataclass
class CandidateService:
    """A single service within a candidate network."""

    service_id: int
    vessel_class: str
    port_sequence: List[str]
    n_vs: Optional[float] = None

    def to_service_definition(self) -> ServiceDefinition:
        return ServiceDefinition(
            service_id=self.service_id,
            vessel_class=self.vessel_class,
            port_sequence=list(self.port_sequence),
        )


@dataclass
class CandidateSolution:
    """
    Canonical representation of a candidate network solution.

    Method-agnostic: works for RL, GA, MILP, or reference solutions.
    """

    dataset: str
    instance: str
    services: List[CandidateService] = field(default_factory=list)
    method: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    @property
    def n_services(self) -> int:
        return len(self.services)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "instance": self.instance,
            "services": [
                {
                    "service_id": s.service_id,
                    "vessel_class": s.vessel_class,
                    "port_sequence": s.port_sequence,
                    "n_vs": s.n_vs,
                }
                for s in self.services
            ],
            "method": self.method,
            "provenance": self.provenance,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateSolution":
        services = [
            CandidateService(
                service_id=s["service_id"],
                vessel_class=s["vessel_class"],
                port_sequence=s["port_sequence"],
                n_vs=s.get("n_vs"),
            )
            for s in data.get("services", [])
        ]
        return cls(
            dataset=data["dataset"],
            instance=data["instance"],
            services=services,
            method=data.get("method", ""),
            provenance=data.get("provenance", {}),
            seed=data.get("seed"),
        )
