"""
P13 — Inference Result Schema.

Defines the canonical output container for a completed inference run.
Every field is traceable back to the input checkpoint, instance, and seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ServiceStep:
    """
    Record of a single service added during inference.

    Attributes
    ----------
    step_index : int
        0-based step number within the episode.
    vessel_class : str
        Vessel class selected for this service.
    port_sequence : list[str]
        Ordered port UNLOCODEs forming the cyclic rotation.
    service_id : int
        Auto-assigned service ID from the environment.
    log_prob : float or None
        Log-probability of the action under the policy (None if unavailable).
    entropy : float or None
        Entropy of the policy distribution at this step.
    reward_raw : float
        Raw incremental reward η_{t+1} - η_t.
    reward_normalized : float
        Normalized reward (η_{t+1} - η_t) / η_1.
    eta_cumulative : float
        Cumulative network profit after this service.
    fleet_after : dict[str, float]
        Remaining fleet counts per class after this step.
    terminated : bool
        Whether the episode terminated after this step.
    truncated : bool
        Whether the episode was truncated (safety cap reached).
    termination_reason : str or None
        Reason for termination if applicable.
    """

    step_index: int
    vessel_class: str
    port_sequence: List[str]
    service_id: int
    log_prob: Optional[float]
    entropy: Optional[float]
    reward_raw: float
    reward_normalized: float
    eta_cumulative: float
    fleet_after: Dict[str, float]
    terminated: bool = False
    truncated: bool = False
    termination_reason: Optional[str] = None


@dataclass
class InferenceResult:
    """
    Complete result of a single inference episode.

    This is the authoritative output schema for P13. Every field is required
    for downstream evaluation (P14 experiment tracking, P15 common evaluator).

    Attributes
    ----------
    dataset : str
        Dataset/instance identifier (e.g., "Baltic").
    instance_name : str
        Instance name (may differ from dataset if multiple instances exist).
    policy_type : str
        "encoder_only" or "encoder_decoder".
    checkpoint_path : str
        Path to the loaded checkpoint file.
    checkpoint_hash : str or None
        SHA-256 of the checkpoint file (for traceability).
    seed : int
        Random seed used for this inference run.
    deterministic : bool
        Whether this run used deterministic inference.
    services : list[ServiceStep]
        All services added during the episode, in order.
    total_services : int
        Number of services successfully added.
    final_eta : float
        Final network profit η computed by P3 MCF evaluation.
    final_mcf_result : dict or None
        Full MCFResult summary (cost decomposition) from final evaluation.
    runtime_seconds : float
        Wall-clock time for the full inference episode.
    termination_reason : str
        One of: "vessel_exhaustion", "demand_satisfied", "safety_cap_reached",
        "error", or None if still running.
    is_truncated : bool
        True if the episode hit the safety cap (truncated, not terminated).
    warnings : list[str]
        Non-fatal issues encountered during inference.
    errors : list[str]
        Fatal errors that prevented completion.
    diagnostics : dict or None
        Per-step diagnostics if record_diagnostics=True.
    timestamp : str
        ISO-format UTC timestamp of completion.
    software_version : dict
        Versions of key libraries used.
    """

    dataset: str
    instance_name: str
    policy_type: str
    checkpoint_path: str
    checkpoint_hash: Optional[str]
    seed: int
    deterministic: bool
    services: List[ServiceStep] = field(default_factory=list)
    total_services: int = 0
    final_eta: float = 0.0
    final_mcf_result: Optional[Dict[str, Any]] = None
    runtime_seconds: float = 0.0
    termination_reason: Optional[str] = None
    is_truncated: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    diagnostics: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    software_version: Dict[str, str] = field(default_factory=dict)

    # ---- Convenience accessors ----

    @property
    def is_complete(self) -> bool:
        """True if the inference episode terminated naturally or by truncation."""
        return self.termination_reason is not None

    @property
    def is_success(self) -> bool:
        """True if the episode completed without fatal errors."""
        return self.is_complete and len(self.errors) == 0

    @property
    def rejected_demand(self) -> float:
        """Total rejected demand from final MCF evaluation."""
        if self.final_mcf_result:
            return self.final_mcf_result.get("rejected_demand", 0.0)
        return 0.0

    @property
    def routed_demand(self) -> float:
        """Total routed demand from final MCF evaluation."""
        if self.final_mcf_result:
            return self.final_mcf_result.get("routed_demand", 0.0)
        return 0.0

    @property
    def fleet_remaining(self) -> Dict[str, float]:
        """Remaining fleet after the last step."""
        if self.services:
            return self.services[-1].fleet_after
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dict for JSON storage."""
        return {
            "dataset": self.dataset,
            "instance_name": self.instance_name,
            "policy_type": self.policy_type,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "total_services": self.total_services,
            "final_eta": self.final_eta,
            "final_mcf_result": self.final_mcf_result,
            "runtime_seconds": self.runtime_seconds,
            "termination_reason": self.termination_reason,
            "is_truncated": self.is_truncated,
            "warnings": self.warnings,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "software_version": self.software_version,
            "services": [
                {
                    "step_index": s.step_index,
                    "vessel_class": s.vessel_class,
                    "port_sequence": s.port_sequence,
                    "service_id": s.service_id,
                    "log_prob": s.log_prob,
                    "entropy": s.entropy,
                    "reward_raw": s.reward_raw,
                    "reward_normalized": s.reward_normalized,
                    "eta_cumulative": s.eta_cumulative,
                    "fleet_after": s.fleet_after,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                    "termination_reason": s.termination_reason,
                }
                for s in self.services
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """Deserialize from a flat dict."""
        services_data = data.pop("services", [])
        services = [
            ServiceStep(**s)
            for s in services_data
        ]
        result = cls(
            **data,
            services=services,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"InferenceResult(dataset={self.dataset!r}, "
            f"policy={self.policy_type!r}, "
            f"services={self.total_services}, "
            f"eta={self.final_eta:,.2f}, "
            f"term={self.termination_reason!r}, "
            f"trunc={self.is_truncated})"
        )
