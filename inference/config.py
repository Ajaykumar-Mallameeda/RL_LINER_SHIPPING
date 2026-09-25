"""
P13 — Inference Configuration.

Defines the hyperparameters that control how the inference/solver operates.
Separate from training configuration (P10/P12); these knobs only affect
inference behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InferenceConfig:
    """
    Configuration for P13 inference/solver runs.

    Parameters
    ----------
    policy_type : str
        "encoder_only" or "encoder_decoder". Must match the trained checkpoint.
    deterministic : bool
        If True, use argmax selection (no sampling). If False, sample from the
        policy distribution. [ENGINEERING DECISION] Paper uses stochastic
        inference for evaluation; deterministic for reproducibility checks.
    max_services : int
        Safety cap on number of services per episode. Matches P4's
        MAX_SERVICES_SAFETY_CAP = 100. Episodes hitting this are truncated.
    seed : Optional[int]
        Random seed for stochastic inference. Required if ``deterministic=False``.
    validate_actions : bool
        If True, validate every produced ServiceAction against P6 rules before
        executing in the environment. If False, skip validation (not recommended).
    check_numerical_stability : bool
        If True, check for NaN/Inf in policy outputs and MCF results.
    record_diagnostics : bool
        If True, record per-step diagnostics (log-prob, entropy, backbone shapes).
    """

    policy_type: str = "encoder_only"
    deterministic: bool = True
    max_services: int = 100
    seed: Optional[int] = None
    validate_actions: bool = True
    check_numerical_stability: bool = True
    record_diagnostics: bool = False

    def validate(self) -> None:
        """Raise ValueError on invalid configuration."""
        if self.policy_type not in ("encoder_only", "encoder_decoder"):
            raise ValueError(
                f"policy_type must be 'encoder_only' or 'encoder_decoder', "
                f"got {self.policy_type!r}."
            )
        if self.max_services < 1:
            raise ValueError(f"max_services must be >= 1, got {self.max_services}")
        if not self.deterministic and self.seed is None:
            raise ValueError(
                "seed is required when deterministic=False"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "deterministic": self.deterministic,
            "max_services": self.max_services,
            "seed": self.seed,
            "validate_actions": self.validate_actions,
            "check_numerical_stability": self.check_numerical_stability,
            "record_diagnostics": self.record_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
