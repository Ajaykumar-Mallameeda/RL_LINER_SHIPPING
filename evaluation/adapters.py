"""
P15 — Adapters for converting various solution formats to CandidateSolution.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .candidate import CandidateService, CandidateSolution


def rl_result_to_candidate(
    inference_result: Any,
    instance_name: str,
    method: str = "rl_encoder_only",
) -> CandidateSolution:
    """Convert an InferenceResult to a CandidateSolution."""
    from inference.result import InferenceResult
    assert isinstance(inference_result, InferenceResult)

    services = [
        CandidateService(
            service_id=step.service_id,
            vessel_class=step.vessel_class,
            port_sequence=step.port_sequence,
        )
        for step in inference_result.services
    ]

    return CandidateSolution(
        dataset=inference_result.dataset,
        instance=instance_name,
        services=services,
        method=method,
        provenance={
            "checkpoint_path": inference_result.checkpoint_path,
            "checkpoint_hash": inference_result.checkpoint_hash,
            "seed": inference_result.seed,
            "deterministic": inference_result.deterministic,
        },
        seed=inference_result.seed,
    )


def reference_solution_to_candidate(
    services: List[Dict[str, Any]],
    instance_name: str,
    method: str = "linerlib_reference",
    source_log: str = "",
) -> CandidateSolution:
    """Convert a reference/LINERLIB solution to a CandidateSolution."""
    candidate_services = [
        CandidateService(
            service_id=i,
            vessel_class=s["vessel_class"],
            port_sequence=s["port_sequence"],
            n_vs=s.get("n_vs"),
        )
        for i, s in enumerate(services)
    ]
    return CandidateSolution(
        dataset=instance_name,
        instance=instance_name,
        services=candidate_services,
        method=method,
        provenance={"source_log": source_log},
    )
