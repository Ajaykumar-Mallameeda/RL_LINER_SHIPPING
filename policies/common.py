"""
P8/P9 shared diagnostic containers.

Both policy pathways expose `PolicyDiagnostics` for downstream inspection
(see P8.7 / P9.7). No training, no PPO. Just data classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyDiagnostics:
    """
    Diagnostic bookkeeping attached to every policy decision output.

    Attributes
    ----------
    num_ports : int
        P — physical ports on the graph.
    num_candidates : int
        Number of available candidates after masking (P or V, depending on
        which mask was applied).
    num_selected : int
        Ports/vessels actually selected by the policy.
    fallback_applied : bool
        True when a deterministic fallback was used to repair an invalid draw.
    vessel_class : str or None
        Selected vessel class (encoder-only) or vessel index token selection
        at decoder sub-step 1 (encoder-decoder).
    validation_reasons : list[str]
        P6 structural validation failures, empty if valid.
    extra : dict
        Optional per-pathway fields. Consumers should not depend on its
        contents except as documented.
    """

    num_ports: int
    num_candidates: int
    num_selected: int
    fallback_applied: bool
    vessel_class: Optional[str]
    validation_reasons: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
