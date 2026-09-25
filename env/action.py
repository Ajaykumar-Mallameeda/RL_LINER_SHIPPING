"""
P4 — Service action abstraction for the LSNDP environment.

Defines the canonical representation of an action A_t = (A_v,t, A_p,t):
  - A_v,t: selected vessel class (string name from instance.vessel_types)
  - A_p,t: ordered sequence of port UNLOCODEs forming a cyclic rotation

This is the environment-level action contract. P6 will later own the mechanism
for generating such actions via neural policies.

Evidence tags:
  [PAPER] — Paper Section 4, Eq. 3: A_t = (vessel selection, port sequence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ServiceAction:
    """
    Canonical representation of a service-action for the LSNDP environment.

    A single action generates one complete round-trip liner service (rotation).

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class selected (e.g., "Feeder_450").
        Must exist in the instance's vessel_types.
    port_sequence : list[str]
        Ordered sequence of port UNLOCODEs forming a cyclic rotation.
        The last port connects back to the first implicitly.
    service_id : Optional[int]
        Optional unique identifier for the service. If None, auto-assigned
        by the environment based on insertion order.

    Evidence
    --------
    [PAPER] Section 4, Eq. 3:
        A_t = (A_{v,t}, A_{p,t})
        A_{v,t} ∈ {1, ..., V}  — index of vessel class selected
        A_{p,t} = (p_1, p_2, ..., p_m)  — ordered sequence of ports
    """
    vessel_class: str
    port_sequence: List[str]
    service_id: Optional[int] = None

    @property
    def num_ports(self) -> int:
        """Number of ports in the service sequence."""
        return len(self.port_sequence)

    @property
    def is_valid_structure(self) -> bool:
        """Check basic structural validity of the action."""
        if not self.vessel_class:
            return False
        if not self.port_sequence:
            return False
        # Need at least 2 ports to form a meaningful round-trip
        if len(self.port_sequence) < 2:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"ServiceAction(vessel_class={self.vessel_class!r}, "
            f"port_sequence={self.port_sequence!r}, "
            f"service_id={self.service_id})"
        )
