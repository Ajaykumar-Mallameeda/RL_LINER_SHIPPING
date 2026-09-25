"""
P6 — Action & Service Generation package.

Exports the public API for generating and validating liner shipping services.
"""

from .service_generator import (
    ActionValidator,
    ServiceGenerator,
    ServiceValidationResult,
    calculate_vessel_requirement,
    canonicalize_service,
    are_services_equivalent,
    make_service_action,
    make_service_definition,
    _nearest_neighbor_tsp,
    select_largest_available_vessel,
)

__all__ = [
    "ServiceGenerator",
    "ServiceValidationResult",
    "ActionValidator",
    "calculate_vessel_requirement",
    "canonicalize_service",
    "are_services_equivalent",
    "make_service_action",
    "make_service_definition",
    "_nearest_neighbor_tsp",
    "select_largest_available_vessel",
]
