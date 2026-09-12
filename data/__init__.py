"""
P1 — LINERLIB Data Foundation package.

Public API (for later phases P2+):
    from data.linerlib_loader import LINERLIBLoader
    from data.validation import Validator, Severity, DataQualityReport
    from data.normalization import Normalizer
    from data.provenance import build_manifest, sha256_of

Usage:
    loader = LINERLIBLoader(root="data/LINERLIB-master (1)/LINERLIB-master/data")
    instance = loader.load("WorldSmall")
"""

from .instance import (
    LINERLIBInstance,
    Port,
    VesselType,
    FleetEntry,
    DistanceArc,
    Demand,
    InstanceMetadata,
    ProvenanceRecord,
    DatasetProvenance,
)
from .schema import SCHEMA_VERSION, INSTANCE_DEFS
from .linerlib_loader import LINERLIBLoader
from .validation import Validator, Severity, DataQualityReport
from .normalization import Normalizer, NormalizedInstance, NormalizerState
from .provenance import sha256_of, build_manifest, save_manifest, load_manifest

__all__ = [
    "LINERLIBInstance",
    "Port",
    "VesselType",
    "FleetEntry",
    "DistanceArc",
    "Demand",
    "InstanceMetadata",
    "ProvenanceRecord",
    "DatasetProvenance",
    "SCHEMA_VERSION",
    "INSTANCE_DEFS",
    "LINERLIBLoader",
    "Validator",
    "Severity",
    "DataQualityReport",
    "Normalizer",
    "NormalizedInstance",
    "NormalizerState",
    "sha256_of",
    "build_manifest",
    "save_manifest",
    "load_manifest",
]
