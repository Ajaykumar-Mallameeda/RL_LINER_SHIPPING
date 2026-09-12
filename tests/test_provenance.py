"""
P1 – Tests for provenance and manifest generation.

Covers:
  - SHA-256 hash correctness
  - Manifest structure
  - Manifest persistence round-trip
  - Per-record provenance attached to instance objects
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.linerlib_loader import LINERLIBLoader
from data.provenance import sha256_of, build_manifest, save_manifest, load_manifest
from data.instance import ProvenanceRecord


DATA_ROOT = ROOT / "data" / "LINERLIB-master (1)" / "LINERLIB-master" / "data"
TMP_MANIFEST = ROOT / "data" / "processed" / "test_manifest.json"


def _loader():
    return LINERLIBLoader(root=str(DATA_ROOT))


# ===========================================================================
# SHA-256 hashes
# ===========================================================================

def test_sha256_deterministic():
    h1 = sha256_of(DATA_ROOT / "ports.csv")
    h2 = sha256_of(DATA_ROOT / "ports.csv")
    assert h1 == h2
    assert len(h1) == 64  # hex digest length


def test_sha256_differs_between_files():
    h_ports = sha256_of(DATA_ROOT / "ports.csv")
    h_fleet = sha256_of(DATA_ROOT / "fleet_data.csv")
    assert h_ports != h_fleet


def test_loader_file_hashes_populated():
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    hashes = loader.file_hashes()
    assert "ports.csv" in hashes
    assert "fleet_Baltic.csv" in hashes
    assert "Demand_Baltic.csv" in hashes
    assert "dist_dense.csv" in hashes
    assert len(hashes["ports.csv"]) == 64


# ===========================================================================
# Manifest generation
# ===========================================================================

def test_manifest_structure():
    loader = _loader()
    manifest = build_manifest(
        source_root=DATA_ROOT,
        instance_names=["Baltic", "WAF"],
    )
    assert manifest["source"] == "LINERLIB"
    assert manifest["schema_version"] == "1.0"
    assert "files" in manifest
    assert "instances" in manifest
    assert "Baltic" in manifest["instances"]
    assert "WAF" in manifest["instances"]


def test_manifest_file_hashes_present():
    loader = _loader()
    manifest = build_manifest(
        source_root=DATA_ROOT,
        instance_names=["Baltic"],
    )
    ports_entry = next((f for f in manifest["files"] if f["path"] == "ports.csv"), None)
    assert ports_entry is not None
    assert len(ports_entry["sha256"]) == 64
    assert ports_entry["size_bytes"] > 0


def test_manifest_save_and_load_roundtrip():
    loader = _loader()
    manifest = build_manifest(
        source_root=DATA_ROOT,
        instance_names=["Baltic"],
    )
    TMP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, TMP_MANIFEST)
    loaded = load_manifest(TMP_MANIFEST)
    assert loaded["source"] == manifest["source"]
    assert loaded["schema_version"] == manifest["schema_version"]
    # Clean up temp file.
    TMP_MANIFEST.unlink(missing_ok=True)


# ===========================================================================
# Per-record provenance
# ===========================================================================

def test_port_provenance_attached():
    loader = _loader()
    inst = loader.load("Baltic", validate=False)
    p = inst.ports["DEBRV"]
    assert p.provenance.source_file == "ports.csv"
    assert p.provenance.source_row >= 1
    assert p.provenance.dataset_version == "1.2"


def test_demand_provenance_has_instance():
    loader = _loader()
    inst = loader.load("WAF", validate=False)
    d = inst.demands[0]
    assert d.provenance.instance == "WAF"
    assert d.provenance.source_file == "Demand_WAF.csv"


def test_distance_provenance():
    loader = _loader()
    inst = loader.load("Pacific", validate=False)
    arc = inst.distances[0]
    assert arc.provenance.source_file == "dist_dense.csv"
    assert arc.provenance.instance == "Pacific"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
