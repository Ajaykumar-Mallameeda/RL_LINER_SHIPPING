"""
Provenance and manifest utilities.

Computes SHA-256 hashes on raw bytes, builds a machine-readable manifest
(JSON-serialisable), and attaches per-record provenance traces back to the
source LINERLIB file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file without loading it into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    source_root: Path,
    instance_names: List[str],
    file_hashes: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Build a JSON-serialisable data manifest.

    Parameters
    ----------
    source_root :
        Absolute path to the LINERLIB data directory.
    instance_names :
        Verified instance names to include.
    file_hashes :
        Pre-computed {relative_path: sha256_hex} mapping.
        If None, hashes are computed on-the-fly.

    Returns
    -------
    dict compatible with json.dumps for disk persistence.
    """
    if file_hashes is None:
        file_hashes = {}
        for p in sorted(source_root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(source_root))
                file_hashes[rel] = sha256_of(p)

    now_iso = datetime.now(timezone.utc).isoformat()

    manifest: Dict[str, Any] = {
        "source": "LINERLIB",
        "schema_version": "1.0",
        "dataset_version": "1.2",
        "provenance": {
            "original_publication": (
                "Brouer, Berit D., Alvarez, J. Fernando, Plum, Christian Edinger Munk, "
                "Pisinger, David, Sigurd, Mikkel M. "
                "\"A base integer programming model and benchmark suite for liner shipping "
                "network design.\" Transportation Science (forthcoming 2013)."
            ),
            "technical_report": (
                "Løfstedt, Berit, et al. "
                "\"An integer programming model and benchmark suite for liner shipping "
                "network design.\" DTU Management Engineering report 19.2010."
            ),
            "external_sources": [
                "NIMA Pub. 151 (distances)",
                "Hamburg Index (port info)",
                "Alphaliner charter rates 2000–2010",
                "Drewry Shipping Consultants (freight rates)",
                "Hapag-Lloyd freight quotes",
            ],
            "data_last_viewed": "January 2011",
            "manifest_generated": now_iso,
        },
        "files": [
            {"path": rel, "sha256": h, "size_bytes": source_root.joinpath(rel).stat().st_size}
            for rel, h in sorted(file_hashes.items())
        ],
        "instances": {},
    }

    for name in instance_names:
        manifest["instances"][name] = {
            "demand_file": f"Demand_{name}.csv",
            "fleet_file": f"fleet_{name}.csv",
            "verification_status": "VERIFIED" if name != "WorldLarge" and name != "EuropeAsia" else "DISCREPANCY",
        }

    return manifest


def save_manifest(manifest: dict, output_path: Path) -> None:
    """Persist the manifest as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: Path) -> dict:
    """Load a previously-saved manifest."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
