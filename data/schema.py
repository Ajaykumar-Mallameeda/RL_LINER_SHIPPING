"""
Canonical schema definitions and version tracker for the LINERLIB data layer.

The schema is documented in docs/P1_DATA_FOUNDATION.md §7.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Column specifications per file type.
# Each tuple is (column_name, expected_type_name, nullable).
# ---------------------------------------------------------------------------

PORTS_SCHEMA = [
    ("UNLocode",       "str",   False),
    ("name",           "str",   False),
    ("Country",        "str",   True),
    ("Cabotage_Region","str",   False),
    ("D_Region",       "str",   True),
    ("Longitude",      "float", True),
    ("Latitude",       "float", True),
    ("Draft",          "float", True),
    ("CostPerFULL",    "float", False),
    ("CostPerFULLTrnsf","float",False),
    ("PortCallCostFixed","float",False),
    ("PortCallCostPerFFE","float",False),
]

VESSEL_SCHEMA = [
    ("Vessel class",                            "str",   False),
    ("Capacity FFE",                            "int",   False),
    ("TC rate daily (fixed Cost)",              "int",   False),
    ("draft",                                   "float", False),
    ("minSpeed",                                "float", False),
    ("maxSpeed",                                "float", False),
    ("designSpeed",                             "float", False),
    ("Bunker ton per day at designSpeed",       "float", False),
    ("Idle Consumption ton/day",                "float", False),
    ("panamaFee",                               "int",   True),
    ("suezFee",                                 "int",   True),
]

DEMAND_SCHEMA = [
    ("Origin",       "str",  False),
    ("Destination",  "str",  False),
    ("FFEPerWeek",   "float",False),
    ("Revenue_1",    "float",False),
    ("TransitTime",  "int",  False),
]

DIST_DENSE_SCHEMA = [
    ("fromUNLOCODe", "str",  False),
    ("ToUNLOCODE",   "str",  False),
    ("Distance",     "float",False),
    ("Draft",        "float",True),
    ("IsPanama",     "int",  False),
    ("IsSuez",       "int",  False),
]

# dist_sparse has NO header row; columns are inferred as:
DIST_SPARSE_SCHEMA = [
    ("origin",      "str",  False),
    ("destination", "str",  False),
    ("distance_nm", "float",False),
]

ROTS_SCHEMA_KEYS = {
    "rot_id", "rot_speed", "rot_num_v", "rot_class", "rot_calls", "cargo",
}
ROT_CARGO_KEYS = {"orig", "dest", "entry", "exit", "quantity"}

# ---------------------------------------------------------------------------
# Instance definitions (verified identities from actual data inspection).
# ---------------------------------------------------------------------------

INSTANCE_DEFS: dict[str, dict] = {
    "Baltic": {
        "demand_file": "Demand_Baltic.csv",
        "fleet_file": "fleet_Baltic.csv",
        "expected_ports": 12,
        "paper_ports": 12,
    },
    "WAF": {
        "demand_file": "Demand_WAF.csv",
        "fleet_file": "fleet_WAF.csv",
        "expected_ports": 20,
        "paper_ports": 19,  # known discrepancy, acknowledged in readme v1.2
        "paper_demands": 38,  # known discrepancy: actual = 37
    },
    "Mediterranean": {
        "demand_file": "Demand_Mediterranean.csv",
        "fleet_file": "fleet_Mediterranean.csv",
        "expected_ports": 39,
        "paper_ports": 39,
    },
    "Pacific": {
        "demand_file": "Demand_Pacific.csv",
        "fleet_file": "fleet_Pacific.csv",
        "expected_ports": 45,
        "paper_ports": 45,
    },
    "WorldSmall": {
        "demand_file": "Demand_WorldSmall_Fixed_Sep.csv",
        "fleet_file": "fleet_WorldSmall.csv",
        "expected_ports": 47,
        "paper_ports": 47,
        "note": "Original Demand_WorldSmall.csv has 7 decimal-FFE rows; default uses _Fixed_Sep.",
    },
    "WorldLarge": {
        "demand_file": "Demand_WorldLarge.csv",
        "fleet_file": "fleet_WorldLarge.csv",
        "expected_ports": 201,
        "paper_ports": 197,  # discrepancy
    },
    "EuropeAsia": {
        "demand_file": "Demand_EuropeAsia.csv",
        "fleet_file": "fleet_EuropeAsia.csv",
        "expected_ports": 114,
        "paper_ports": 111,  # discrepancy
    },
}

TRANSTIME_REVISION_DIR = "transittime_revision"
TRANSTIME_APPLIED_TO = {"WAF", "Pacific", "WorldSmall", "EuropeAsia", "WorldLarge"}

# WorldSmall: use fixed variant by default (original had 7 rows with decimal
# FFEPerWeek like 1.86 instead of 1860 — clearly a formatting error).
WORLDSMALL_FIXED_DEFAULT = True
