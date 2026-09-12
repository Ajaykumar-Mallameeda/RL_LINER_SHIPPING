"""
SYNTHETIC TEST FIXTURE — not part of the real LINERLIB dataset.

Tiny hand-verifiable example networks used ONLY for isolated error-condition
tests in test_data_validation.py.  These fixtures must never be mixed with
the actual LINERLIB data directory.

Example: 2-port test network with a single demand and one vessel type.
"""

from data.instance import (
    Demand, DistanceArc, FleetEntry, InstanceMetadata,
    LINERLIBInstance, Port, ProvenanceRecord, VesselType,
    DatasetProvenance,
)


def make_2port_test_instance() -> LINERLIBInstance:
    """A 2-port, 1-demand synthetic fixture for unit testing."""
    ports = {
        "TESTA": Port(
            unlocode="TESTA", name="Test A", country="X",
            cabotage_region="X", d_region=None,
            longitude=0.0, latitude=0.0, draft=5.0,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
        "TESTB": Port(
            unlocode="TESTB", name="Test B", country="Y",
            cabotage_region="Y", d_region=None,
            longitude=1.0, latitude=1.0, draft=6.0,
            cost_per_full=120.0, cost_per_full_transfer=12.0,
            port_call_cost_fixed=600.0, port_call_cost_per_ffe=3.0,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=2),
        ),
    }
    vessels = {
        "Synthetic_Vessel": VesselType(
            vessel_class="Synthetic_Vessel", capacity_ffe=500, tc_rate_daily=5000,
            draft=7.0, min_speed=10.0, max_speed=15.0, design_speed=12.0,
            bunker_ton_per_day_at_design=10.0, idle_consumption_ton_per_day=2.0,
            panama_fee=1000, suez_fee=2000,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
    }
    demands = [
        Demand(
            origin="TESTA", destination="TESTB",
            ffe_per_week=100.0, revenue=500.0, max_transit_time=7,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
    ]
    distances = [
        DistanceArc(
            origin="TESTA", destination="TESTB", distance_nm=1000.0,
            draft_required=5.0, is_panama=False, is_suez=False,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
    ]
    fleet = [FleetEntry(vessel_class="Synthetic_Vessel", quantity=2)]
    metadata = InstanceMetadata(
        name="SYNTHETIC_2PORT",
        active_port_count=2,
        vessel_type_count=1,
        total_vessels=2,
        demand_count=1,
        distance_arc_count=1,
    )
    return LINERLIBInstance(
        name="SYNTHETIC_2PORT",
        ports=ports,
        vessel_types=vessels,
        demands=demands,
        distances=distances,
        fleet=fleet,
        metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE]"),
    )


def make_duplicate_od_instance() -> LINERLIBInstance:
    """Instance with duplicate OD pairs for testing warning detection."""
    ports = {
        "DUPA": Port(
            unlocode="DUPA", name="Dup A", country="X",
            cabotage_region="X", d_region=None,
            longitude=0.0, latitude=0.0, draft=5.0,
            cost_per_full=100.0, cost_per_full_transfer=10.0,
            port_call_cost_fixed=500.0, port_call_cost_per_ffe=2.0,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=1),
        ),
        "DUPB": Port(
            unlocode="DUPB", name="Dup B", country="Y",
            cabotage_region="Y", d_region=None,
            longitude=1.0, latitude=1.0, draft=6.0,
            cost_per_full=120.0, cost_per_full_transfer=12.0,
            port_call_cost_fixed=600.0, port_call_cost_per_ffe=3.0,
            provenance=ProvenanceRecord(source_file="synthetic", source_row=2),
        ),
    }
    vessels: dict = {}
    demands = [
        Demand(origin="DUPA", destination="DUPB", ffe_per_week=10.0, revenue=100.0,
               max_transit_time=5,
               provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        Demand(origin="DUPA", destination="DUPB", ffe_per_week=20.0, revenue=150.0,
               max_transit_time=5,
               provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
    ]
    distances = []
    fleet = []
    metadata = InstanceMetadata(
        name="SYNTHETIC_DUP_OD",
        active_port_count=2,
        vessel_type_count=0,
        total_vessels=0,
        demand_count=2,
        distance_arc_count=0,
    )
    return LINERLIBInstance(
        name="SYNTHETIC_DUP_OD",
        ports=ports,
        vessel_types=vessels,
        demands=demands,
        distances=distances,
        fleet=fleet,
        metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE]"),
    )
