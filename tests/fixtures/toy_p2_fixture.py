"""
SYNTHETIC TEST FIXTURE -- P2 MATHEMATICAL VALIDATION

This fixture is a minimal hand-verifiable example used ONLY for testing
the P2 mathematical formulation. It is NOT part of the LINERLIB benchmark
dataset and must never be mixed with real data.

Instance: TOY_2PORT
- 2 ports, 1 demand, 1 vessel class
- Designed for exact manual verification of all cost components
"""

from __future__ import annotations
from data.instance import (
    Demand, DistanceArc, FleetEntry, InstanceMetadata,
    LINERLIBInstance, Port, ProvenanceRecord, VesselType,
    DatasetProvenance,
)


def make_toy_2port_instance(profitable: bool = False) -> LINERLIBInstance:
    """
    A 2-port synthetic fixture for P2 mathematical validation.

    Parameters
    ----------
    profitable : bool
        If True, uses reduced port costs and higher revenue to produce
        a positive-profit instance. If False (default), uses high port
        costs producing negative profit (tests cost calculation accuracy).
    """
    if profitable:
        port_f_fixed = 100.0
        revenue = 50.0
    else:
        port_f_fixed = 5000.0
        revenue = 10.0

    ports = {
        "TOY_A": Port(
            unlocode="TOY_A", name="Test A", country="X",
            cabotage_region="X", d_region=None,
            longitude=0.0, latitude=0.0, draft=10.0,
            cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=port_f_fixed,
            port_call_cost_per_ffe=0.5,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=1),
        ),
        "TOY_B": Port(
            unlocode="TOY_B", name="Test B", country="Y",
            cabotage_region="Y", d_region=None,
            longitude=1.0, latitude=1.0, draft=10.0,
            cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=port_f_fixed,
            port_call_cost_per_ffe=0.5,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=2),
        ),
    }

    vessels = {
        "Toy_V1": VesselType(
            vessel_class="Toy_V1", capacity_ffe=200, tc_rate_daily=100,
            draft=10.0, min_speed=8.0, max_speed=15.0, design_speed=10.0,
            bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
            panama_fee=0, suez_fee=0,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=1),
        ),
    }

    demands = [
        Demand(
            origin="TOY_A", destination="TOY_B",
            ffe_per_week=100.0, revenue=revenue, max_transit_time=10,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=1),
        ),
    ]

    distances = [
        DistanceArc(
            origin="TOY_A", destination="TOY_B", distance_nm=100.0,
            draft_required=10.0, is_panama=False, is_suez=False,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=1),
        ),
        DistanceArc(
            origin="TOY_B", destination="TOY_A", distance_nm=100.0,
            draft_required=10.0, is_panama=False, is_suez=False,
            provenance=ProvenanceRecord(source_file="synthetic_p2", source_row=2),
        ),
    ]

    fleet = [FleetEntry(vessel_class="Toy_V1", quantity=2)]

    metadata = InstanceMetadata(
        name="TOY_2PORT" + ("_PROFITABLE" if profitable else ""),
        active_port_count=2,
        vessel_type_count=1,
        total_vessels=2,
        demand_count=1,
        distance_arc_count=2,
    )

    return LINERLIBInstance(
        name="TOY_2PORT" + ("_PROFITABLE" if profitable else ""),
        ports=ports,
        vessel_types=vessels,
        demands=demands,
        distances=distances,
        fleet=fleet,
        metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE -- P2]"),
    )


def compute_toy_expected_profit(profitable: bool = False) -> dict:
    """
    Compute the expected profit for the toy instance using hand-derived formulas.

    Returns a dictionary with all intermediate cost components and final profit.
    """
    # Parameters
    tour_distance = 200.0  # nm (A->B->A)
    design_speed = 10.0  # knots
    vessel_cap = 200  # FFE
    tc_daily = 100  # USD/day
    fs_daily = 50  # USD/day (sailing fuel, pre-converted)
    fi_daily = 10  # USD/day (idle fuel, pre-converted)
    pf = 5000.0 if not profitable else 100.0  # fixed port call cost
    pv = 0.5  # variable port call cost per FFE
    pl = 1.0  # loading/unloading cost per FFE
    pt = 0.5  # transshipment cost per FFE
    fleet_qty = 2
    demand_q = 100.0  # FFE/week
    demand_r = 10.0 if not profitable else 50.0  # USD/FFE
    Y_d = 1000  # USD/FFE (paper value, Appendix A.1)
    num_ports = 2  # A and B

    # Step 1: Vessel requirement
    n_vs = tour_distance / (design_speed * 7)  # = 200/70 = 2.857143

    # Step 2: Edge capacity
    edge_capacity = n_vs * vessel_cap  # = 571.43 FFE/week

    # Step 3: Flow assignment
    satisfied = min(demand_q, edge_capacity)  # = 100 (all satisfied)
    rejected = demand_q - satisfied  # = 0

    # Step 4: Revenue
    R_total = demand_r * satisfied

    # Step 5: Rejected demand penalty
    C_reject = Y_d * rejected

    # Step 6: Handling cost
    C_handle = pl * satisfied + pl * satisfied  # loading + unloading = 200

    # Step 7: Service cost (weekly factor 7)
    C_service = 7 * n_vs * tc_daily

    # Step 8: Unused vessel cost
    C_unused = -7 * (fleet_qty - n_vs) * tc_daily

    # Step 9: Voyage cost
    # Port calls: num_ports * (pf + pv * cap) * n_vs
    C_port = num_ports * (pf + pv * vessel_cap) * n_vs
    # Sailing fuel: (tour_distance / (speed * 24)) * fs * n_vs
    sailing_days = tour_distance / (design_speed * 24)
    C_sailing_fuel = sailing_days * fs_daily * n_vs
    # Idle fuel: num_ports * 1 day * fi * n_vs
    C_idle_fuel = num_ports * 1.0 * fi_daily * n_vs
    C_canal = 0  # no canals in toy instance
    C_voyage = C_port + C_sailing_fuel + C_idle_fuel + C_canal

    # Step 10: Total profit
    eta = R_total - C_reject - C_handle - C_service - C_unused - C_voyage

    return {
        "n_vs": n_vs,
        "edge_capacity": edge_capacity,
        "satisfied": satisfied,
        "rejected": rejected,
        "R_total": R_total,
        "C_reject": C_reject,
        "C_handle": C_handle,
        "C_service": C_service,
        "C_unused": C_unused,
        "C_port": C_port,
        "C_sailing_fuel": C_sailing_fuel,
        "C_idle_fuel": C_idle_fuel,
        "C_canal": C_canal,
        "C_voyage": C_voyage,
        "eta": eta,
    }


if __name__ == "__main__":
    for profitable in [False, True]:
        result = compute_toy_expected_profit(profitable)
        label = "PROFITABLE" if profitable else "NON-PROFITABLE"
        print(f"\n{'='*60}")
        print(f"TOY INSTANCE: TOY_2PORT_{label}")
        print(f"{'='*60}")
        print(f"  n_vs (vessels):     {result['n_vs']:.6f}")
        print(f"  Edge capacity:      {result['edge_capacity']:.2f} FFE/week")
        print(f"  Satisfied demand:   {result['satisfied']:.1f} FFE/week")
        print(f"  Rejected demand:    {result['rejected']:.1f} FFE/week")
        print(f"  R_total:            {result['R_total']:,.2f} USD")
        print(f"  C_reject:           {result['C_reject']:,.2f} USD")
        print(f"  C_handle:           {result['C_handle']:,.2f} USD")
        print(f"  C_service:          {result['C_service']:,.2f} USD")
        print(f"  C_unused:           {result['C_unused']:,.2f} USD")
        print(f"  C_port:             {result['C_port']:,.2f} USD")
        print(f"  C_sailing_fuel:     {result['C_sailing_fuel']:,.2f} USD")
        print(f"  C_idle_fuel:        {result['C_idle_fuel']:,.2f} USD")
        print(f"  C_canal:            {result['C_canal']:,.2f} USD")
        print(f"  C_voyage:           {result['C_voyage']:,.2f} USD")
        print(f"  NET PROFIT (eta):   {result['eta']:,.2f} USD")
