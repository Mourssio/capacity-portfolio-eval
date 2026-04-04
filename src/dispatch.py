"""
dispatch.py — Hourly merit-order dispatch simulator.

For a given portfolio and common random inputs, simulates 8,760 hours:
  1. Subtract renewable output from demand to get residual load
  2. Dispatch/charge storage based on threshold rules
  3. Dispatch firm generators in merit order subject to forced outages
  4. Record unserved energy for any remaining shortfall

Returns (F_i, C_ij, U_ij, Y_ij).
"""

import numpy as np
from typing import Dict, Tuple
from config import (
    Portfolio, FIRM_SUBTYPES, MERIT_ORDER, VOLL,
    STORAGE_RTE, N_HOURS,
)
from crn import DEMAND_P70, DEMAND_P30


def compute_fixed_cost(portfolio: Portfolio) -> float:
    """F_i: annualized capital cost ($). Deterministic, can be cached."""
    return portfolio.annualized_fixed_cost()


def simulate_replication(
    portfolio: Portfolio,
    common_inputs: Dict[str, np.ndarray],
) -> Tuple[float, float, float, float]:
    """
    Simulate one replication (8,760 hours) for a given portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        Candidate portfolio to simulate.
    common_inputs : dict
        Output of crn.generate_common_inputs(). Contains 'demand',
        'wind_cf', 'solar_cf', 'outage_avail'.

    Returns
    -------
    (F_i, C_ij, U_ij, Y_ij) : tuple of float
        F_i   = annualized fixed cost ($)
        C_ij  = total operating cost in this replication ($)
        U_ij  = total unserved energy (MWh)
        Y_ij  = F_i + C_ij + VOLL * U_ij
    """
    n_hours = len(common_inputs["demand"])

    # --- Unpack inputs ---
    demand = common_inputs["demand"]                    # (n_hours,)
    wind_cf = common_inputs["wind_cf"]                  # (n_hours,)
    solar_cf = common_inputs["solar_cf"]                # (n_hours,)
    outage_avail = common_inputs["outage_avail"]        # dict[str, (n_hours,)]

    # --- Portfolio parameters ---
    wind_mw = portfolio.wind_mw
    solar_mw = portfolio.solar_mw
    storage_mw = portfolio.storage_mw           # Power capacity (MW)
    storage_mwh = portfolio.storage_mwh         # Energy capacity (MWh)
    firm_subtypes = portfolio.firm_subtype_mw() # dict[str, float] MW per sub-type

    # --- Step 1-2: Renewable output and residual demand ---
    renewable_output = wind_mw * wind_cf + solar_mw * solar_cf   # (n_hours,)
    residual = np.maximum(0.0, demand - renewable_output)        # (n_hours,)

    # Also track excess renewables for charging
    excess_renew = np.maximum(0.0, renewable_output - demand)    # (n_hours,)

    # --- Step 3: Storage dispatch (threshold-based heuristic) ---
    soc = storage_mwh * 0.5  # Start at 50% state of charge
    storage_discharge = np.zeros(n_hours)
    storage_charge = np.zeros(n_hours)

    for t in range(n_hours):
        if residual[t] > 0 and demand[t] > DEMAND_P70:
            # Discharge: min(SoC, P_max, residual)
            discharge = min(soc, storage_mw, residual[t])
            storage_discharge[t] = discharge
            soc -= discharge
        elif excess_renew[t] > 0 or demand[t] < DEMAND_P30:
            # Charge from excess renewables or during off-peak
            headroom = storage_mwh - soc
            # Available charging power, accounting for RTE
            charge_power = min(storage_mw, headroom / STORAGE_RTE)
            # If excess renewables, use those; otherwise charge from grid (free in this model)
            if excess_renew[t] > 0:
                charge_power = min(charge_power, excess_renew[t])
            storage_charge[t] = charge_power
            soc += charge_power * STORAGE_RTE

    # Update residual after storage
    residual_after_storage = residual - storage_discharge  # (n_hours,)
    residual_after_storage = np.maximum(0.0, residual_after_storage)

    # --- Step 4: Firm dispatch in merit order ---
    total_operating_cost = 0.0
    remaining = residual_after_storage.copy()

    for name in MERIT_ORDER:
        capacity = firm_subtypes[name]                  # MW
        avail = outage_avail[name]                      # (n_hours,) 0 or 1
        mc = FIRM_SUBTYPES[name]["mc"]                  # $/MWh

        available_capacity = capacity * avail            # (n_hours,)
        dispatched = np.minimum(available_capacity, remaining)
        total_operating_cost += np.sum(dispatched * mc)
        remaining -= dispatched
        remaining = np.maximum(0.0, remaining)

    # --- Step 5: Unserved energy ---
    U_ij = np.sum(remaining)  # MWh

    # --- Performance measure ---
    F_i = compute_fixed_cost(portfolio)
    C_ij = total_operating_cost
    Y_ij = F_i + C_ij + VOLL * U_ij

    return F_i, C_ij, U_ij, Y_ij


if __name__ == "__main__":
    from config import PORTFOLIOS
    from crn import generate_common_inputs

    # Pilot: 1 replication, all 16 portfolios
    inputs = generate_common_inputs(base_seed=42, replication=0)

    print(f"{'Portfolio':<22} {'F_i ($M)':>10} {'C_ij ($M)':>10} "
          f"{'U_ij (MWh)':>12} {'Y_ij ($M)':>10}")
    print("-" * 70)

    for p in PORTFOLIOS:
        F_i, C_ij, U_ij, Y_ij = simulate_replication(p, inputs)
        print(f"{p.label:<22} {F_i/1e6:>10.1f} {C_ij/1e6:>10.1f} "
              f"{U_ij:>12,.0f} {Y_ij/1e6:>10.1f}")
