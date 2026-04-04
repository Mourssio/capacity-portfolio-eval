"""
config.py — Parameters and systematic portfolio generation.

All numerical assumptions are documented and sourced.
Change any parameter here; no other file should contain magic numbers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# =============================================================================
# System Parameters
# =============================================================================

PEAK_DEMAND_MW = 8_000           # Stylized medium-scale system
RESERVE_MARGIN = 0.50            # 50% above peak (accounts for variable renewables)
TOTAL_CAPACITY_MW = int(PEAK_DEMAND_MW * (1 + RESERVE_MARGIN))  # 12,000 MW
LOAD_FACTOR = 0.60               # Average demand ~ 4,800 MW
N_HOURS = 8_760                  # One year, hourly resolution


# =============================================================================
# Technology Cost Parameters (Stylized, informed by NREL ATB)
# =============================================================================

# Annualized capital costs ($/kW-yr)
CAPITAL_COST = {
    "nuclear":      400,
    "flex_firm":    100,
    "mid_merit":     70,   # Note: mid-merit gas has lower capital than flex firm
    "peaker":        70,
    "wind":         130,
    "solar":        100,
    "storage":      150,
}

# Marginal operating costs ($/MWh)
MARGINAL_COST = {
    "nuclear":      10,
    "flex_firm":    35,
    "mid_merit":    50,
    "peaker":       80,
    "wind":          0,
    "solar":         0,
    "storage":       0,   # No fuel cost; round-trip losses handled in dispatch
}

# Forced outage modeling: per-unit, per-day Bernoulli
# Each sub-type is split into N_UNITS identical units.
# This avoids the unrealistic "all-or-nothing" outage of monolithic blocks.
N_UNITS_PER_SUBTYPE = 10

# Merit order: dispatch cheapest first
MERIT_ORDER = ["nuclear", "flex_firm", "mid_merit", "peaker"]


# =============================================================================
# Firm Generator Sub-Type Mix (shares of firm allocation)
# =============================================================================

FIRM_SUBTYPES = {
    "nuclear":    {"share": 0.20, "FOR": 0.05, "mc": MARGINAL_COST["nuclear"]},
    "flex_firm":  {"share": 0.35, "FOR": 0.06, "mc": MARGINAL_COST["flex_firm"]},
    "mid_merit":  {"share": 0.30, "FOR": 0.05, "mc": MARGINAL_COST["mid_merit"]},
    "peaker":     {"share": 0.15, "FOR": 0.08, "mc": MARGINAL_COST["peaker"]},
}

# Renewable sub-type split
RENEWABLE_SPLIT = {"wind": 0.60, "solar": 0.40}


# =============================================================================
# Stochastic Input Parameters
# =============================================================================

DEMAND_NOISE_SIGMA = 0.05        # Default 5%; varied in factorial (2%, 10%)

# Demand profile: D_base(t) = peak * [0.55 + 0.15*sin(seasonal) + 0.10*sin(diurnal)]
DEMAND_PROFILE = {
    "base_fraction": 0.375,
    "seasonal_amplitude": 0.15,
    "diurnal_amplitude": 0.10,
}

# Wind: Beta(2, 5), mean ~ 0.29
WIND_BETA_PARAMS = (2, 5)

# Solar: diurnal envelope * Beta(5, 2) cloud factor
SOLAR_BETA_PARAMS = (5, 2)
SOLAR_DAYLIGHT_HOURS = (6, 20)   # Hours 6-20 have nonzero solar potential


# =============================================================================
# Storage Parameters
# =============================================================================

STORAGE_DURATION_HOURS = 4       # 4-hour Li-ion
STORAGE_RTE = 0.85               # 85% round-trip efficiency
STORAGE_DISCHARGE_THRESHOLD = 0.70  # Discharge when demand > 70th pctile
STORAGE_CHARGE_THRESHOLD = 0.30     # Charge when demand < 30th pctile


# =============================================================================
# Adequacy and Performance Measure
# =============================================================================

VOLL = 10_000                    # Value of lost load ($/MWh)


# =============================================================================
# Kim-Nelson R&S Parameters
# =============================================================================

KN_ALPHA = 0.05                  # P(correct selection) >= 1 - alpha
KN_DELTA = 5_000_000             # Indifference zone ($5M)
KN_N0 = 20                       # Initial replications
KN_MAX_REPS = 500                # Safety cap


# =============================================================================
# Experimental Design Parameters
# =============================================================================

DBG_N_REPS = 200                 # Replications for deterministic benchmark gap
FACTORIAL_N_REPS = 30            # Replications per cell in sensitivity analysis

# 2^3 factorial factor levels
FACTORIAL_LEVELS = {
    "demand_noise":  {"low": 0.02, "high": 0.10},
    "forced_outage": {
        "low":  {"nuclear": 0.01, "flex_firm": 0.01, "mid_merit": 0.01, "peaker": 0.01},
        "high": {"nuclear": 0.05, "flex_firm": 0.06, "mid_merit": 0.05, "peaker": 0.08},
    },
    "renewable_var": {"low": "fixed_mean", "high": "stochastic"},
}


# =============================================================================
# Portfolio Generation: Fixed-Budget Simplex Discretization
# =============================================================================

# Feasibility bounds (fraction of TOTAL_CAPACITY_MW)
PORTFOLIO_BOUNDS = {
    "firm":      (0.20, 0.70),
    "renewable": (0.10, 0.60),
    "storage":   (0.10, 0.30),
}

PORTFOLIO_STEP = 0.10            # 10% discretization


@dataclass
class Portfolio:
    """A candidate capacity portfolio."""
    pid: int                     # Portfolio ID (0-indexed)
    firm_frac: float
    renew_frac: float
    storage_frac: float

    @property
    def firm_mw(self) -> float:
        return self.firm_frac * TOTAL_CAPACITY_MW

    @property
    def renew_mw(self) -> float:
        return self.renew_frac * TOTAL_CAPACITY_MW

    @property
    def storage_mw(self) -> float:
        return self.storage_frac * TOTAL_CAPACITY_MW

    @property
    def storage_mwh(self) -> float:
        return self.storage_mw * STORAGE_DURATION_HOURS

    @property
    def wind_mw(self) -> float:
        return self.renew_mw * RENEWABLE_SPLIT["wind"]

    @property
    def solar_mw(self) -> float:
        return self.renew_mw * RENEWABLE_SPLIT["solar"]

    @property
    def label(self) -> str:
        return (f"P{self.pid:02d} "
                f"({int(self.firm_frac*100)}/"
                f"{int(self.renew_frac*100)}/"
                f"{int(self.storage_frac*100)})")

    def firm_subtype_mw(self) -> Dict[str, float]:
        """MW capacity for each firm sub-type."""
        return {name: info["share"] * self.firm_mw
                for name, info in FIRM_SUBTYPES.items()}

    def annualized_fixed_cost(self) -> float:
        """F_i: annualized capital cost ($), deterministic."""
        cost = 0.0
        # Firm sub-types
        for name, info in FIRM_SUBTYPES.items():
            mw = info["share"] * self.firm_mw
            cost += mw * 1000 * CAPITAL_COST[name]  # $/kW-yr * 1000 kW/MW
        # Renewables
        cost += self.wind_mw * 1000 * CAPITAL_COST["wind"]
        cost += self.solar_mw * 1000 * CAPITAL_COST["solar"]
        # Storage
        cost += self.storage_mw * 1000 * CAPITAL_COST["storage"]
        return cost


def generate_portfolios() -> List[Portfolio]:
    """
    Enumerate all feasible portfolios on the discretized simplex.

    Constraint: firm + renewable + storage = 1.0
    Bounds:     firm in [0.20, 0.70], renewable in [0.10, 0.60], storage in [0.10, 0.30]
    Step:       0.10

    Returns a list of Portfolio objects sorted by (storage, firm, renewable).
    """
    portfolios = []
    pid = 0
    tol = 1e-9  # Floating-point tolerance

    f_lo, f_hi = PORTFOLIO_BOUNDS["firm"]
    r_lo, r_hi = PORTFOLIO_BOUNDS["renewable"]
    s_lo, s_hi = PORTFOLIO_BOUNDS["storage"]

    # Iterate over storage levels first (outer), then firm
    s_vals = np.arange(s_lo, s_hi + tol, PORTFOLIO_STEP)
    f_vals = np.arange(f_lo, f_hi + tol, PORTFOLIO_STEP)

    for s in s_vals:
        for f in f_vals:
            r = round(1.0 - f - s, 4)  # Budget constraint
            if r_lo - tol <= r <= r_hi + tol:
                portfolios.append(Portfolio(
                    pid=pid,
                    firm_frac=round(f, 2),
                    renew_frac=round(r, 2),
                    storage_frac=round(s, 2),
                ))
                pid += 1

    return portfolios


# =============================================================================
# Module-level portfolio list (generated once on import)
# =============================================================================

PORTFOLIOS = generate_portfolios()
N_PORTFOLIOS = len(PORTFOLIOS)


# =============================================================================
# Convenience: print summary on direct execution
# =============================================================================

if __name__ == "__main__":
    print(f"Total capacity budget: {TOTAL_CAPACITY_MW:,} MW")
    print(f"Generated {N_PORTFOLIOS} feasible portfolios:\n")
    print(f"{'ID':<6} {'Label':<20} {'Firm MW':>10} {'Renew MW':>10} "
          f"{'Stor MW':>10} {'F_i ($M)':>12}")
    print("-" * 72)
    for p in PORTFOLIOS:
        fi = p.annualized_fixed_cost() / 1e6
        print(f"{p.pid:<6} {p.label:<20} {p.firm_mw:>10,.0f} "
              f"{p.renew_mw:>10,.0f} {p.storage_mw:>10,.0f} {fi:>12,.1f}")
