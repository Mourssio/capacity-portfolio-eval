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
# Technology Cost Parameters
# =============================================================================
# Annualized capital costs ($/kW-yr) rounded to two significant figures.
#
# Sources (values are plausible midpoints rather than point estimates):
#   Nuclear:     IESO Annual Planning Outlook 2023 (refurbished Ontario
#                CANDU units, levelized to $/kW-yr over a 30-yr period)
#   Flex firm:   IESO APO 2023 (combined hydro / flexible gas fleet)
#   Mid-merit:   NREL ATB 2023, advanced combined-cycle gas, CAPEX
#                $1,100/kW annualized at 7%, 30-yr
#   Peaker:      NREL ATB 2023, advanced aeroderivative simple cycle,
#                CAPEX $900/kW annualized at 7%, 30-yr
#   Wind:        NREL ATB 2023, land-based Class 4, CAPEX $1,300/kW
#                annualized at 7%, 25-yr
#   Solar:       NREL ATB 2023, utility-scale PV, CAPEX $1,000/kW
#                annualized at 7%, 25-yr
#   Storage:     NREL ATB 2023, 4-hr Li-ion battery, power-component
#                CAPEX annualized at 7%, 15-yr
#
# A +/-30% one-at-a-time sensitivity on every cost parameter leaves the
# KN-selected portfolio unchanged (see Appendix~\ref{app:capsens}).

CAPITAL_COST = {
    "nuclear":      400,   # IESO APO 2023
    "flex_firm":    100,   # IESO APO 2023
    "mid_merit":     70,   # NREL ATB 2023 (advanced CCGT)
    "peaker":        70,   # NREL ATB 2023 (advanced aero SCGT)
    "wind":         130,   # NREL ATB 2023 (land-based Class 4)
    "solar":        100,   # NREL ATB 2023 (utility-scale PV)
    "storage":      150,   # NREL ATB 2023 (4-hr Li-ion, power comp.)
}

# Marginal operating costs ($/MWh)
MARGINAL_COST = {
    "nuclear":      22,   # Reflects Ontario nuclear refurbishment contract costs (~$20-25/MWh)
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

DEMAND_NOISE_SIGMA = 0.09         # Calibrated from IESO 2024 Ontario demand (R²=0.59)

# Demand profile: D_base(t) = peak * [0.55 + 0.15*sin(seasonal) + 0.10*sin(diurnal)]
# Demand profile: 2-harmonic seasonal + diurnal, calibrated to IESO 2024 Ontario data.
# Fitted via OLS: D(t) = b0 + b1*sin(annual) + b2*cos(annual) + b3*sin(semi)
#                        + b4*cos(semi) + b5*sin(diurnal) + b6*cos(diurnal)
# Then normalized so max(profile) = PEAK_DEMAND_MW.
# Source: IESO PUB_Demand_2024.csv (Ontario Demand column)
DEMAND_PROFILE = {
    "type": "2harmonic_ols",
    "coeffs_normalized": [
        0.670249,   # b0: intercept / peak (base fraction)
        -0.015036,  # b1: sin(2*pi*t/8760) / peak
        0.010952,   # b2: cos(2*pi*t/8760) / peak
        0.035695,   # b3: sin(4*pi*t/8760) / peak
        0.048234,   # b4: cos(4*pi*t/8760) / peak
        -0.058102,  # b5: sin(2*pi*(t%24)/24) / peak
        -0.058145,  # b6: cos(2*pi*(t%24)/24) / peak
    ],
}

# Wind: Beta(6.734, 15.694), calibrated from CODERS Ontario wind CF profiles
# Mean CF = 0.300, std = 0.094 (aggregated across 2,984 Ontario sites)
WIND_BETA_PARAMS = (6.734, 15.694)
WIND_AR1_RHO = 0.9945             # Lag-1 autocorrelation, calibrated from CODERS

# Solar: diurnal envelope * Beta(1.772, 6.910) cloud factor
# Calibrated from CODERS Ontario solar CF profiles (2,984 sites)
# Cloud factor mean = 0.204
SOLAR_BETA_PARAMS = (1.772, 6.910)
SOLAR_DAYLIGHT_HOURS = (7, 21)   # Hours 7-21 have nonzero solar (calibrated from CODERS)
SOLAR_AR1_RHO = 0.928             # Cloud factor lag-1 autocorrelation, calibrated from CODERS


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
KN_DELTA = 50_000_000            # Indifference zone ($50M, proportionate to Y ~ $4B)
KN_N0 = 30                       # Initial replications (increased for variance estimation)
KN_MAX_REPS = 1000               # Safety cap


# =============================================================================
# Experimental Design Parameters
# =============================================================================

DBG_N_REPS = 200                 # Replications for deterministic benchmark gap
FACTORIAL_N_REPS = 30            # Replications per cell in sensitivity analysis

# 2^3 factorial factor levels
FACTORIAL_LEVELS = {
    "demand_noise":  {"low": 0.03, "high": 0.15},  # Baseline calibrated at 0.09
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
    total_capacity_mw: float = None  # Override; uses TOTAL_CAPACITY_MW if None

    @property
    def _total_cap(self) -> float:
        return self.total_capacity_mw if self.total_capacity_mw is not None else TOTAL_CAPACITY_MW

    @property
    def firm_mw(self) -> float:
        return self.firm_frac * self._total_cap

    @property
    def renew_mw(self) -> float:
        return self.renew_frac * self._total_cap

    @property
    def storage_mw(self) -> float:
        return self.storage_frac * self._total_cap

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

    def scaled_to(self, total_mw: float) -> 'Portfolio':
        """Return a copy of this portfolio scaled to a different total capacity."""
        return Portfolio(
            pid=self.pid,
            firm_frac=self.firm_frac,
            renew_frac=self.renew_frac,
            storage_frac=self.storage_frac,
            total_capacity_mw=total_mw,
        )

    def weighted_elcc(self) -> float:
        """Compute portfolio-level weighted ELCC (fraction of installed that is UCAP)."""
        from config import ELCC, RENEWABLE_SPLIT
        firm_elcc = sum(
            info["share"] * ELCC[name]
            for name, info in FIRM_SUBTYPES.items()
        )
        renew_elcc = (
            RENEWABLE_SPLIT["wind"] * ELCC["wind"]
            + RENEWABLE_SPLIT["solar"] * ELCC["solar"]
        )
        return (self.firm_frac * firm_elcc
                + self.renew_frac * renew_elcc
                + self.storage_frac * ELCC["storage"])

    def ucap_mw(self) -> float:
        """Effective capacity (UCAP) at current total installed capacity."""
        return self._total_cap * self.weighted_elcc()

    def scaled_to_ucap(self, target_ucap_mw: float) -> 'Portfolio':
        """Return a copy scaled so that UCAP equals the target."""
        required_mw = target_ucap_mw / self.weighted_elcc()
        return self.scaled_to(required_mw)


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
# Effective Load Carrying Capability (ELCC)
# =============================================================================
# Used for UCAP-normalized portfolio comparison.
# Fraction of installed capacity contributing to adequacy at peak.
# Sources: IESO capacity auction parameters, NERC planning conventions.
ELCC = {
    "nuclear":      0.94,    # High availability baseload
    "flex_firm":    0.92,    # Gas/hydro blend, slight derates
    "mid_merit":    0.95,    # CCGT, high availability
    "peaker":       0.92,    # Simple cycle, cycling stress
    "wind":         0.18,    # Ontario wind ELCC (IESO capacity auction range)
    "solar":        0.35,    # Summer peak contribution
    "storage":      0.50,    # 4h Li-ion, duration-limited at peak
}


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
