"""
crn.py — Common Random Numbers via NumPy SeedSequence.

Pre-generates all stochastic inputs for a given replication BEFORE
any portfolio is simulated. This ensures all portfolios face identical
demand, weather, and outage realizations, inducing positive covariance
in pairwise differences and reducing Var(Y_i - Y_l).

Key design rule: "same random number for the same purpose" across portfolios.
"""

import numpy as np
from numpy.random import SeedSequence, default_rng
from typing import Dict
from config import (
    N_HOURS, DEMAND_NOISE_SIGMA, DEMAND_PROFILE, PEAK_DEMAND_MW,
    WIND_BETA_PARAMS, SOLAR_BETA_PARAMS, SOLAR_DAYLIGHT_HOURS,
    WIND_AR1_RHO, SOLAR_AR1_RHO,
    FIRM_SUBTYPES, MERIT_ORDER, N_UNITS_PER_SUBTYPE,
)
from scipy import stats as sp_stats


def _norta_ar1(rng, n, rho, beta_a, beta_b):
    """
    Generate temporally correlated Beta-distributed random variates
    using NORTA (Normal-to-Anything) with AR(1) dependence.

    Process:
      1. Generate AR(1) standard normals: Z(t) = rho*Z(t-1) + sqrt(1-rho^2)*eps(t)
      2. Transform to Uniform via Phi(Z(t))
      3. Transform to Beta via inverse CDF

    This preserves the Beta marginal distribution while inducing
    temporal autocorrelation calibrated from empirical data.
    """
    # AR(1) in normal space
    innovations = rng.normal(0, 1, size=n)
    z = np.zeros(n)
    z[0] = innovations[0]
    scale = np.sqrt(1 - rho**2)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + scale * innovations[t]

    # Normal -> Uniform -> Beta
    u = sp_stats.norm.cdf(z)
    # Clip to avoid exact 0 or 1 (Beta inverse CDF boundary issues)
    u = np.clip(u, 1e-8, 1 - 1e-8)
    x = sp_stats.beta.ppf(u, beta_a, beta_b)
    return x


N_FIRM_SUBTYPES = len(FIRM_SUBTYPES)


def generate_base_demand(n_hours: int = N_HOURS) -> np.ndarray:
    """
    Deterministic base demand profile (MW), calibrated from IESO 2024 Ontario data.

    Uses a 2-harmonic seasonal (annual + semi-annual) plus diurnal model
    fitted via OLS to empirical hourly demand. The normalized coefficients
    are scaled so max(profile) = PEAK_DEMAND_MW.

    Returns shape (n_hours,).
    """
    t = np.arange(n_hours, dtype=np.float64)
    c = DEMAND_PROFILE["coeffs_normalized"]

    raw = (c[0]
           + c[1] * np.sin(2 * np.pi * t / 8760)
           + c[2] * np.cos(2 * np.pi * t / 8760)
           + c[3] * np.sin(4 * np.pi * t / 8760)
           + c[4] * np.cos(4 * np.pi * t / 8760)
           + c[5] * np.sin(2 * np.pi * (t % 24) / 24)
           + c[6] * np.cos(2 * np.pi * (t % 24) / 24))

    # Normalize so max = PEAK_DEMAND_MW
    d_base = PEAK_DEMAND_MW * (raw / raw.max())
    return d_base


# Pre-compute once (deterministic, same for all replications)
BASE_DEMAND = generate_base_demand()

# Pre-compute demand percentile thresholds for storage dispatch
DEMAND_P70 = np.percentile(BASE_DEMAND, 70)
DEMAND_P30 = np.percentile(BASE_DEMAND, 30)


def generate_solar_envelope(n_hours: int = N_HOURS) -> np.ndarray:
    """
    Deterministic solar diurnal envelope: sinusoidal from sunrise to sunset,
    zero at night.

    Returns shape (n_hours,) with values in [0, 1].
    """
    t_hour = np.arange(n_hours) % 24
    sunrise, sunset = SOLAR_DAYLIGHT_HOURS
    envelope = np.zeros(n_hours)
    mask = (t_hour >= sunrise) & (t_hour < sunset)
    # Normalize to [0, pi] over daylight hours
    day_frac = (t_hour[mask] - sunrise) / (sunset - sunrise)
    envelope[mask] = np.sin(np.pi * day_frac)
    return envelope


SOLAR_ENVELOPE = generate_solar_envelope()


def generate_common_inputs(
    base_seed: int,
    replication: int,
    sigma_d: float = DEMAND_NOISE_SIGMA,
    for_overrides: Dict[str, float] = None,
    renewable_mode: str = "stochastic",
    n_hours: int = N_HOURS,
) -> Dict[str, np.ndarray]:
    """
    Generate all stochastic inputs for one replication.

    Parameters
    ----------
    base_seed : int
        Root seed for reproducibility.
    replication : int
        Replication index (0-based).
    sigma_d : float
        Demand noise standard deviation (fraction).
    for_overrides : dict or None
        Override forced outage rates, e.g. {"nuclear": 0.01, ...}.
        If None, uses default FORs from FIRM_SUBTYPES.
    renewable_mode : str
        "stochastic" (default) or "fixed_mean" for factorial low level.
    n_hours : int
        Number of hours to simulate.

    Returns
    -------
    dict with keys:
        'demand'         : (n_hours,) realized demand in MW
        'wind_cf'        : (n_hours,) wind capacity factors
        'solar_cf'       : (n_hours,) solar capacity factors (envelope * cloud)
        'outage_avail'   : dict[str, (n_hours,)] availability arrays per sub-type
    """
    # Spawn independent streams from (base_seed, replication)
    ss = SeedSequence([base_seed, replication])
    rng_demand, rng_outage, rng_renew = [default_rng(s) for s in ss.spawn(3)]

    # --- Demand ---
    noise = rng_demand.normal(0, 1, size=n_hours)
    demand = BASE_DEMAND * (1.0 + sigma_d * noise)
    demand = np.maximum(demand, 0.0)  # Floor at zero

    # --- Renewables (NORTA AR(1) for temporal autocorrelation) ---
    if renewable_mode == "stochastic":
        # Wind: correlated Beta via NORTA AR(1)
        wind_cf = _norta_ar1(
            rng_renew, n_hours,
            rho=WIND_AR1_RHO,
            beta_a=WIND_BETA_PARAMS[0],
            beta_b=WIND_BETA_PARAMS[1],
        )
        # Solar: correlated cloud factor via NORTA AR(1), then apply envelope
        # Use a separate stream for solar (spawn from rng_renew)
        rng_solar = default_rng(SeedSequence(rng_renew.integers(2**63)))
        solar_cloud = _norta_ar1(
            rng_solar, n_hours,
            rho=SOLAR_AR1_RHO,
            beta_a=SOLAR_BETA_PARAMS[0],
            beta_b=SOLAR_BETA_PARAMS[1],
        )
        solar_cf = SOLAR_ENVELOPE * solar_cloud
    else:
        # Fixed at annual mean (factorial low level)
        wind_mean = WIND_BETA_PARAMS[0] / sum(WIND_BETA_PARAMS)
        solar_mean = SOLAR_BETA_PARAMS[0] / sum(SOLAR_BETA_PARAMS)
        wind_cf = np.full(n_hours, wind_mean)
        solar_cf = SOLAR_ENVELOPE * solar_mean

    # --- Forced outages (per sub-type, per unit, per day, Bernoulli) ---
    # Shape: (N_FIRM_SUBTYPES, N_UNITS_PER_SUBTYPE, 365)
    # Each unit within a sub-type gets an independent outage draw.
    # Portfolios use the same draws but scale unit MW differently.
    outage_draws = rng_outage.random(
        (N_FIRM_SUBTYPES, N_UNITS_PER_SUBTYPE, 365)
    )

    outage_avail = {}
    for idx, name in enumerate(MERIT_ORDER):
        if for_overrides is not None:
            for_rate = for_overrides[name]
        else:
            for_rate = FIRM_SUBTYPES[name]["FOR"]

        # Per-unit, per-day availability: True if draw >= FOR
        unit_daily_avail = (outage_draws[idx] >= for_rate)  # (N_UNITS, 365)
        # Fraction of units available each day
        frac_daily_avail = unit_daily_avail.mean(axis=0)    # (365,)
        # Expand to hourly
        frac_hourly_avail = np.repeat(frac_daily_avail, 24)[:n_hours]
        outage_avail[name] = frac_hourly_avail

    return {
        "demand": demand,
        "wind_cf": wind_cf,
        "solar_cf": solar_cf,
        "outage_avail": outage_avail,
    }


if __name__ == "__main__":
    # Quick sanity check
    inputs = generate_common_inputs(base_seed=42, replication=0)
    print(f"Demand — mean: {inputs['demand'].mean():,.0f} MW, "
          f"peak: {inputs['demand'].max():,.0f} MW")
    print(f"Wind CF — mean: {inputs['wind_cf'].mean():.3f}")
    print(f"Solar CF — mean: {inputs['solar_cf'].mean():.3f}")
    for name, avail_frac in inputs["outage_avail"].items():
        # Convert hourly fractions back to daily for reporting
        daily = avail_frac.reshape(365, 24)[:, 0]
        mean_avail = daily.mean()
        print(f"{name} — mean availability: {mean_avail:.3f} "
              f"(expected: {1 - FIRM_SUBTYPES[name]['FOR']:.2f})")
