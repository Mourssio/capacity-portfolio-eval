"""
input_fitting.py — Fit parametric input models to CODERS/IESO empirical data.

Fits:
  1. Demand: seasonal + diurnal profile shape and noise sigma
  2. Wind CF: Beta(a, b) via MLE
  3. Solar CF: diurnal envelope parameters and Beta cloud factor via MLE

Produces diagnostic plots and outputs calibrated parameters for config.py.

Data sources:
  - IESO PUB_Demand (2024): hourly Ontario demand
  - CODERS wind_capacity_factor.csv: hourly wind CFs across Ontario sites
  - CODERS solar_capacity_factor.csv: hourly solar CFs across Ontario sites
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_demand(path_2024, path_2025=None):
    """Load IESO hourly demand data. Use 2024 as primary (full year)."""
    df = pd.read_csv(path_2024)
    demand = df['Ontario Demand'].values.astype(float)
    # 2024 is leap year (8784 hours); truncate to 8760 for consistency
    if len(demand) > 8760:
        demand = demand[:8760]
    return demand


def load_wind_cf(path):
    """Load CODERS wind capacity factors, average across sites."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != 'h']
    # Average across all sites to get aggregate hourly CF
    cf = df[cols].mean(axis=1).values[:8760]
    return cf


def load_solar_cf(path):
    """Load CODERS solar capacity factors, average across sites."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != 'h']
    cf = df[cols].mean(axis=1).values[:8760]
    return cf


# =========================================================================
# 1. Demand Profile Fitting
# =========================================================================

def fit_demand_profile(demand):
    """
    Fit demand to a seasonal + diurnal deterministic profile plus noise.

    Uses 2-harmonic seasonal (annual + semi-annual) to capture Ontario's
    bi-modal pattern (winter heating + summer cooling peaks).
    """
    n = len(demand)
    t = np.arange(n, dtype=float)
    peak = demand.max()

    # Construct regression matrix: intercept + 2 seasonal + 1 diurnal harmonics
    X = np.column_stack([
        np.ones(n),                                          # intercept
        np.sin(2 * np.pi * t / 8760),                       # annual sin
        np.cos(2 * np.pi * t / 8760),                       # annual cos
        np.sin(4 * np.pi * t / 8760),                       # semi-annual sin
        np.cos(4 * np.pi * t / 8760),                       # semi-annual cos
        np.sin(2 * np.pi * (t % 24) / 24),                  # diurnal sin
        np.cos(2 * np.pi * (t % 24) / 24),                  # diurnal cos
    ])

    # OLS fit
    beta = np.linalg.lstsq(X, demand, rcond=None)[0]
    det_profile = X @ beta

    # Noise sigma from relative residuals
    residuals = (demand - det_profile) / det_profile
    sigma_d = residuals.std()

    # R-squared
    ss_res = np.sum((demand - det_profile)**2)
    ss_tot = np.sum((demand - demand.mean())**2)
    r_squared = 1 - ss_res / ss_tot

    # Extract amplitudes for reporting
    A_annual = np.sqrt(beta[1]**2 + beta[2]**2) / peak
    A_semi = np.sqrt(beta[3]**2 + beta[4]**2) / peak
    A_diurnal = np.sqrt(beta[5]**2 + beta[6]**2) / peak

    results = {
        'peak': peak,
        'base_fraction': beta[0] / peak,
        'seasonal_amplitude': A_annual,
        'semi_annual_amplitude': A_semi,
        'diurnal_amplitude': A_diurnal,
        'sigma_d': sigma_d,
        'r_squared': r_squared,
        'load_factor': demand.mean() / peak,
        'det_profile': det_profile,
        'regression_coeffs': beta,
    }

    return results


# =========================================================================
# 2. Wind CF Fitting
# =========================================================================

def fit_wind_beta(wind_cf):
    """
    Fit Beta(a, b) to hourly aggregate wind capacity factors via MLE.
    """
    # Clip to (0, 1) open interval for Beta MLE
    cf = np.clip(wind_cf, 1e-6, 1 - 1e-6)
    a, b, loc, scale = stats.beta.fit(cf, floc=0, fscale=1)

    # KS test
    ks_stat, ks_pval = stats.kstest(cf, 'beta', args=(a, b, 0, 1))

    results = {
        'a': a,
        'b': b,
        'mean': a / (a + b),
        'std': np.sqrt(a * b / ((a + b)**2 * (a + b + 1))),
        'empirical_mean': wind_cf.mean(),
        'empirical_std': wind_cf.std(),
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
    }
    return results


# =========================================================================
# 3. Solar CF Fitting
# =========================================================================

def fit_solar(solar_cf):
    """
    Fit solar CF model: CF(t) = envelope(t) * cloud_factor

    Note: CODERS data appears to be in UTC. Ontario is UTC-5 (EST).
    We shift the hourly pattern to local time before fitting.
    """
    n = len(solar_cf)
    # Shift to local time (UTC-5 for Ontario EST)
    utc_offset = 5
    local_hours = (np.arange(n) - utc_offset) % 24

    # Determine sunrise/sunset from average hourly profile in local time
    hourly_avg_cf = np.array([solar_cf[local_hours == h].mean() for h in range(24)])
    # Use 5% of peak hourly CF as threshold for daylight
    threshold = hourly_avg_cf.max() * 0.05
    daylight_hours_set = np.where(hourly_avg_cf > threshold)[0]
    if len(daylight_hours_set) == 0:
        return {'error': 'No significant solar generation found'}
    sunrise = int(daylight_hours_set[0])
    sunset = int(daylight_hours_set[-1] + 1)

    # Compute envelope using local time hours
    envelope = np.zeros(n)
    daylight_mask = (local_hours >= sunrise) & (local_hours < sunset)
    day_frac = (local_hours[daylight_mask] - sunrise) / (sunset - sunrise)
    envelope[daylight_mask] = np.sin(np.pi * day_frac)

    # Extract cloud factor for daylight hours with nonzero envelope
    valid = (envelope > 0.01) & (solar_cf > 0)
    if valid.sum() < 100:
        return {'error': 'Insufficient daylight data'}

    cloud_factor = solar_cf[valid] / envelope[valid]
    cloud_factor = np.clip(cloud_factor, 1e-6, 1 - 1e-6)

    # Fit Beta to cloud factor
    a, b, loc, scale = stats.beta.fit(cloud_factor, floc=0, fscale=1)
    ks_stat, ks_pval = stats.kstest(cloud_factor, 'beta', args=(a, b, 0, 1))

    results = {
        'sunrise': sunrise,
        'sunset': sunset,
        'cloud_a': a,
        'cloud_b': b,
        'cloud_mean': a / (a + b),
        'empirical_mean_cf': solar_cf.mean(),
        'empirical_daylight_mean': solar_cf[daylight_mask].mean(),
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
    }
    return results


# =========================================================================
# Diagnostic Plots
# =========================================================================

def plot_demand_fit(demand, fit_results):
    """Plot demand profile fit diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series (first 2 weeks)
    ax = axes[0, 0]
    hrs = min(336, len(demand))
    ax.plot(range(hrs), demand[:hrs] / 1000, 'b-', alpha=0.6, label='Empirical', linewidth=0.8)
    ax.plot(range(hrs), fit_results['det_profile'][:hrs] / 1000, 'r-', alpha=0.8,
            label='Fitted profile', linewidth=1)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Demand (GW)')
    ax.set_title('First 2 Weeks: Empirical vs Fitted')
    ax.legend(fontsize=8)

    # Seasonal pattern (daily averages)
    ax = axes[0, 1]
    n_days = len(demand) // 24
    daily_emp = demand[:n_days*24].reshape(n_days, 24).mean(axis=1)
    daily_fit = fit_results['det_profile'][:n_days*24].reshape(n_days, 24).mean(axis=1)
    ax.plot(range(n_days), daily_emp / 1000, 'b-', alpha=0.6, label='Empirical', linewidth=0.8)
    ax.plot(range(n_days), daily_fit / 1000, 'r-', alpha=0.8, label='Fitted', linewidth=1)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Daily Avg Demand (GW)')
    ax.set_title('Seasonal Pattern')
    ax.legend(fontsize=8)

    # Diurnal pattern
    ax = axes[1, 0]
    hourly_emp = np.array([demand[np.arange(len(demand)) % 24 == h].mean() for h in range(24)])
    hourly_fit = np.array([fit_results['det_profile'][np.arange(len(demand)) % 24 == h].mean()
                           for h in range(24)])
    ax.plot(range(24), hourly_emp / 1000, 'bo-', label='Empirical', markersize=4)
    ax.plot(range(24), hourly_fit / 1000, 'rs-', label='Fitted', markersize=4)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Demand (GW)')
    ax.set_title('Diurnal Pattern')
    ax.legend(fontsize=8)

    # Residual histogram
    ax = axes[1, 1]
    residuals = (demand - fit_results['det_profile']) / fit_results['det_profile']
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, 0, fit_results['sigma_d']), 'r-', linewidth=2,
            label=f'N(0, {fit_results["sigma_d"]:.3f})')
    ax.set_xlabel('Relative Residual')
    ax.set_title(f'Noise Distribution (σ = {fit_results["sigma_d"]:.3f})')
    ax.legend(fontsize=8)

    fig.suptitle(f'Demand Profile Fit (R² = {fit_results["r_squared"]:.4f})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'input_demand_fit.png'), dpi=150)
    plt.close(fig)


def plot_wind_fit(wind_cf, fit_results):
    """Plot wind CF Beta fit diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    a, b = fit_results['a'], fit_results['b']

    # Histogram + fitted PDF
    ax = axes[0]
    ax.hist(wind_cf, bins=60, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
    x = np.linspace(0.01, 0.99, 200)
    ax.plot(x, stats.beta.pdf(x, a, b), 'r-', linewidth=2,
            label=f'Beta({a:.2f}, {b:.2f})')
    ax.set_xlabel('Capacity Factor')
    ax.set_ylabel('Density')
    ax.set_title('Wind CF: Empirical vs Fitted Beta')
    ax.legend(fontsize=9)

    # Q-Q plot
    ax = axes[1]
    theoretical = stats.beta.ppf(np.linspace(0.01, 0.99, 200), a, b)
    empirical = np.quantile(wind_cf, np.linspace(0.01, 0.99, 200))
    ax.plot(theoretical, empirical, 'bo', markersize=2)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(f'Q-Q Plot (KS p = {fit_results["ks_pval"]:.4f})')
    ax.set_aspect('equal')

    fig.suptitle(f'Wind CF Fit: Mean = {fit_results["empirical_mean"]:.3f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'input_wind_fit.png'), dpi=150)
    plt.close(fig)


def plot_solar_fit(solar_cf, fit_results):
    """Plot solar CF fit diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Average diurnal profile in local time (UTC-5)
    ax = axes[0]
    local_hours = (np.arange(len(solar_cf)) - 5) % 24
    hourly_avg = np.array([solar_cf[local_hours == h].mean() for h in range(24)])
    ax.bar(range(24), hourly_avg, alpha=0.7, color='orange', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Hour of Day (EST)')
    ax.set_ylabel('Average CF')
    ax.set_title(f'Solar Diurnal Profile (sunrise={fit_results["sunrise"]}, '
                 f'sunset={fit_results["sunset"]})')

    # Cloud factor histogram
    ax = axes[1]
    sunrise, sunset = fit_results['sunrise'], fit_results['sunset']
    envelope = np.zeros(len(solar_cf))
    daylight = (local_hours >= sunrise) & (local_hours < sunset)
    day_frac = (local_hours[daylight] - sunrise) / (sunset - sunrise)
    envelope[daylight] = np.sin(np.pi * day_frac)
    valid = (envelope > 0.01) & (solar_cf > 0)
    cloud = np.clip(solar_cf[valid] / envelope[valid], 0, 1)

    a, b = fit_results['cloud_a'], fit_results['cloud_b']
    ax.hist(cloud, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
    x = np.linspace(0.01, 0.99, 200)
    ax.plot(x, stats.beta.pdf(x, a, b), 'r-', linewidth=2,
            label=f'Beta({a:.2f}, {b:.2f})')
    ax.set_xlabel('Cloud Factor')
    ax.set_ylabel('Density')
    ax.set_title(f'Solar Cloud Factor (KS p = {fit_results["ks_pval"]:.4f})')
    ax.legend(fontsize=9)

    fig.suptitle(f'Solar CF Fit: Overall Mean = {fit_results["empirical_mean_cf"]:.3f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'input_solar_fit.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def run_fitting(demand_path, wind_path, solar_path):
    """Run all fits and produce outputs."""
    print("=" * 60)
    print("INPUT MODEL FITTING (CODERS / IESO Data)")
    print("=" * 60)

    # Load data
    demand = load_demand(demand_path)
    wind_cf = load_wind_cf(wind_path)
    solar_cf = load_solar_cf(solar_path)

    # 1. Demand
    print("\n1. DEMAND PROFILE")
    d_fit = fit_demand_profile(demand)
    print(f"   Peak: {d_fit['peak']:,.0f} MW")
    print(f"   Load factor: {d_fit['load_factor']:.3f}")
    print(f"   Base fraction: {d_fit['base_fraction']:.4f}")
    print(f"   Annual seasonal amplitude: {d_fit['seasonal_amplitude']:.4f}")
    print(f"   Semi-annual amplitude: {d_fit['semi_annual_amplitude']:.4f}")
    print(f"   Diurnal amplitude: {d_fit['diurnal_amplitude']:.4f}")
    print(f"   Noise sigma: {d_fit['sigma_d']:.4f}")
    print(f"   R-squared: {d_fit['r_squared']:.4f}")
    plot_demand_fit(demand, d_fit)

    # 2. Wind
    print("\n2. WIND CAPACITY FACTOR")
    w_fit = fit_wind_beta(wind_cf)
    print(f"   Fitted Beta({w_fit['a']:.3f}, {w_fit['b']:.3f})")
    print(f"   Fitted mean: {w_fit['mean']:.3f} (empirical: {w_fit['empirical_mean']:.3f})")
    print(f"   KS statistic: {w_fit['ks_stat']:.4f}, p-value: {w_fit['ks_pval']:.4f}")
    plot_wind_fit(wind_cf, w_fit)

    # 3. Solar
    print("\n3. SOLAR CAPACITY FACTOR")
    s_fit = fit_solar(solar_cf)
    print(f"   Sunrise hour: {s_fit['sunrise']}, Sunset hour: {s_fit['sunset']}")
    print(f"   Cloud factor Beta({s_fit['cloud_a']:.3f}, {s_fit['cloud_b']:.3f})")
    print(f"   Cloud factor mean: {s_fit['cloud_mean']:.3f}")
    print(f"   Overall mean CF: {s_fit['empirical_mean_cf']:.3f}")
    print(f"   KS statistic: {s_fit['ks_stat']:.4f}, p-value: {s_fit['ks_pval']:.4f}")
    plot_solar_fit(solar_cf, s_fit)

    # Summary: parameters for config.py
    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS FOR config.py")
    print("=" * 60)
    print(f"""
DEMAND_NOISE_SIGMA = {d_fit['sigma_d']:.4f}
# Demand profile fitted via OLS with 2-harmonic seasonal + diurnal.
# Load factor = {d_fit['load_factor']:.3f}, R² = {d_fit['r_squared']:.4f}
# Annual seasonal amplitude: {d_fit['seasonal_amplitude']:.4f}
# Semi-annual amplitude: {d_fit['semi_annual_amplitude']:.4f}
# Diurnal amplitude: {d_fit['diurnal_amplitude']:.4f}

WIND_BETA_PARAMS = ({w_fit['a']:.3f}, {w_fit['b']:.3f})  # mean = {w_fit['mean']:.3f}

SOLAR_BETA_PARAMS = ({s_fit['cloud_a']:.3f}, {s_fit['cloud_b']:.3f})  # cloud factor mean = {s_fit['cloud_mean']:.3f}
SOLAR_DAYLIGHT_HOURS = ({s_fit['sunrise']}, {s_fit['sunset']})
""")

    print("Diagnostic plots saved to figures/")

    return {
        'demand': d_fit,
        'wind': w_fit,
        'solar': s_fit,
    }


if __name__ == "__main__":
    INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'inputs')
    run_fitting(
        demand_path=os.path.join(INPUT_DIR, 'PUB_Demand_2024.csv'),
        wind_path=os.path.join(INPUT_DIR, 'wind_capacity_factor.csv'),
        solar_path=os.path.join(INPUT_DIR, 'solar_capacity_factor.csv'),
    )
