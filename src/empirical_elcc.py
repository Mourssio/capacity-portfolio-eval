"""
empirical_elcc.py -- Simulation-based ELCC compared to static planning values.

ELCC (effective load-carrying capability) is the MW of "perfect" firm capacity
that yields the same adequacy improvement as installing MW of technology X.
Conventional ELCC values used in UCAP normalization (wind 18%, solar 35%,
storage-4h 50%) are based on peak-period contribution metrics and do not
capture sustained low-renewable events. The NORTA AR(1) wind model produces
multi-hour wind droughts in which effective renewable capacity contribution
collapses; this experiment measures that collapse quantitatively.

Procedure:
  1. Build a stressed reference system S_0 (firm-only, undersized).
  2. For each technology X in {wind, solar, storage-4h, perfect-firm},
     add dose c = 1000 MW and measure mean EUE over N CRN replications.
  3. Compute ELCC_X^emp = (EUE_0 - EUE_X) / (EUE_0 - EUE_perfect_firm).
  4. Compare to static ELCC values.
"""

import numpy as np
from config import (
    FIRM_SUBTYPES, STORAGE_DURATION_HOURS, STORAGE_RTE, ELCC, MERIT_ORDER, VOLL,
)
from crn import generate_common_inputs, DEMAND_P70, DEMAND_P30
from dispatch import simulate_replication


class RawSystem:
    """Duck-typed system that bypasses Portfolio's fraction-based split,
    so we can inject arbitrary (firm, wind, solar, storage) combinations."""
    def __init__(self, firm_mw, wind_mw=0.0, solar_mw=0.0, storage_mw=0.0):
        self.firm_mw = firm_mw
        self.wind_mw = wind_mw
        self.solar_mw = solar_mw
        self.storage_mw = storage_mw
        self.storage_mwh = storage_mw * STORAGE_DURATION_HOURS

    def firm_subtype_mw(self):
        return {name: info["share"] * self.firm_mw
                for name, info in FIRM_SUBTYPES.items()}

    def annualized_fixed_cost(self):
        return 0.0  # Not used in ELCC computation


def _mean_eue(system, n_reps, for_overrides=None):
    """Average unserved energy over n_reps CRN replications."""
    eues = []
    for rep in range(n_reps):
        inp = generate_common_inputs(
            base_seed=42, replication=rep,
            for_overrides=for_overrides,
        )
        _, _, U_ij, _ = simulate_replication(system, inp)
        eues.append(U_ij)
    return float(np.mean(eues)), float(np.std(eues, ddof=1))


def run_empirical_elcc(baseline_firm_mw=5000.0, dose_mw=1000.0, n_reps=100):
    """
    Compute empirical ELCC for wind, solar, storage, and realistic firm
    against a perfect-firm reference.
    """
    print("=" * 70)
    print("EMPIRICAL ELCC EXPERIMENT")
    print("=" * 70)
    print(f"  Baseline firm capacity: {baseline_firm_mw:,.0f} MW (realistic FOR)")
    print(f"  Technology dose:        {dose_mw:,.0f} MW")
    print(f"  Replications:           {n_reps}")
    print()

    # --- Reference systems ---
    S0 = RawSystem(firm_mw=baseline_firm_mw)
    # Perfect-firm reference: baseline + 1000 MW firm, override FOR=0 on everything
    Sperfect = RawSystem(firm_mw=baseline_firm_mw + dose_mw)
    zero_for = {name: 0.0 for name in FIRM_SUBTYPES}

    # Technology additions
    systems = {
        "baseline":        (S0, None),
        "perfect_firm":    (Sperfect, zero_for),
        "realistic_firm":  (RawSystem(firm_mw=baseline_firm_mw + dose_mw), None),
        "wind":            (RawSystem(firm_mw=baseline_firm_mw, wind_mw=dose_mw), None),
        "solar":           (RawSystem(firm_mw=baseline_firm_mw, solar_mw=dose_mw), None),
        "storage":         (RawSystem(firm_mw=baseline_firm_mw, storage_mw=dose_mw), None),
    }

    # --- Measure mean EUE for each ---
    results = {}
    for name, (sys, fo) in systems.items():
        mean_eue, std_eue = _mean_eue(sys, n_reps, for_overrides=fo)
        results[name] = {"mean_eue": mean_eue, "std_eue": std_eue}
        print(f"  {name:<18} mean EUE = {mean_eue:>14,.0f} MWh  "
              f"(std {std_eue:>12,.0f})")

    # --- Compute ELCCs ---
    eue0 = results["baseline"]["mean_eue"]
    eue_perfect = results["perfect_firm"]["mean_eue"]
    delta_perfect = eue0 - eue_perfect
    print()
    print(f"  Baseline EUE:                {eue0:>14,.0f} MWh")
    print(f"  Perfect-firm reference EUE:  {eue_perfect:>14,.0f} MWh")
    print(f"  Perfect-firm reduction:      {delta_perfect:>14,.0f} MWh "
          f"(per {dose_mw:.0f} MW)")
    print()

    static_elcc = {
        "wind": ELCC["wind"],
        "solar": ELCC["solar"],
        "storage": ELCC["storage"],
        "realistic_firm": sum(info["share"] * ELCC[name]
                               for name, info in FIRM_SUBTYPES.items()),
    }

    print(f"  {'Technology':<18} {'Delta EUE':>14} {'Empirical ELCC':>16} "
          f"{'Static ELCC':>14} {'Ratio emp/static':>18}")
    print(f"  {'-'*82}")
    elccs = {}
    for tech in ("wind", "solar", "storage", "realistic_firm"):
        delta = eue0 - results[tech]["mean_eue"]
        if delta_perfect > 1.0:
            emp = delta / delta_perfect
        else:
            emp = float("nan")
        emp = max(0.0, min(1.0, emp))  # Clip to [0, 1]
        elccs[tech] = emp
        static = static_elcc[tech]
        ratio = emp / static if static > 0 else float("nan")
        print(f"  {tech:<18} {delta:>14,.0f} {emp:>16.3f} {static:>14.3f} "
              f"{ratio:>18.2f}")

    return {
        "eues": results,
        "empirical_elcc": elccs,
        "static_elcc": static_elcc,
        "delta_perfect": delta_perfect,
        "baseline_firm_mw": baseline_firm_mw,
        "dose_mw": dose_mw,
        "n_reps": n_reps,
    }


if __name__ == "__main__":
    out = run_empirical_elcc(baseline_firm_mw=5000.0, dose_mw=1000.0, n_reps=100)
    print()
    print("=" * 70)
    print("Interpretation:")
    for tech in ("wind", "solar", "storage"):
        emp = out["empirical_elcc"][tech]
        static = out["static_elcc"][tech]
        gap = emp - static
        direction = "above" if gap > 0 else "below"
        print(f"  {tech:<10} empirical={emp:.2f}  static={static:.2f}  "
              f"({abs(gap):.2f} {direction} planning value)")
