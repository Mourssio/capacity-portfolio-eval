# Capacity Portfolio Evaluation Under Multi-Dimensional Uncertainty

MIE1613H Stochastic Simulation — Final Project (Winter 2026)

## Overview
Chronological Monte Carlo simulation of a stylized one-zone electricity system
over 8,760 hours, used to compare 16 capacity portfolios under demand,
renewable, and forced-outage uncertainty. The main experiment runs at a fixed
12 GW capacity budget and uses Kim–Nelson ranking and selection with Common
Random Numbers. Input distributions are calibrated from IESO hourly demand and
the CODERS dataset.

## Headline result
The value of uncertainty-aware evaluation in this model is scenario-dependent:

- **Within the 12 GW budget and the original 16-portfolio simplex**, stochastic
  evaluation selects P04 (70/20/10) while mean-input evaluation selects
  P03 (60/30/10). The Deterministic Benchmark Gap is **\$3,968M per year**.
- **At 15.8 GW and 18 GW**, deterministic and stochastic recommendations
  converge on P03 within the re-evaluated top-4 set, and the DBG closes within
  the indifference zone.
- **A boundary probe** shows that relaxing the 70% firm-share cap exposes a
  cheaper 12 GW mix: P_b3 (85/10/05) beats P04 by \$746M.
- **An empirical ELCC experiment** finds that static ELCC values overstate
  solar and storage adequacy contribution on a stressed baseline (wind 0.25
  vs 0.18, solar 0.07 vs 0.35, storage 0.13 vs 0.50); the storage result is
  partly entangled with the rule-based dispatch model.

P04 is therefore the in-budget winner within the original constrained design,
not an absolute optimum. The full regime-dependent finding and its scope
limits are discussed in `report/main_5.pdf`.

## Project Structure
```
src/
├── config.py              # Parameters, portfolio generation, capital costs
├── crn.py                 # Common Random Numbers (SeedSequence), NORTA AR(1)
├── dispatch.py            # Hourly merit-order dispatch simulator
├── kn.py                  # Kim–Nelson sequential elimination (R&S)
├── ocba.py                # OCBA consistency check against KN
├── experiment.py          # DBG, factorial, Bonferroni CIs, VOLL sensitivity
├── enhancements.py        # CRN efficiency, cost decomposition, RHW
├── adequacy.py            # Per-portfolio stochastic adequacy search
├── ucap_comparison.py     # UCAP-normalized portfolio comparison
├── empirical_elcc.py      # Empirical ELCC for wind, solar, storage
├── capital_sensitivity.py # ±30% capital-cost sensitivity (post-processing)
├── deeper_sensitivity.py  # Boundary probe and capacity sweep (structural)
├── verification.py        # Internal verification suite (5 checks)
├── input_fitting.py       # Beta/harmonic fits to IESO and CODERS data
├── extension_plots.py     # Figures for capital sens, ELCC, OCBA
├── structural_plots.py    # Figures for boundary probe and capacity sweep
├── plots.py               # Core pipeline figures
└── main.py                # Pipeline orchestration (10 stages)
report/                    # LaTeX source, compiled report, bibliography
figures/                   # Generated PNG figures referenced by the report
inputs/                    # IESO demand and CODERS capacity-factor inputs
tests/                     # Unit tests for dispatch, CRN, KN primitives
requirements.txt           # Python dependencies (NumPy, SciPy, matplotlib)
```

## Usage

### Core pipeline
```bash
pip install -r requirements.txt
python src/main.py
```
Runs KN selection, extended top-4 evaluation, DBG, factorial, VOLL sensitivity,
CRN enhancement analysis, adequacy search, and UCAP-normalized comparison.
Runtime: roughly 5 minutes on a consumer laptop.

### Methodological extensions
Each extension is a separately runnable module:
```bash
python src/verification.py       # Internal verification suite
python src/capital_sensitivity.py # ±30% capital-cost sensitivity
python src/empirical_elcc.py     # Empirical ELCC experiment
python src/ocba.py               # OCBA consistency check vs KN
python src/deeper_sensitivity.py # Boundary probe + capacity sweep
python src/extension_plots.py    # Regenerate extension figures
python src/structural_plots.py   # Regenerate structural-sensitivity figures
```

### Reproducibility
- Base seed for the core pipeline: 42.
- Base seeds for the OCBA macro-replications: 100–104.
- All figures in `report/figures/` are regenerated from the source data with
  no external downloads required.

## Methods used
- **Common Random Numbers (CRN)** for variance reduction; empirical reduction
  2× to 35× on pairwise differences.
- **Kim–Nelson sequential elimination** at α = 0.05, δ = \$50M, n₀ = 30.
- **OCBA (Chen et al. 2000)** as an independent consistency check on KN.
- **Bonferroni-corrected pairwise 95% CIs** on the top-4 evaluation.
- **NORTA AR(1)** wind and solar models calibrated from CODERS
  (ρ_wind = 0.995, ρ_solar = 0.928).
- **Deterministic Benchmark Gap (DBG)** as a scenario-specific comparison
  within the 12 GW budget.
- **2³ factorial** on demand noise, forced outages, and renewable variability
  (conditional on the top-4).
- **Stochastic adequacy search** via bisection on E[EUE(C)] ≤ 10 MWh/year.
- **Capacity sweep and boundary probe** to test regime dependence.
- **Internal verification suite**: deterministic limit, input mean recovery,
  outage-rate recovery, monotonicity, and CRN variance reduction.

## Minimal tests
```bash
python -m unittest discover -s tests -v
```

## References
- Park, H. and Baldick, R. (2016). *Multi-year stochastic generation capacity
  expansion planning under environmental energy policy.* Applied Energy 183.
- Kim, S.-H. and Nelson, B. L. (2001). *A fully sequential procedure for
  indifference-zone selection in simulation.* ACM TOMACS 11(3).
- Chen, C.-H., Lin, J., Yücesan, E., and Chick, S. E. (2000). *Simulation
  budget allocation for further enhancing the efficiency of ordinal
  optimization.* Discrete Event Dynamic Systems 10(3).
- IESO (2025). *2025 Annual Planning Outlook.*
- NREL (2023). *Annual Technology Baseline.*
- CODERS (2025). *Canadian Open-source Database for Energy Research and
  Systems-Modelling.*

Full bibliography in `report/references.bib`.

## Generative AI disclosure
Generative AI (Claude, Anthropic) was used for code development, debugging,
statistical methodology review, and drafting support. All modelling decisions,
parameter choices, experimental design, interpretation, and final writing are
the author's own.
