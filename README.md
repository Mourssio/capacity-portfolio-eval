# Capacity Portfolio Evaluation Under Multi-Dimensional Uncertainty

MIE1613H Stochastic Simulation - Final Project (Winter 2026)

## Overview
Monte Carlo simulation and Kim-Nelson ranking & selection to compare
systematically generated capacity portfolios under uncertainty in demand,
renewable availability, and firm-resource forced outages.

## Project Structure
```
src/
├── config.py          # Parameters, portfolio generation
├── crn.py             # Common Random Numbers (SeedSequence)
├── dispatch.py        # Hourly merit-order dispatch simulator
├── kn.py              # Kim-Nelson sequential elimination
├── experiment.py      # DBG, factorial, confidence intervals
├── enhancements.py    # CRN efficiency, decomposition, RHW
├── adequacy.py        # Stochastic adequacy search
├── ucap_comparison.py # UCAP-normalized comparison
├── input_fitting.py   # Input model fitting and diagnostics
├── main.py            # Orchestration and output
└── plots.py           # Visualization
report/                # LaTeX source
figures/               # Generated figures
inputs/                # Demand and renewable empirical inputs
requirements.txt       # Python dependencies
```

## Usage
```bash
pip install -r requirements.txt
python src/main.py
```

## Reproducibility
- Full pipeline runtime is roughly 3-6 minutes on a typical laptop.
- Running `python src/main.py` reproduces the core results and regenerates figures in `figures/`.
- Input datasets are included in `inputs/` and no external downloads are required.

## Quick Sanity Check
- The run should end with `COMPLETE - Total runtime: ...`.
- A typical best portfolio is `P04 (70/20/10)` under the current calibration.
- Key figures should be created/updated in `figures/` (e.g., `fig2_kn_elimination.png`, `fig6_dbg.png`, `fig12_ucap_comparison.png`).

## Minimal Tests
```bash
python -m unittest discover -s tests -v
```

## Methods
- Common Random Numbers (CRN) for variance reduction
- Kim-Nelson sequential elimination (R&S)
- Bonferroni-corrected pairwise confidence intervals
- Deterministic Benchmark Gap (DBG)
- Sensitivity analysis on top-k survivors

## References
- Park & Baldick (2016), Applied Energy
- Kim & Nelson (2001), Management Science
