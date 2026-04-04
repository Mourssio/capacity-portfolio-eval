# Capacity Portfolio Evaluation Under Multi-Dimensional Uncertainty

MIE1613H Stochastic Simulation — Final Project (Winter 2026)

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
├── main.py            # Orchestration and output
└── plots.py           # Visualization
tests/
├── test_dispatch.py   # Simulator validation
└── test_crn.py        # CRN correlation checks
report/                # LaTeX source
figures/               # Generated figures
```

## Usage
```bash
python src/main.py
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
