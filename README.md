# The Value Function Semi-Algebraic Set in POMDPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code for the paper **"The Value Function Semi-Algebraic Set in POMDPs"** (Anderson & Montúfar, 2026).

## Overview

This repository contains the experiments and figures from the paper. The central result is that the set of feasible value functions for a POMDP under memoryless stochastic policies admits an explicit semi-algebraic description — a finite system of polynomial equations and inequalities — extending classical polyhedral results for fully observable MDPs to the partially observable setting.

The experiments illustrate two consequences of this geometric structure:
1. The feasible value set has curved (nonlinear) boundary components, captured exactly by finitely many polynomial constraints.
2. Partial observability induces genuinely non-convex optimization landscapes with multiple isolated local optima under policy gradient, unlike the fully observable case.

## Components by paper section

### Figures

| Figure | Paper reference | Source |
|--------|----------------|--------|
| `fo_po_mdp_policy_region.png` (Fig. 1) | Section 2 | Schematic / hand-drawn — no script |
| `linear_inequalities_2.png` (Fig. 2) | Theorem 3.1 illustration | `pomdp_linear_nonlinear_inequalities.py` |
| `linear_nonlinear_ineqs.png` (Fig. 3) | Theorem 3.3 illustration | `pomdp_linear_nonlinear_inequalities.py` |
| `pomdp_initial_state_ablation.png` (Fig. 4) | ρ-dependence visualization | `initial_distribution_analysis.py` (saves as `initial_distribution_analysis.png`) |

### Experiments (Appendix B.2)

The MATLAB scripts in `matlab/` are the canonical implementations that produced the tables in the paper. The Python `local_optima_experiments.py` is an independent reference port.

| Experiment | Paper reference | Canonical script | Reference Python port |
|------------|----------------|------------------|----------------------|
| **A** — Spread of policy-gradient outcomes (partial vs. full observability) | Table 1 | `matlab/pomdp_localopt/run_pomdp_localopt_experiment.m` | `local_optima_experiments.py` |
| **B** — Counting distinct local optima | Table 2 | `matlab/pomdp_localopt/run_pomdp_localopt_experiment.m` | `local_optima_experiments.py` (with adjusted (S, A, O) grid) |
| **C** — Finite-memory policies via observation enrichment | Tables 3a/3b | `matlab/pomdp_memory_enhancement/run_pomdp_memory_enhancement_experiment.m` | — |

The MATLAB outputs are checked in alongside the scripts:
- `matlab/pomdp_localopt/summary_table.md` matches Table 1 exactly.
- `matlab/pomdp_memory_enhancement/summary_table.md` matches Tables 3a/3b exactly.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── pomdp_linear_nonlinear_inequalities.py   # Figs 2, 3
├── initial_distribution_analysis.py          # Fig 4
├── initial_distribution_analysis_multi.py    # Fig 4 variants across configurations
├── pomdp_optim_dynamics.py                   # Supporting: optimization dynamics + POMDPAnalyzer
├── pomdp_optim_dynamics_very_noisy.py        # Supporting: very noisy three-region case
├── pomdp_structural_ablations.py             # Exploratory structural comparisons
├── local_optima_experiments.py               # Python reference port of Experiments A/B
├── test_local_optima.py                      # Gradient correctness tests
├── matlab/
│   ├── pomdp_localopt/                       # Canonical Experiments A & B (MATLAB)
│   │   ├── run_pomdp_localopt_experiment.m
│   │   ├── export_summary_table_markdown.m
│   │   ├── summary_table.md
│   │   └── README.txt
│   └── pomdp_memory_enhancement/             # Canonical Experiment C (MATLAB)
│       ├── run_pomdp_memory_enhancement_experiment.m
│       ├── summary_table.md
│       └── README.txt
└── figures/                                   # Generated output figures
```

## Installation

### Python

```bash
git clone https://github.com/ryan-a-anderson/pomdp-value-geometry.git
cd pomdp-value-geometry
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `scipy`.

### MATLAB

The MATLAB scripts use only base MATLAB (no toolboxes required beyond the Statistics Toolbox for `gamrnd`). Tested on MATLAB R2022b+.

## Reproducing paper results

### Figures (Python)

```bash
# Figs 2 and 3 (linear and semi-algebraic boundaries)
python pomdp_linear_nonlinear_inequalities.py

# Fig 4 (initial state distribution dependence)
python initial_distribution_analysis.py
```

### Experiments A & B — Local optima (MATLAB)

```matlab
cd matlab/pomdp_localopt
results = run_pomdp_localopt_experiment;
```

Outputs `output/batch_summary.png`, `output/landscape_summary.png`, `output/summary_table.md`, `output/results.mat`.

### Experiment C — Finite-memory (MATLAB)

```matlab
cd matlab/pomdp_memory_enhancement
results = run_pomdp_memory_enhancement_experiment;
```

Outputs `output/memory_enhancement_summary.png`, `output/summary_table.md`, `output/results.mat`.

## POMDP instance for Figs 2 & 3

States S = {0, 1}, actions A = {0, 1}, observations O = {0, 1, 2}, discount γ = 0.9.

```
P^0 = [[0.85, 0.15],    P^1 = [[0.65, 0.35],
        [0.25, 0.75]]           [0.15, 0.85]]

R   = [[1, 0],          β = [[0.80, 0.10, 0.10],
        [0, 1]]               [0.30, 0.65, 0.05]]
```

## Experiment setup (Appendix B.2)

All experiments draw POMDP instances from a fixed distribution:
- **Transition kernels**: each row $\alpha_a(\cdot|s) \sim \text{Dirichlet}(\mathbf{1}_S)$ per action.
- **Observation kernel**: each row $\beta(\cdot|s) \sim \text{Dirichlet}(\mathbf{1}_O)$.
- **Rewards**: $r_a(s) \sim \text{Uniform}[0, 10]$.
- **Initial state distribution**: $\rho \sim \text{Dirichlet}(\mathbf{1}_S)$.

Memoryless stochastic policies are parametrized by softmax over observation-conditioned logits. Optimization uses vanilla gradient ascent ($\eta = 0.005$, $T = 3000$) with closed-form gradients via the adjoint $\lambda = (I - \gamma P^\pi)^{-1} \rho$.

## Citation

```bibtex
@article{anderson2026value,
  title={The Value Function Semi-Algebraic Set in POMDPs},
  author={Anderson, Ryan A. and Montúfar, Guido},
  year={2026}
}
```

## Contact

Ryan Anderson — raanderson@g.ucla.edu

## License

MIT — see [LICENSE](LICENSE).
