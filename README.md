# Value Function Geometry for Memoryless POMDPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code for the paper **"Value Function Geometry for Partially Observable Markov Decision Processes"** (Anderson, 2026).

## Overview

This repository contains the experiments and figures from the paper. The central result is that the set of feasible value functions for a POMDP under memoryless stochastic policies admits an explicit semi-algebraic description — a finite system of polynomial equations and inequalities — extending classical polyhedral results for fully observable MDPs to the partially observable setting.

The experiments illustrate two consequences of this geometric structure:
1. The feasible value set has curved (nonlinear) boundary components, captured exactly by finitely many polynomial constraints.
2. Partial observability induces genuinely non-convex optimization landscapes with multiple isolated local optima under policy gradient, unlike the fully observable case.

## Installation

```bash
git clone https://github.com/ryan-a-anderson/pomdp-value-geometry.git
cd pomdp-value-geometry
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `matplotlib`, `scipy` (see `requirements.txt`).

## Scripts and Paper Correspondence

| Script | Paper section | What it produces |
|--------|--------------|-----------------|
| `pomdp_linear_nonlinear_inequalities.py` | Figures 1 & 2 (Appendix B.1) | Piecewise-linear boundary samples (blue) and semi-algebraic boundary curves (red/green dashed) for a (S=2, A=2, O=3) instance |
| `initial_distribution_analysis.py` | Figure 3 (Appendix B.1) | Optimal policy regions as the initial state distribution ρ varies |
| `initial_distribution_analysis_multi.py` | Figure 3 variants | Same analysis across multiple POMDP configurations |
| `pomdp_structural_ablations.py` | Tables 1–3 (Appendix B.2, Experiments A/B/C) | Policy-gradient spread, local optima count, and finite-memory comparisons across an (S, A, O) grid |
| `pomdp_optim_dynamics.py` | Supporting | Multi-start optimization trajectories and feasible value region visualization |
| `pomdp_optim_dynamics_very_noisy.py` | Supporting | Same for the "very noisy" three-region configuration |

### Reproducing paper figures

**Figures 1 & 2** (linear and semi-algebraic boundaries):
```bash
python pomdp_linear_nonlinear_inequalities.py
```

**Figure 3** (initial state distribution dependence):
```bash
python initial_distribution_analysis.py
```

**Tables 1–3** (Experiments A, B, C):
```bash
python pomdp_structural_ablations.py
```

## POMDP instance used in Figures 1 & 2

States S = {0, 1}, actions A = {0, 1}, observations O = {0, 1, 2}, discount γ = 0.9.

```
P^0 = [[0.85, 0.15],    P^1 = [[0.65, 0.35],
        [0.25, 0.75]]           [0.15, 0.85]]

R   = [[1, 0],          β = [[0.80, 0.10, 0.10],
        [0, 1]]               [0.30, 0.65, 0.05]]
```

## Experiment setup (Appendix B.2)

All experiments sample random POMDP instances from (S, A, O) configurations on the grid {4, 8, 12} × {2, 3, 4} × {2, 3}. For each instance, 50 independent policy-gradient runs are launched from i.i.d. standard-normal logit initializations with learning rate η = 0.005 and T = 3000 steps. All randomness is seeded to 42 for reproducibility.

- **Experiment A**: measures value spread, suboptimal fraction, and policy spread across restarts for partial vs. fully observable instances.
- **Experiment B**: clusters converged value vectors to count distinct local optima reached.
- **Experiment C**: extends to finite-memory policies via observation enhancement and measures how memory reduces (but does not eliminate) value spread.

## Citation

```bibtex
@article{anderson2026value,
  title={Value Function Geometry for Partially Observable Markov Decision Processes},
  author={Anderson, Ryan},
  year={2026}
}
```

## Contact

Ryan Anderson — raanderson@g.ucla.edu

## License

MIT — see [LICENSE](LICENSE).
