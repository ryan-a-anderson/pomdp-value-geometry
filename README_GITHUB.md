# Value Function Geometry for Memoryless POMDPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code repository for the paper "Value Function Geometry for Partially Observable Markov Decision Processes" (2026).

## Overview

This repository contains implementations and experiments exploring the geometric structure of feasible value functions for **memoryless POMDP policies**. We provide:

1. **Theoretical characterization** of the value function set as solutions to parametric linear systems
2. **Optimization dynamics visualization** showing how initial state distributions affect convergence to local maxima
3. **Initial distribution analysis** identifying optimal policy regions
4. **Computational tools** for analyzing POMDPs with finite state and observation spaces

### Key Contributions

- **Exact characterization**: The set of feasible value functions is characterized via infinitely many piecewise linear inequalities (Theorem 3.1) and finitely many polynomial equations and inequalities (Theorem 3.3)
- **Optimization landscape**: Visualization tools reveal how partial observability creates multiple local maxima in the optimization landscape
- **Three-region phenomenon**: Identification of POMDP configurations with three distinct optimal policies depending on initial state distribution

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pomdp-value-geometry.git
cd pomdp-value-geometry

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Optimization Dynamics Visualization

Generate multi-panel plots showing how different initial state distributions lead to different optimization trajectories:

```bash
python pomdp_optim_dynamics.py
```

**Output**: `optimization_dynamics_multistart.png`

This creates a 5-panel figure showing:
- Feasible value function regions (light blue)
- Iso-objective contour lines J(v) = ρᵀv
- Optimization trajectories from multiple random starts
- Convergence to different local maxima

### 2. Very Noisy Configuration (3 Optimal Regions)

Test the configuration exhibiting three distinct optimal policies:

```bash
python pomdp_optim_dynamics_very_noisy.py
```

**Outputs**:
- `optimization_dynamics_very_noisy_multistart.png` - Multi-panel comparison
- `very_noisy_rho_endpoints.png` - Endpoint scatter plot
- `very_noisy_basin_shares.png` - Basin of attraction analysis

### 3. Initial Distribution Analysis

Analyze which extreme point policies are optimal for different initial distributions:

```bash
python initial_distribution_analysis.py
```

**Output**: `initial_distribution_analysis.png` (3-panel figure with extreme points, value curves, and optimal policy regions)

## Repository Structure

```
.
├── pomdp_optim_dynamics.py              # Main optimization dynamics code
├── pomdp_optim_dynamics_very_noisy.py   # Very noisy configuration experiments
├── initial_distribution_analysis.py      # Initial distribution optimizer
├── initial_distribution_analysis_multi.py # Multi-configuration testing
├── pomdp_linear_nonlinear_inequalities.py # Core POMDP analyzer
├── paper_in_prog_jan_21.md              # Paper manuscript
├── README.md                             # Detailed technical documentation
├── OPTIMIZATION_DYNAMICS_RESULTS.md      # Experimental results summary
├── requirements.txt                      # Python dependencies
└── figures/                              # Generated figures
```

## Usage Examples

### Basic POMDP Analysis

```python
import numpy as np
from pomdp_optim_dynamics import POMDPAnalyzer

# Define POMDP parameters
P0 = np.array([[0.85, 0.15], [0.25, 0.75]])  # Transition for action 0
P1 = np.array([[0.65, 0.35], [0.15, 0.85]])  # Transition for action 1
P = np.array([P0, P1])

R = np.array([[1.0, 0.0], [0.0, 1.0]])       # Rewards
Beta = np.array([[0.80, 0.20], [0.30, 0.70]]) # Observation kernel
gamma = 0.9                                   # Discount factor

# Create analyzer
pomdp = POMDPAnalyzer(P, R, Beta, gamma)

# Solve for value function given policy
policy = np.array([0.5, 0.8])  # π(a=1|o) for each observation
value = pomdp.solve_v(policy)
print(f"Value function: {value}")

# Check if value function is feasible
q = pomdp._get_q_of_x(value)
is_feasible = np.all(np.abs(q) <= 1.0)
print(f"Feasible: {is_feasible}")
```

### Optimization from Initial Distribution

```python
# Define initial state distribution
rho = np.array([0.3, 0.7])  # Favor state 1

# Run projected gradient ascent
history = pomdp.optimize_projected_gradient(
    rho=rho,
    p_init=np.random.rand(2),  # Random initial policy
    steps=200,
    lr=0.25,
    momentum=0.6
)

# Access results
final_policy = history["p"][-1]
final_value = history["v"][-1]
final_objective = history["J"][-1]

print(f"Optimal policy: {final_policy}")
print(f"Value function: {final_value}")
print(f"Objective J(ρ, v) = {final_objective:.4f}")
```

### Multi-Start Comparison

```python
# Compare optimization across different initial distributions
rhos = [
    (0.2, 0.8),   # Favor state 1
    (0.5, 0.5),   # Balanced
    (0.8, 0.2),   # Favor state 0
]

fig = pomdp.plot_multistart_comparison(
    rhos=rhos,
    n_starts=6,      # Random initializations per ρ
    steps=200,
    lr=0.25,
    momentum=0.6,
    grid_res=200,
    num_iso_levels=12
)
fig.savefig('my_comparison.png', dpi=300, bbox_inches='tight')
```

## Key Results

### Configuration: Very Noisy Observations

**Parameters**: Beta = [[0.65, 0.35], [0.35, 0.65]]

This configuration exhibits **three distinct optimal regions**:

| Region | α range | Optimal Policy | Value Function | Interpretation |
|--------|---------|----------------|----------------|----------------|
| 1 | α < 0.575 | p=[1,1] | v=[5.73, 7.55] | Always action 1 |
| 2 | 0.575 < α < 0.740 | p=[0,1] | v=[6.50, 6.50] | Action depends on obs |
| 3 | α > 0.740 | p=[0,0] | v=[7.07, 4.89] | Always action 0 |

### Optimization Landscape

Our experiments reveal:

1. **Multiple local maxima**: High observation noise creates non-convex optimization landscapes
2. **Basin structure**: Different random initializations converge to different local optima
3. **ρ-dependent convergence**: The initial state distribution determines which basins are favored
4. **Non-convex feasible region**: Unlike fully observable MDPs, the POMDP value function set is non-convex

## Mathematical Framework

### Bellman Equation as Parametric System

For a POMDP with policy π over observations, the value function v satisfies:

```
A(π)v = b(π)
```

where A(π) and b(π) depend affinely on the policy parameters:

```
A(π) = A⁰ + Σ_{o,a} π(a|o)β(o|s) A^{o,a}
b(π) = b⁰ + Σ_{o,a} π(a|o)β(o|s) b^{o,a}
```

### Feasibility Characterization

A vector v ∈ ℝ^|S| is a feasible value function if and only if:

**Necessary condition** (Theorem 3.2):
```
|A(p^c)v - b(p^c)| ≤ Σ_k p_k^Δ |A^k v - b^k|
```

**Sufficient and necessary** (Theorem 3.1):
```
For all y ∈ ℝ^|S|:
  y^T(A(p^c)v - b(p^c)) ≤ Σ_k p_k^Δ |y^T(A^k v - b^k)|
```

**Finite characterization** (Theorem 3.3):
The feasible set can be described by finitely many polynomial equations and inequalities via zonotope membership.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anderson2026value,
  title={Value Function Geometry for Partially Observable Markov Decision Processes},
  author={Anderson, Ryan},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Paper

The full paper is available at: [arXiv:XXXX.XXXXX](https://arxiv.org)

## Documentation

- **README.md**: Detailed technical documentation and theoretical background
- **OPTIMIZATION_DYNAMICS_RESULTS.md**: Comprehensive experimental results
- **paper_in_prog_jan_21.md**: Full paper manuscript

## Examples

See the `examples/` directory for additional usage examples:
- `example_basic.py`: Basic POMDP analysis
- `example_optimization.py`: Policy optimization examples
- `example_visualization.py`: Custom visualization examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Ryan Anderson - [your-email@example.com]

Project Link: [https://github.com/yourusername/pomdp-value-geometry](https://github.com/yourusername/pomdp-value-geometry)

## Acknowledgments

- This work builds on the theoretical framework of parametric linear systems by Hladík (2012)
- Geometric characterization of MDP value functions by Dadashi et al. (2019)
- POMDP theory foundations from Åström (1965) and Sondik (1978)
