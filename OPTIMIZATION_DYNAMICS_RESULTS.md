# POMDP Optimization Dynamics - Experimental Results

**Date**: January 21, 2026
**Author**: Ryan Anderson
**Experiment**: Visualization of optimization trajectories in value function space

## Overview

This document summarizes the optimization dynamics experiments showing how different initial state distributions (ρ) lead to convergence to different local maxima in the POMDP value function space.

## Key Contributions

1. **Iso-objective line visualization**: Shows J(v) = ρᵀv contours in value space
2. **Multi-start trajectory analysis**: Demonstrates basin of attraction structure
3. **Three-region phenomenon**: Documents the very noisy case with 3 distinct optimal policies

## Experiments Conducted

### 1. Standard Noisy Configuration
**File**: `optimization_dynamics_multistart.png`

**Configuration**:
- Beta = [[0.80, 0.20], [0.30, 0.70]] (Noisy observations)
- P0 = [[0.85, 0.15], [0.25, 0.75]]
- R = [[1.0, 0.0], [0.0, 1.0]]
- γ = 0.9

**ρ values tested**: (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)

**Key findings**:
- Iso-objective lines (gray) are perpendicular to the gradient ∇J = ρ (red arrow)
- Different ρ values create different preferred directions in value space
- Multiple random starts converge to different local maxima (stars)
- The feasible region (light blue) constrains where trajectories can exist

### 2. Very Noisy Configuration (3 Optimal Regions)
**File**: `optimization_dynamics_very_noisy_multistart.png`

**Configuration**:
- Beta = [[0.65, 0.35], [0.35, 0.65]] (Very noisy observations)
- Same P0, R, γ as above

**ρ values tested**: (0.2, 0.8), (0.5, 0.5), (0.65, 0.35), (0.75, 0.25), (0.85, 0.15)

**Three distinct optimal policies identified**:

| Region | α range | Optimal Policy | Value Function | Interpretation |
|--------|---------|----------------|----------------|----------------|
| 1 | α < 0.575 | p=[1,1] | v=[5.73, 7.55] | Always action 1 |
| 2 | 0.575 < α < 0.740 | p=[0,1] | v=[6.50, 6.50] | Action depends on observation |
| 3 | α > 0.740 | p=[0,0] | v=[7.07, 4.89] | Always action 0 |

**Key findings**:
- **High observation noise creates multiple local maxima**: The very noisy observation kernel leads to a complex optimization landscape
- **Basin structure varies with ρ**: Different initial distributions favor different basins
- **Trajectories can converge to wrong local optimum**: From cluster analysis, we see that random starts can get stuck in suboptimal maxima
- **Three-cluster phenomenon**: At intermediate α values, optimization finds all three distinct maxima

### 3. Detailed Single-ρ Analysis (α=0.65)
**File**: `optimization_value_space.png`

**Configuration**: Very noisy (Beta = [[0.65, 0.35], [0.35, 0.65]])
**ρ**: (0.65, 0.35) - in the middle of Region 2

**Key observations**:
- 10 random starts from different initial policies
- Purple iso-objective contours show J(v) = 0.65·v(s=0) + 0.35·v(s=1)
- Red/green dashed lines show the exact boundary constraints (q₀=±1, q₁=±1)
- Blue shaded region is the feasible value function set
- Trajectories converge to at least 2 different local maxima:
  - Primary maximum around v ≈ [6.5, 6.5] (most trajectories)
  - Secondary maximum around v ≈ [7.0, 4.9] (some trajectories)

## Cluster Analysis Results

From the fine-grained α sweep (34 values from 0.0 to 1.0):

### Number of Clusters vs α
- **2 clusters**: α ∈ {0.0, 0.125, 0.25, 0.375, 0.606, 1.0}
- **3 clusters**: All other α values (majority of the range)

This indicates that:
1. At extreme α values, only 2 basins exist
2. Throughout most of the range, all 3 distinct maxima have non-empty basins
3. The observation noise creates a persistently multi-modal optimization landscape

### Three Value Function Clusters

Across all α values, trajectories consistently converge to one of three value functions:

1. **v ≈ [5.73, 7.55]** with **p ≈ [1.00, 1.00]**
   - Always take action 1 regardless of observation
   - Optimal when α is low (favor state 1)

2. **v ≈ [6.50, 6.50]** with **p ≈ [0.00, 1.00]**
   - Take action 0 when observing o=0, action 1 when observing o=1
   - Optimal in the middle range of α
   - Most frequently found by optimization

3. **v ≈ [7.07, 4.89]** with **p ≈ [0.00, 0.00]**
   - Always take action 0 regardless of observation
   - Optimal when α is high (favor state 0)

## Visualization Elements Explained

### Multi-panel plots (`optimization_dynamics_*_multistart.png`)
Each panel shows:
1. **Light blue shaded region**: Feasible value function set (satisfies Bellman equation for some policy)
2. **Black boundary**: Exact boundary of feasible region
3. **Gray contour lines**: Iso-objective lines J(v) = ρᵀv with labeled values
4. **Colored trajectories**: Optimization paths from different random starts
5. **Circles (○)**: Initial value functions
6. **Stars (★)**: Final converged value functions
7. **Red arrow**: Gradient direction ∇J = ρ

### Detailed value space plots (`optimization_value_space.png`)
Shows:
1. **Purple contours**: Iso-objective lines
2. **Red dashed lines**: Boundary constraint q₀ = ±1
3. **Green dashed lines**: Boundary constraint q₁ = ±1
4. **Gray hatching**: Support function inequalities (approximation of boundary)
5. **Multiple trajectories**: Different colored paths from random starts

## Mathematical Insights

### 1. Linear Iso-Objectives
The objective J(v) = ρᵀv creates **linear** iso-contours in value space:
- Equation: ρ₀·v(s=0) + ρ₁·v(s=1) = constant
- These are straight lines perpendicular to ρ
- Gradient ∇J = ρ is constant everywhere (not affected by v)

### 2. Non-Convex Feasible Region
The feasible value function set for POMDPs:
- Is **not convex** (visible in the plots)
- Has curved boundaries defined by nonlinear constraints
- Can be described by infinitely many piecewise linear inequalities
- Has exact boundary where q_k = ±1 (from the zonotope characterization)

### 3. Local Maxima from Non-Convexity
Multiple local maxima arise because:
- The feasible region is non-convex (unlike fully observable MDPs)
- Different "corners" of the region can be optimal for different ρ
- Projected gradient ascent can get trapped in local maxima
- The observation noise (partial observability) creates this complexity

### 4. Basin of Attraction Structure
- Each local maximum has a basin of attraction in policy space
- The size of basins varies with ρ (seen in cluster counts)
- Some basins are larger/more stable than others
- Random initialization can lead to any of the local maxima

## Files Generated

### Main Figures
1. `optimization_dynamics_multistart.png` (1.2 MB)
   - 5-panel comparison for standard noisy case

2. `optimization_dynamics_very_noisy_multistart.png` (1.4 MB)
   - 5-panel comparison for very noisy case (3 regions)

3. `optimization_value_space.png` (1.6 MB)
   - Detailed single-ρ analysis with 10 trajectories

4. `optimization_J_curve.png` (88 KB)
   - Objective value J(t) over iterations

5. `optimization_policy_space.png` (221 KB)
   - Heatmap in policy parameter space (p₀, p₁)

### Analysis Plots (Very Noisy)
6. `very_noisy_rho_endpoints.png` (15 KB)
   - Scatter plot of final endpoints colored by α

7. `very_noisy_rho_trajectories.png` (15 KB)
   - Overlay of all trajectories across α sweep

8. `very_noisy_basin_shares.png` (15 KB)
   - Fraction of starts converging to each cluster vs α

### Code Files
9. `pomdp_optim_dynamics.py`
   - Main implementation with POMDPAnalyzer class
   - Includes gradient computation, optimization, and visualization

10. `pomdp_optim_dynamics_very_noisy.py`
    - Specialized script for very noisy experiments

## Usage

### Run standard noisy experiments:
```bash
cd /Users/ryananderson/Documents/Programming/Value\ Function\ Geometry/new_experiments_jan_20
source ../pomdp_env/bin/activate
python pomdp_optim_dynamics.py
```

### Run very noisy experiments:
```bash
python pomdp_optim_dynamics_very_noisy.py
```

## Paper Integration

These visualizations are ideal for:

### Main Text
- **Figure**: Multi-panel comparison (`optimization_dynamics_multistart.png` or very noisy version)
  - Caption: "Optimization dynamics in value function space for different initial state distributions. Each panel shows iso-objective contours (gray), feasible region (blue), and trajectories from multiple random initializations (colored curves). The red arrow indicates the gradient direction ∇J = ρ."

### Supplementary Material
- Detailed single-ρ analysis with boundary constraints
- Basin shares plot showing three-region structure
- Cluster analysis table

### Experimental Section
The experiments demonstrate:
1. How the choice of initial state distribution (ρ) affects which local maximum is found
2. The non-convex geometry of the POMDP value function set
3. The relationship between observation noise and optimization difficulty
4. Explicit evidence of the three-region phenomenon predicted by theory

## Next Steps / Extensions

### Potential Improvements
1. **Larger state spaces**: Test n > 2 states (requires 3D visualization)
2. **Adaptive learning rates**: Could improve convergence
3. **Second-order methods**: Try natural policy gradient or trust region methods
4. **Global optimization**: Compare with global solvers (branch and bound, etc.)

### Additional Experiments
1. Test effect of discount factor γ on basin structure
2. Vary the noise level continuously and track number of local maxima
3. Compare to belief-based POMDP methods
4. Analyze computational cost of different optimization approaches

### Theoretical Questions
1. Can we predict the number of local maxima from POMDP parameters?
2. What conditions guarantee global convergence?
3. How does basin size relate to optimality gap?
4. Connection to value function geometry theorems in the paper?

## References

- Main paper: `paper_in_prog_jan_21.md`
- README with theoretical background: `README.md`
- Initial distribution analysis: `initial_distribution_analysis.py`
