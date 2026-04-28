#!/usr/bin/env python3
"""
Test multiple POMDP configurations to find examples where different
extreme points are optimal for different initial state distributions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from initial_distribution_analysis import InitialDistributionAnalyzer
from pomdp_linear_nonlinear_inequalities import POMDPAnalyzer


def test_pomdp_config(P, R, Beta, name="POMDP", verbose=True):
    """Test a single POMDP configuration."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

    pomdp = POMDPAnalyzer(P, R, Beta)
    analyzer = InitialDistributionAnalyzer(pomdp)

    # Find extreme points
    extreme_values, extreme_policies, all_values = analyzer.find_extreme_points(n_samples=5000)

    if verbose:
        print(f"\nFound {len(extreme_values)} extreme points:")
        for i, (v, p) in enumerate(zip(extreme_values, extreme_policies)):
            print(f"  Policy {i}: p = {p}, v = {v[:2]}")

    # Find optimal regions
    alphas, optimal_idx, values, boundaries = analyzer.find_optimal_regions(extreme_values)

    if verbose:
        print(f"\nFound {len(boundaries)} boundary points.")

    # Check if this is interesting (has multiple optimal regions)
    has_multiple_regions = len(boundaries) > 0

    if has_multiple_regions and verbose:
        analyzer.print_boundary_equations(boundaries, extreme_values)

        # Generate and save plot
        plot_name = name.replace(' ', '_').replace(':', '').replace('=', '').replace('[', '').replace(']', '').replace(',', '')
        analyzer.plot_analysis(extreme_values, extreme_policies, all_values,
                             alphas, optimal_idx, values, boundaries)
        plt.savefig(f'init_dist_{plot_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to 'init_dist_{plot_name}.png'")

    return {
        'name': name,
        'n_extreme': len(extreme_values),
        'n_boundaries': len(boundaries),
        'extreme_values': extreme_values,
        'extreme_policies': extreme_policies,
        'boundaries': boundaries,
        'has_multiple_regions': has_multiple_regions
    }


def main():
    print("Searching for POMDP configurations with multiple optimal regions...")

    results = []

    # Base transition and reward
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    PDelta = np.array([[-0.20,  0.20], [-0.10,  0.10]])
    P1 = P0 + PDelta
    P = np.array([P0, P1])

    # Test 1: Original (perfect observations)
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    Beta = np.eye(2)
    results.append(test_pomdp_config(P, R, Beta, "Original: Beta=I, R=[[1,0],[0,1]]"))

    # Test 2: Noisy observations
    Beta_noisy = np.array([[0.80, 0.20], [0.30, 0.70]])
    results.append(test_pomdp_config(P, R, Beta_noisy, "Noisy: Beta=[[0.8,0.2],[0.3,0.7]], R=[[1,0],[0,1]]"))

    # Test 3: Different rewards
    R_alt = np.array([[1.0, 0.5], [0.5, 1.0]])
    results.append(test_pomdp_config(P, R_alt, Beta, "AltReward: Beta=I, R=[[1,0.5],[0.5,1]]"))

    # Test 4: Asymmetric rewards
    R_asym = np.array([[2.0, 0.0], [0.0, 1.0]])
    results.append(test_pomdp_config(P, R_asym, Beta, "AsymReward: Beta=I, R=[[2,0],[0,1]]"))

    # Test 5: Noisy observations + different rewards
    results.append(test_pomdp_config(P, R_alt, Beta_noisy, "NoisyAltReward: Beta=[[0.8,0.2],[0.3,0.7]], R=[[1,0.5],[0.5,1]]"))

    # Test 6: Very noisy observations
    Beta_very_noisy = np.array([[0.65, 0.35], [0.35, 0.65]])
    results.append(test_pomdp_config(P, R, Beta_very_noisy, "VeryNoisy: Beta=[[0.65,0.35],[0.35,0.65]], R=[[1,0],[0,1]]"))

    # Test 7: Different transition structure
    P0_alt = np.array([[0.9, 0.1], [0.1, 0.9]])
    P1_alt = np.array([[0.3, 0.7], [0.7, 0.3]])
    P_alt = np.array([P0_alt, P1_alt])
    results.append(test_pomdp_config(P_alt, R, Beta_noisy, "AltTransition: P0=[[0.9,0.1],[0.1,0.9]], P1=[[0.3,0.7],[0.7,0.3]]"))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    interesting = [r for r in results if r['has_multiple_regions']]
    boring = [r for r in results if not r['has_multiple_regions']]

    print(f"\nConfigurations with multiple optimal regions: {len(interesting)}")
    for r in interesting:
        print(f"  ✓ {r['name']}: {r['n_extreme']} extreme points, {r['n_boundaries']} boundaries")

    print(f"\nConfigurations with single optimal policy: {len(boring)}")
    for r in boring:
        print(f"  × {r['name']}: {r['n_extreme']} extreme points, {r['n_boundaries']} boundaries")

    if interesting:
        print(f"\n{'='*80}")
        print("DETAILED RESULTS FOR INTERESTING CONFIGURATIONS")
        print(f"{'='*80}")

        for r in interesting:
            print(f"\n{r['name']}:")
            print(f"  Extreme points:")
            for i, v in enumerate(r['extreme_values']):
                print(f"    Policy {i}: v = {v[:2]}")
            print(f"  Boundaries:")
            for b in r['boundaries']:
                print(f"    α = {b['alpha']:.6f}: Policy {b['policy1_idx']} ↔ Policy {b['policy2_idx']}")


if __name__ == "__main__":
    main()
