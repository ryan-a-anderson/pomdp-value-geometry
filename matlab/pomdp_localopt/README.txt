POMDP local optima experiment (MATLAB)
=====================================

Files
-----
- run_pomdp_localopt_experiment.m
    Main script. Run this file from MATLAB.

What the code does
------------------
The script illustrates how gradient-based optimization over memoryless
stochastic policies in discounted POMDPs can converge to different final
values from different initializations.

For each policy pi(a|o), the code computes the induced value function V^pi
exactly from the Bellman equation:
    V = r_pi + gamma * P_pi * V.

The objective being optimized is
    J(pi) = mu' * V^pi,
where mu is the initial state distribution.

Optimization variable
---------------------
The actual optimization variable is a matrix of policy logits theta(a,o),
with
    pi(:,o) = softmax(theta(:,o)).
This keeps each policy column on the probability simplex automatically.

Experiments included
--------------------
1) Batch experiment:
   - Sample many random discounted control problems.
   - For each fixed instance, run gradient ascent from many random starts.
   - Compare:
       * partially observable case,
       * fully observable baseline using the same transitions/rewards.

   The script then plots:
   - histogram of normalized suboptimality gaps,
   - distribution of final objective values,
   - fraction of suboptimal restarts by instance,
   - spread of converged values by instance.

2) Low-dimensional landscape:
   - Builds a 2-observation, 2-action POMDP,
   - evaluates J on a fine grid of policies,
   - overlays gradient-ascent trajectories and endpoint locations.

Outputs
-------
The script writes an "output" folder containing:
- batch_summary.png
- landscape_summary.png
- results.mat

Practical notes
---------------
- The code uses only basic MATLAB functionality.
- It uses exact gradients obtained by implicit differentiation of the
  Bellman system, not finite differences.
- You can adjust dimensions, number of restarts, and optimization settings
  in the top part of run_pomdp_localopt_experiment.m.

Suggested usage
---------------
Run:
    results = run_pomdp_localopt_experiment;

Then inspect:
    results.batch
    results.landscape
