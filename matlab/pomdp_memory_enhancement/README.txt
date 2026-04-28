POMDP memory enhancement experiment (MATLAB)
============================================

Main file
---------
- run_pomdp_memory_enhancement_experiment.m

This experiment compares:
- partial_k0 : memoryless stochastic policies on the original POMDP
- partial_k1 : policies using current observation and 1-step observation memory
- partial_k2 : policies using current observation and 2-step observation memory
- partial_k3 : policies using current observation and 3-step observation memory
- full       : fully observable memoryless baseline

For memory k, the policy input is the tuple
    (o_t, o_{t-1}, ..., o_{t-k})
and the hidden augmented state is
    (s_t, o_{t-1}, ..., o_{t-k}).

Outputs
-------
- summary_table.csv
- summary_table.md
- memory_enhancement_summary.png
- results.mat

Usage 
-------
results = run_pomdp_memory_enhancement_experiment;
