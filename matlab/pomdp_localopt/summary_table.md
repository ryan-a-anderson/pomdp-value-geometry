### Additional experiments 

To address the reviewer’s concern, we conducted a systematic empirical study across a range of randomly generated POMDP instances with varying numbers of states \(S \in \{4,8,12\}\), actions \(A \in \{2,3,4\}\), and observations \(O \in \{2,3\}\).  
For each configuration, we sampled multiple independent instances and ran multiple random restarts of a policy gradient method over memoryless stochastic policies.

The table below reports, for each configuration, aggregate statistics over instances:
(i) the spread of achieved values across restarts,  
(ii) the fraction of materially suboptimal runs (gap \(> 0.01\)), and  
(iii) the spread of the resulting policies.

Across all configurations, we consistently observe that **partial observability induces a substantial spread in both value and policy space**, whereas the fully observable baseline exhibits **negligible spread**.  
In particular, the value spread in the partially observable setting is often significantly larger than the near-zero spread in the fully observable case, indicating the presence of multiple local optima with significantly different performance levels.  
Similarly, the policy spread is consistently large under partial observability, indicating convergence to different policies depending on the initialization, while remaining close to zero in the fully observable case, indicating convergence to essentially identical policies in the latter.

From the perspective of our theory, these observations are consistent with the fact that, under partial observability, the set of achievable value functions induced by memoryless policies forms a **semi-algebraic set with nontrivial geometry**, including curved boundaries, which gives rise to multiple local optima of different values. In contrast, in the fully observable case, the feasible set reduces to a polyhedral set, leading to essentially unimodal optimization behavior.  The empirical results therefore provide direct evidence for the geometric mechanism underlying our theoretical analysis. 

These experiments exhibit a **clear and consistent pattern** across all tested configurations, demonstrating that the phenomenon is robust and not limited to toy examples, but instead reflects a fundamental difference in the optimization landscape induced by partial observability. We are currently extending these preliminary experiments to larger sizes and additional settings. 


| Config (S,A,O) | Value Spread (Partial) | Value Spread (Full) | Subopt. Fraction (Partial) | Subopt. Fraction (Full) | Policy Spread (Partial) | Policy Spread (Full) |
|---|---:|---:|---:|---:|---:|---:|
| (4,2,2) | 18.783 ± 11.845 | 0.007 ± 0.006 | 0.263 ± 0.216 | 0.000 ± 0.000 | 0.317 ± 0.145 | 0.009 ± 0.021 |
| (4,2,3) | 17.062 ± 16.408 | 0.013 ± 0.020 | 0.103 ± 0.205 | 0.000 ± 0.000 | 0.117 ± 0.194 | 0.007 ± 0.015 |
| (4,3,2) | 15.672 ± 20.881 | 0.029 ± 0.049 | 0.223 ± 0.264 | 0.000 ± 0.000 | 0.192 ± 0.175 | 0.006 ± 0.013 |
| (4,3,3) | 23.090 ± 8.862 | 0.041 ± 0.033 | 0.287 ± 0.218 | 0.000 ± 0.000 | 0.279 ± 0.167 | 0.005 ± 0.008 |
| (4,4,2) | 5.704 ± 9.200 | 0.270 ± 0.422 | 0.160 ± 0.228 | 0.003 ± 0.008 | 0.131 ± 0.168 | 0.016 ± 0.031 |
| (4,4,3) | 24.628 ± 18.339 | 0.301 ± 0.229 | 0.180 ± 0.219 | 0.000 ± 0.000 | 0.144 ± 0.140 | 0.011 ± 0.011 |
| (8,2,2) | 26.695 ± 4.932 | 0.043 ± 0.051 | 0.220 ± 0.145 | 0.000 ± 0.000 | 0.314 ± 0.153 | 0.006 ± 0.008 |
| (8,2,3) | 18.780 ± 14.591 | 0.028 ± 0.018 | 0.120 ± 0.125 | 0.000 ± 0.000 | 0.184 ± 0.173 | 0.011 ± 0.014 |
| (8,3,2) | 19.638 ± 11.285 | 0.054 ± 0.057 | 0.313 ± 0.182 | 0.000 ± 0.000 | 0.312 ± 0.088 | 0.020 ± 0.036 |
| (8,3,3) | 25.778 ± 14.546 | 0.158 ± 0.120 | 0.220 ± 0.192 | 0.000 ± 0.000 | 0.233 ± 0.122 | 0.030 ± 0.031 |
| (8,4,2) | 25.460 ± 18.234 | 0.180 ± 0.302 | 0.270 ± 0.200 | 0.003 ± 0.008 | 0.237 ± 0.128 | 0.019 ± 0.037 |
| (8,4,3) | 10.616 ± 7.909 | 0.204 ± 0.279 | 0.293 ± 0.220 | 0.050 ± 0.122 | 0.240 ± 0.148 | 0.054 ± 0.058 |
| (12,2,2) | 15.348 ± 8.547 | 0.041 ± 0.037 | 0.197 ± 0.159 | 0.000 ± 0.000 | 0.279 ± 0.155 | 0.010 ± 0.019 |
| (12,2,3) | 14.260 ± 5.207 | 0.051 ± 0.039 | 0.243 ± 0.166 | 0.000 ± 0.000 | 0.328 ± 0.185 | 0.009 ± 0.014 |
| (12,3,2) | 19.506 ± 11.628 | 0.128 ± 0.131 | 0.293 ± 0.187 | 0.000 ± 0.000 | 0.297 ± 0.117 | 0.024 ± 0.033 |
| (12,3,3) | 19.336 ± 9.808 | 0.105 ± 0.114 | 0.223 ± 0.180 | 0.000 ± 0.000 | 0.243 ± 0.127 | 0.039 ± 0.041 |
| (12,4,2) | 13.337 ± 9.849 | 0.202 ± 0.403 | 0.367 ± 0.177 | 0.003 ± 0.008 | 0.298 ± 0.082 | 0.002 ± 0.004 |
| (12,4,3) | 18.199 ± 6.600 | 0.151 ± 0.171 | 0.267 ± 0.149 | 0.000 ± 0.000 | 0.254 ± 0.119 | 0.019 ± 0.025 |
