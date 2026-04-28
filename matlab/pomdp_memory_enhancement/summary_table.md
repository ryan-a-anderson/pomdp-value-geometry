### Additional experiments with finite-memory policies via observation enhancement

To further investigate whether the observed optimization landscape is an artifact of restricting to memoryless policies, we extended our experiments to include **finite-memory policies**.  
We implement these via *observation enhancement*: for a memory length $k$, the policy receives as input the tuple $(o_t, o_{t-1}, \dots, o_{t-k})$, while the underlying process is augmented to remain Markovian. 

We evaluate $k \in \{0,1,2\}$ and compare against the fully observable baseline. 
To ensure a fair comparison across different memory lengths, we (i) initialize all histories with a deterministic null symbol (avoiding artificial random histories), and (ii) scale the optimization budget with \(k\) to account for the increasing policy dimension. 

The results are summarized in the table below. Across all tested configurations, we observe a clear and consistent pattern: 

- **Increasing memory reduces the value spread**: as $k$ increases, the variability of the final objective across random restarts decreases compared to the memoryless case ($k=0$).  
- **Nevertheless, partial observability still induces significant spread**: even for $k=2$, the value spread remains consistently larger than in the fully observable setting, where it is close to zero across all configurations. 
- A similar trend is observed for the fraction of materially suboptimal runs and the policy spread. 

From the perspective of our theoretical framework, these findings are highly consistent with the semi-algebraic characterization of the feasible value set. 
Increasing $k$ enlarges the policy class and effectively refines the parametrization of this set, which can mitigate some optimization difficulties (reflected in the reduced spread).  
However, the **intrinsic geometric complexity induced by partial observability persists**, as evidenced by the remaining variability compared to the fully observable case. 
Overall, these experiments indicate that while incorporating finite memory alleviates some optimization challenges, the presence of multiple local optima with different values is not merely an artifact of memoryless policies, but instead reflects a property of partially observable systems. 


| Config (S,A,O) | Value Spread (partial_k0) | Subopt. Fraction (partial_k0) | Policy Spread (partial_k0) | Value Spread (partial_k1) | Subopt. Fraction (partial_k1) | Policy Spread (partial_k1) | Value Spread (partial_k2) | Subopt. Fraction (partial_k2) | Policy Spread (partial_k2) | Value Spread (full) | Subopt. Fraction (full) | Policy Spread (full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (4,2,2) | 1.126 ± 2.407 | 0.315 ± 0.416 | 0.121 ± 0.150 | 1.122 ± 2.410 | 0.110 ± 0.232 | 0.081 ± 0.153 | 0.111 ± 0.239 | 0.125 ± 0.280 | 0.074 ± 0.161 | 0.021 ± 0.022 | 0.000 ± 0.000 | 0.009 ± 0.011 |
| (4,3,2) | 1.339 ± 2.355 | 0.030 ± 0.054 | 0.034 ± 0.055 | 0.243 ± 0.368 | 0.215 ± 0.427 | 0.017 ± 0.018 | 0.292 ± 0.409 | 0.125 ± 0.240 | 0.057 ± 0.088 | 0.055 ± 0.089 | 0.010 ± 0.022 | 0.019 ± 0.038 |
| (8,2,2) | 0.284 ± 0.385 | 0.205 ± 0.303 | 0.155 ± 0.215 | 0.523 ± 0.719 | 0.260 ± 0.369 | 0.128 ± 0.164 | 0.545 ± 0.819 | 0.310 ± 0.394 | 0.150 ± 0.144 | 0.031 ± 0.016 | 0.000 ± 0.000 | 0.016 ± 0.016 |
| (8,3,2) | 0.287 ± 0.438 | 0.255 ± 0.423 | 0.032 ± 0.054 | 0.305 ± 0.513 | 0.275 ± 0.418 | 0.035 ± 0.047 | 0.138 ± 0.202 | 0.330 ± 0.401 | 0.081 ± 0.113 | 0.113 ± 0.036 | 0.045 ± 0.087 | 0.054 ± 0.051 |
