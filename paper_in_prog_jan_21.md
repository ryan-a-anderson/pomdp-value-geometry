# Introduction {#sec:introduction}

Markov decision processes (MDPs) are a powerful model for sequential
decision problems, with a long history of deployment in fields as
disparate as fisheries management [@whiteSurveyApplicationsMarkov1993],
drone warfare [@zhangAdaptiveCollisionAvoidance2023], and playing go
[@silverMasteringGameGo2017]. Originally introduced in the context of
solving optimal control problems in dynamic programming
[@bellmanMarkovianDecisionProcess1957], MDPs are also well-suited to
modeling reinforcement learning problems
[@suttonReinforcementLearningIntroduction2018].

While the original formulation of MDPs allowed for the stochasticity of
the system to emerge in the transition between states, other work has
sought to understand the implications of an additional layer of
uncertainty, namely one where the agent does not have a full
understanding of the state and must act instead only on what
observations they can make (see [@astromOptimalControlMarkov1965], then
developments by [@smallwoodOptimalControlPartially1973] for the
finite-horizon case and [@sondikOptimalControlPartially1978] for the
infinite-horizon, discounted rewards case). These partially observable
Markov decision processes (POMDPs), given their greater flexibility,
have found several applications [@cassandraSurveyPomdpApplications1998],
especially in robotics [@lauriPartiallyObservableMarkov2023], but also
in diverse applications such as elevator control
[@critesElevatorGroupControl1998] and the conservation of rare species
[@chadesPrimerPartiallyObservable2021].

Studies on the sample complexity of algorithms for learning the optimal
policy in POMDPs has illustrated that the partially observable setting
is far more difficult for an agent to navigate than the fully observable
one --- @chenPartiallyObservableRL2022 attribute the difference to the
fact that observations in POMDPs are non-Markovian. For finite-horizon
POMDPs, finding optimality was shown to be exponential in the horizon
length, meaning that learning algorithms are no better than compelling
the agent to fully explore the state space
[@krishnamurthyPACReinforcementLearning2016]. The problem is undecidable
for discounted-reward, infinite-horizon POMDPs
[@papadimitriouComplexityMarkovDecision1987; @madaniUndecidabilityProbabilisticPlanning2003].
This compares to the fully observable case where learning has a sample
complexity which is merely polynomial in the horizon length
[@azarMinimaxRegretBounds2017]. More recent work has sought conditions
under which learning in POMDPs is more tractable, such as the
$\alpha$-weakly revealing POMDPs, which can be learned with polynomial
sample complexity [@liuSampleEfficientReinforcementLearning2022].

Much like the distinction between the discounted-reward, infinite
horizon and finite-horizon settings for MDPs, one also distinguishes
between MDPs where agents have policies with memory and MDPs where
agents' policies are memoryless, i.e., only depend on the current
observation. If the agent is allowed to maintain a belief vector
consisting of previous observations as an estimate of the current world
state, then the POMDP may be modeled as a fully observed MDP
[@astromOptimalControlMarkov1965; @JMLR:v23:20-1165]. Although
memoryless policies may have previously been "considered worthless"
[@cohenFutureMemoriesAre2023], when "external memory" is used to augment
the environment, memoryless policies can reach optimality for POMDPs
[@toroicarteLearningRewardMachines2023].

The value function is a fundamental object that serves to facilitate
finding optimal policies in RL problems. Standard algorithms for finding
the optimal policy in fully observable MDPs, such as value iteration,
are guaranteed to find the optimal policy by acting greedily with
respect to the current value function
[@putermanMarkovDecisionProcesses2005]. The geometry of the set of value
functions can shed light on the structure and complexities of policy
optimization in sequential decision-making problems. Recent work has
advanced characterizations for MDPs, but the case of POMDPs has remained
unexplored. That is precisely the gap that we seek to plug in this
article.

## Related Works

For fully observable MDPs, the set of value functions is a union of
convex polytopes [@dadashiValueFunctionPolytope2019]. In particular, it
is closed and bounded, and is formed by unions of intersections of
finitely many halfplanes. This characterization of the set of value
functions can be used to construct more efficient algorithms for finding
the optimal policy in fully observable MDPs, such as geometric policy
iteration (GPI). [@wuGeometricPolicyIteration2022a] For robust MDPs,
where in each round the transition kernels are chosen adversarially, the
set of value functions is formed by the intersection of conic
hypersurfaces [@wangGeometryRobustValue2022]. For memoryless,
discounted-reward, infinite-horizon MDPs the value function (as well as
other quantities, such as the state-action frequency) is a rational
function in the entries of the policy
[@mullerGeometryMemorylessStochastic2022]. However, to our knowledge the
specific structure of the set of value functions in POMDPs has remained
unexplored.

## Contributions

We regard the set of feasible value functions of MDPs as the solution
set of a parametric equation and provide an explicit description
depending on the properties of the MDP. First, we construct an enclosure
of the solution set as a necessary condition for given value function to
be feasible.

Second, we provide a tight description in terms of infinitely many
piecewise linear inequalities that serve as a sufficient and necessary
condition for a given value function to be feasible.

Finally, we translate the infinite inequality description to a
description in terms of finitely many polynomial equations and
inequalities.

This result applies to both fully observable and partially observable
settings, and highlights the impact of the transition probabilities,
observation probabilities, and instantaneous rewards on the structure of
the value functions and in turn the complexity of the policy
optimization problem.

We recover as a special case the result of
@dadashiValueFunctionPolytope2019 showing that the value functions of
fully observable MDPs form a finite union of convex polytopes.

# Preliminaries {#sec:prelims}

##  Markov Decision Processes

For a finite set $V$, we denote the probability simplex over $V$ as
$\Delta_V=\{p\in\mathbb{R}^V\colon \sum_{v\in V}p(v)=1, p(v)\geq 0\}$.
For a pair of sets $V, W$, we denote the set of Markov kernels from $W$
to $V$ by
$\Delta_V^W =\{ p \in\mathbb{R}^{V\times W}\colon p(\cdot|w) \in\Delta_V, w\in W \}$.

A Markov decision process (MDP) is a tuple
$\mathcal{M} = \langle S, A,O,\alpha, \beta, r, \gamma \rangle$, with
states $S$, actions $A$, observations $O$, transition kernel
$\alpha: S \times A \to %\mathcal{P}(S)
\Delta_S$, observation kernel $\beta: S \to \Delta_O$, instantaneous
reward function $r: S \times A \to \sR$, and discount factor $\gamma$.
When $\beta$ is deterministic and injective, then the MDP is *fully
observable*; otherwise $\mathcal{M}$ is a partially observable Markov
decision process (POMDP).

In the partially observable setting, policies $\pi$ are kernels
$\pi\colon O \to %\mathcal{P}(A)
\Delta_A$,
$\pi(a | o) = \sP(\text{taking action } a | \text{observed } o)$. The
observation kernel $\beta$ determines the probability of observing $o$
given the agent is in state $s$,
$\beta(o | s) = \sP(\text{observing } o | \text{agent in } s)$, while
the transition kernel gives the next-state probability conditional upon
being in state $s$ and taking action $a$, that is,
$\alpha(s'|s, a) = \sP(\text{next state is } s' | \text{in state } s, \text{taking action } a)$.
In much of what follows we will assume that each state leads to an
observation under the observation kernel, i.e., that
$|\supp(\beta(\cdot|s)| > 0$ --- this is an assumption dating back at
least to the argument developed in [@jaakkolaPOMDP1999].

For any policy over observations $\pi\in \Delta^{O}_{A}$ we can define
its effective policy over states. Concretely, for a fixed observation
kernel $\beta$ we consider the map $\Delta^O_A \to \Delta^S_A$ given by
$\tau = \pi \circ \beta$. Note this is a linear map as
$\tau(a|s) = \sum_o\pi(a|o)\beta(o|s)$.

Given a policy, we can define the policy-weighted transition kernel
$P^\pi(s'|s) \in \Delta^S_S$ via $$\begin{aligned}
    P^\pi(s'|s) = \sum_{a \in A} (\pi \circ \beta)(a|s)\alpha(s'|s,a)
    \label{eq:transition-kernel}
\end{aligned}$$ as well as a reward vector $r^\pi \in \mathbb{R}^{S}$
via $$\begin{aligned}
    r^\pi(s) = \sum_{a\in A} (\pi \circ \beta)(a|s)r(s,a). 
    \label{eq:reward-vector}
\end{aligned}$$

For every policy $\pi$, we can calculate the value of the policy in a
state $s$, which is the expected discounted sum of rewards obtained from
beginning in state $s$ and following the policy:

$$V^{\pi}(s) = \mathbb{E}_{P^{\pi}}\left [\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \middle | s_0 =s  \right ].$$

The *value function* of the policy $\pi$ is obtained by taking the value
of the policy for each state and assembling them into a vector ---
$V^{\pi} \in \sR^{S}$.

Note that we can also condition on the initial action to obtain the
$Q$-value function [@suttonReinforcementLearningIntroduction2018]:
$$Q^{\pi}(s,a) = \mathbb{E}_{P^{\pi}}\left [\sum_{t=0}^\infty \gamma^t r(s_t, a_t) \middle | s_0 =s, a_0 = a \right ].$$

Bellman's optimality equation [@bellmanMarkovianDecisionProcess1957]
relates the value function of a policy to the transition kernel and
reward vector it defines as follows: $$\begin{aligned}
    (I - \gamma P^\pi)V^\pi = r^\pi.
    \label{eq:bellman}
\end{aligned}$$

For any MDP, the entries of the transition kernel $\alpha$ lie in the
interval $[0,1]$. Since the policy-weighted transition kernel $P^\pi$ is
obtained by taking the expectation of the transition kernel with respect
to the given policy $\pi$, $P^\pi$ is a stochastic matrix, whose rows
sum to 1. Hence, $\vert \det(P^\pi) \vert < 1$ and
$\det(I-\gamma P^\pi) \neq 0$ for all $\pi \in \Delta_A^S$ and
$\gamma\in(0,1)$. Therefore, in this case the Bellman equation has a
unique solution.

We regard the Bellman equation
([\[eq:bellman\]](#eq:bellman){reference-type="ref"
reference="eq:bellman"}) as a parametric linear system in the
indeterminate $V^\pi\in\mathbb{R}^S$ whose coefficients
$(I - \gamma P^\pi)\in\mathbb{R}^{S\times S}$ and $r^\pi\in\mathbb{R}^S$
are parametrized by the policy $\pi$. From
([\[eq:transition-kernel\]](#eq:transition-kernel){reference-type="ref"
reference="eq:transition-kernel"}) and
([\[eq:reward-vector\]](#eq:reward-vector){reference-type="ref"
reference="eq:reward-vector"}) we see that the parametrization is linear
in the entries of the policy.

We are interested in the geometry of the set of solutions, i.e., the set
of value functions, as the policy ranges over the set of policies over
observations, $\mathcal{V} = \{V^{\pi} : \pi \in \Delta^O_{A} \}$. The
set $\mathcal{V}$ is the *solution set* to the Bellman equation as a
parametric system with parameters $\pi\in\Delta^O_A$.

Note that for any $\pi \in \Delta_A^O$, its value function $V^\pi$ is
given by the value function of its effective policy over states,
$\tau = \pi \circ \beta\in\Delta^S_A$. Thus we may equivalently regard
the system as being parametrized in terms of the effective policies,
which are subject to corresponding constraints. The set of effective
policies $\tau = \pi \circ \beta \in \Delta^S_A$ for fixed $\beta$ and
arbitrary $\pi\in\Delta^O_A$ is the *effective policy polytope*, denoted
$\Delta_{S,\beta}^A$. Note that this is indeed a polytope since it is
the image of the polytope $\Delta^O_A$ under the linear map
$\pi \to \pi \circ \beta$.

<figure id="figure-1">
<p><img src="Figures/fo_po_mdp_policy_region.png" style="width:75.0%"
alt="image" /> <span id="figure-1" data-label="figure-1"></span></p>
<figcaption>The space of feasible policies under full and partial
observability.</figcaption>
</figure>

## Geometry of Value Functions in MDPs {#sec:geometry-value-functions-MDPs}

For a given policy $\pi$, let $Y^\pi_{s_1, \dots, s_k}$ be the set of
policies which agree with $\pi$ on the states $s_1, \dots, s_k$. Two
policies $\pi_1, \pi_2$ agree on states $s_1, \dots, s_k$ if
$\pi_1(\cdot | s_i) = \pi_2(\cdot | s_i)$ for each
$s_i, i = 1, \dots , k$.

::: theorem
Let $\pi$ be a policy in a fully observable MDP. Then the value function
of any policy in $Y^\pi_{s_1, \dots, s_k}$ is contained in the affine
vector space
$$H^\pi_{s_1,\dots,s_k} = V^\pi + \operatorname{span}(C^\pi_{k+1},\dots,C^\pi_{|S|}) ,$$
where $C^\pi_{k+1},\dots,C^\pi_{|S|}$ refer to the
$S\setminus\{s_1,\ldots, s_k\}$ columns of $(I-\gamma P^\pi)$.
:::

In particular, fixing the behavior of a policy $\pi$ on $k$ states
amounts to constraining the value function into an $(|S|-k)$-dimensional
affine space. This insight leads directly to the *line theorem*, which
shows that the value function of any stochastic policy $\pi$ over states
lies on a line segment between the value functions of two policies
$\pi_l,\pi_u \in Y^\pi_{S \setminus \{s\}}$ which are deterministic on
$s$.

For a fully observable MDP, the set of value functions can be
characterized using this approach as a (not necessarily convex) union of
polytopes.

::: theorem
Let $\pi$ be a policy in a fully observable MDP. Consider states
$s_1, \dots, s_k$ and the set of policies that agree with $\pi$ on
$s_1, \dots, s_k$, $Y^\pi_{s_1, \dots, s_k}$. Then the image of
$Y^\pi_{s_1, \dots, s_k}$ under the map $(I-\gamma P^\pi)^{-1}r^\pi$ is
a (non-convex) polytope. Moreover, the image of $Y^\pi_{\emptyset}$ is a
(non-convex) polytope.
:::

## Interval and Parametric Matrix Systems {#sec:intervals}

We introduce interval and parametric linear systems and review key
results in this context. Later we will explain how the results on MDPs
in
Section [2.2](#sec:geometry-value-functions-MDPs){reference-type="ref"
reference="sec:geometry-value-functions-MDPs"} can be derived using the
framework of interval systems. Then we will use the framework of
parametric systems to generalize the results to the case of POMDPs.

Interval linear systems are linear systems $A x - b = 0$ where the
matrix $A\in\mathbb{R}^{m\times n}$ and the vector $b\in\mathbb{R}^m$
have entries specified only up to intervals. The solution sets to such
systems have been studied intensively in the literature since the 1960s.

The solutions of interval linear systems are characterized in the
celebrated Oettli-Prager theorem that we discuss below.

To state this, we use the following notation. We write $[A]$ for a
matrix of intervals, i.e., a set of matrices $A$ whose entries satisfy
$A_{ij} \in [\underline{a_{ij}}, \overline{a_{ij}}]$. Given such an
interval matrix, we write $A^c$ for the corresponding matrix of interval
centers, i.e., the matrix with entries
$A^c_{ij}=\frac12(\underline{a_{ij}} + \overline{a_{ij}})$, and
$A^\Delta$ for the corresponding matrix of interval lengths, i.e., the
matrix with entries
$(A^\Delta)_{ij}= \frac12(\overline{a_{ij}}-\underline{a_{ij}})$.

::: theorem
[]{#th:Oettli-Prager label="th:Oettli-Prager"} A vector
$x\in\mathbb{R}^n$ is the solution to $Ax-b=0$ for some
$m\times n$-matrix $A\in[A]$ and some $m$-vector $b\in [b]$ if and only
if $$|A^c x - b^c|  \leq   A^\Delta |x| + b^\Delta . 
\label{eq:prager-oettli}$$
:::

::: remark
The condition stated in
Theorem [\[th:Oettli-Prager\]](#th:Oettli-Prager){reference-type="ref"
reference="th:Oettli-Prager"} defines a set
$S = \cup_{r\in\{+1,-1\}^n} P_r$, where $P_r$ is the subset of
$\mathbb{R}^n$ cut by the $n$ linear inequalities that define the $r$th
orthant, $\operatorname{diag}(r) x\geq 0$, and $2m$ further linear
inequalities
$A^\Delta \operatorname{diag}(r)x + b^\Delta \pm (A^c x - b^c) \geq 0$.
Of the $n+2m$ inequalities that define each $P_r$ only $m$ are
boundaries of $S$.
:::

Similarly, we can define a parametric linear system $A(p)x - b(p)$,
where $A(p)$ and $b(p)$ depend linearly on a parameter $p$. We consider
an affine linear parametrization of the form
$$A(p) = A^0 + \sum_{k=1}^K A^k p_k,\quad 
b(p) = b^0 + \sum_{k=1}^K b^k p_k, 
\label{eq:parametrization}$$ with fixed $A^k$ and $b^k$, and a parameter
vector $p$ with entries satisfying
$p_k\in[\underline{p_k},\overline{p_k}]$.

We can convert any interval system into a parametric system by allowing
each entry of the parametric matrix to vary as its own parameter
[@popovaExplicitDescriptionAE2012a]. By contrast, moving from a
parametric system to an interval system is more difficult in general, if
at all possible. One method is via preconditioning --- i.e., pick $R$ to
scale the parametric system such that whenever $x$ solves
$A(p)x-b(p) = 0$ it solves $RAx - Rb = 0$. Preconditioning of matrices
emerged in the course of numerical analysis of matrix solvers
[@osbornePreConditioningMatrices1960]. @Hladik-2012 used preconditioning
to create enclosures of solution sets to interval matrix equations.

For parametric systems, the direct analog of the Oettli-Prager theorem
only provides a necessary condition for a given vector $x$ to solve a
system $A(p)x-b(p)=0$.

::: theorem
[]{#thm:hladik-necessary label="thm:hladik-necessary"} If a vector
$x\in\mathbb{R}^n$ is the solution to $A(p)x-b(p)=0$ for some
$p \in [p] = \times_{k=1}^K[\underline{p_k},\overline{p_k}]$, with
$p_k^c = \frac{1}{2}(\overline{p_k}+\underline{p_k})$ and
$p_k^\Delta = \frac{1}{2}(\overline{p_k}-\underline{p_k})$, then
$$|A(p^c)x - b(p^c)| \leq \sum_{k=1}^K p_k^{\Delta}|A^kx-b^k|.$$
:::

We refer to this necessary condition as an *enclosure for the solution
set*. Observe that the result can be interpreted as a finite list of
piecewise linear inequalities which enclose the solution set.

::: remark
The necessary condition in
Theorem [\[thm:hladik-necessary\]](#thm:hladik-necessary){reference-type="ref"
reference="thm:hladik-necessary"} defines a set of the form
$$S = \cap_{i=1}^m S_i, \quad S_i = \cup_{r\in\{+1,-1\}^{K+1}} P_{i,r},$$
where each $P_{i,r}$ is a polyhedron defined by $K+2$ linear
inequalities in $\mathbb{R}^n$, only one of which is a boundary of
$S_i$.

To see this, note that the stated condition comprises $m$ inequalities
$0 \leq \sum_k p_k^\Delta |A^k_{i:}x - b^k_i| - |A(p^c)_{i:}x-b(p^c)_i| =: F_i(x)$,
$i=1,\ldots, m$. Each of these inequalities verifies the non-negativity
of a continuous piecewise linear function $F_i$. This function has one
linear piece for each possible sign of the terms inside the absolute
values, $r\in\{+1,-1\}^{K+1}$. Its solution set is the union of the
solution sets over all linear pieces, $S_i = \cup_{r} P_{i,r}$. Each
$P_{i,r}$ is a polyhedron defined by $K+1$ linear inequalities that
determine the particular linear region of $F_i$, $$\begin{aligned}
    r_{1}(A^1_{i:} x-b^1_i)\geq& 0 \\
    \vdots& \\
    r_{K}(A^K_{i:} x-b^K_i)\geq& 0 \\
    r_{K+1}(A(p^c)_{i:} x -b(p^c)_i)\geq& 0, 
\end{aligned}$$ and one more linear inequality enforcing the
non-negativity of $F_i$,
$$\sum_{k=1}^Kp_k^\Delta r_k (A^k_{i:} x -b^k_i) - r_{K+1}(A(p^c)_{i:} x - b(p^c)_i) \geq 0 .$$
Of the $K+2$ inequalities that define $P_{i,r}$, only the latter one is
a boundary of $S_i$. Note that some of the $P_{i,r}$ may be empty and
thus redundant in the above description.
:::

@Hladik-2012 also provides a condition for parametric matrix systems
which is both necessary and sufficient for defining the solution set.

::: theorem
[]{#thm:hladik-necessary-sufficient
label="thm:hladik-necessary-sufficient"} Let
$[p] = \times_{k=1}^K[\underline{p_k},\overline{p_k}]$,
$p_k^c = \frac{1}{2}(\overline{p_k}+\underline{p_k})$ and
$p_k^\Delta = \frac{1}{2}(\overline{p_k}-\underline{p_k})$. Consider
$A(p) = \sum_{k=1}^K p_k A^k$, $b(p)=\sum_{k=1}^K p_kb^k$ for
$p\in [p]$. Then
$x\in \Sigma = \{x\in\mathbb{R}^n\colon A(p)x=b(p), p\in [p]\}$ if and
only if for every $y \in \mathbb{R}^n$ it solves
$$y^\top (A(p^c)x -b(p^c)) \leq \sum_{k=1}^K p_k^\Delta |y^\top (A^k x - b^k)|.$$
:::

::: remark
Note that this description involves infinitely many piecewise linear
inequalities. Each of these inequalities verifies the non-negativity of
a continuous piecewise linear function with up to $2^K$ linear pieces.
:::

# Main Results

## Parametric System for the Bellman Equation {#sec:ineq}

The Bellman equation has an affine linear parameterization in the
policy. More precisely,
[\[eq:bellman\]](#eq:bellman){reference-type="eqref"
reference="eq:bellman"} with the transition kernel given in
[\[eq:transition-kernel\]](#eq:transition-kernel){reference-type="eqref"
reference="eq:transition-kernel"} and the reward vector given in
[\[eq:reward-vector\]](#eq:reward-vector){reference-type="eqref"
reference="eq:reward-vector"} can be written as $A(p) x - b(p) = 0$,
where $A(p)\in \mathbb{R}^{\mathcal{S}\times\mathcal{S}}$ and
$b(p)\in\mathbb{R}^\mathcal{S}$ are given by $$\begin{aligned}
    A(p) &= A^{0} + \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}} A^{(o,a)} p_{o,a}, \quad \text{with} \label{eq:eqs1a} \\ 
    (A^0)_{s,s'} &= I_{s,s'}, \nonumber \\
    (A^{(o,a)})_{s,s'} &= -\gamma\alpha(s,a;s')\beta(s;o), \quad (o,a)\in\mathcal{O}\times\mathcal{A} , \nonumber \\
    %
    \intertext{and} b(p) &= b^{0} + \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}} b^{(o,a)} p_{o,a}, \quad \text{with}     \label{eq:eqs1b} \\ 
    (b^{0})_s &= 0, \nonumber\\
    (b^{(o,a)})_s &= r(s;a) \beta(s;o), \quad (o,a)\in\mathcal{O}\times\mathcal{A}, \nonumber 
\end{aligned}$$ []{#eq:pomdp-parametrization-1
label="eq:pomdp-parametrization-1"} with parameter
$p = (p_{o,a}) \in\Delta^\mathcal{O}_\mathcal{A}\subseteq \mathbb{R}^{\mathcal{O}\times\mathcal{A}}$.

The characterization of @Hladik-2012 of the solution sets of parametric
systems given in
Theorem [\[thm:hladik-necessary-sufficient\]](#thm:hladik-necessary-sufficient){reference-type="ref"
reference="thm:hladik-necessary-sufficient"} requires an affine
parametrization by a hyperrectangle. We can express our parametric
system $(A(p),b(p))$, $p\in\Delta^\mathcal{O}_\mathcal{A}$ as the
intersection of two such systems, one capturing the inequalities and the
other the equations that define $\Delta^{\mathcal{O}}_\mathcal{A}$. Then
we can describe the solution set in terms of the combined set of
inequalities given by
Theorem [\[thm:hladik-necessary-sufficient\]](#thm:hladik-necessary-sufficient){reference-type="ref"
reference="thm:hladik-necessary-sufficient"} for each of the two
systems.

#### Parameter inequalities

For the first system, we take the relaxation of
eqs. [\[eq:eqs1a\]](#eq:eqs1a){reference-type="ref"
reference="eq:eqs1a"}--[\[eq:eqs1b\]](#eq:eqs1b){reference-type="ref"
reference="eq:eqs1b"} to
$p\in[0,1]^{\mathcal{O}\times\mathcal{A}}\subseteq \mathbb{R}^{\mathcal{O}\times\mathcal{A}}$.
We write $p^c$, with $(p^c)_{o,a}=\frac12$, for the matrix at the center
of this region. Similarly, we write $p^\Delta$, with
$(p^\Delta)_{o,a}=\frac12$, for the matrix of possible deviations from
the center. This system captures the linear inequalities $p_{o,a}\geq0$
that define $\Delta^\mathcal{O}_\mathcal{A}$ as a subset of
$\mathbb{R}^{\mathcal{O}\times\mathcal{A}}$.

#### Parameter equations

For the second system, we consider $B(v) x - c(v)$, where we fix
$\mathcal{A}'=\mathcal{A} \setminus\{a_0\}$ for some $a_0\in\mathcal{A}$
and for $(o,a)\in \mathcal{O}\times \mathcal{A}'$ take $$\begin{aligned}
        B(v) &= B^{0} + \sum_{(o,a)\in \mathcal{O}\times \mathcal{A}'} B^{(o,a)} v_{o,a},\quad \text{with}      \label{eq:eqs2a}\\
        % &\begin{cases}
        (B^0)_{s,s'} &= (A^0)_{s,s'} + \sum_o (A^{(o,a_0)})_{s,s'},\nonumber\\
        %I_{s,s'} + \alpha(s,a_o;s')\beta(s;o), \\
        (B^{(o,a)})_{s,s'} &= (A^{(o,a)})_{s,s'} - (A^{(o,a_0)})_{s,s'}, %(\alpha(s,a;s')- \alpha(s,a_o;s'))\beta(s;o),  
        \nonumber
        %\end{cases}
        \intertext{and}
        c(v) &= c^{0} + \sum_{(o,a)\in \mathcal{O}\times \mathcal{A}'} c^{(o,a)} v_{o,a},\quad \text{with} \label{eq:eqs2b}\\
        %&\begin{cases}
        (c^{0})_s &= \sum_o (b^{(o,a_0)})_s, \nonumber\\
        %r(s;a_o) \beta(s;o), \\
        (c^{(o,a)})_s &= (b^{(o,a)})_s - (b^{(o,a_0)})_s,  
        %r(s;a) \beta(s;o),
        %\end{cases}
        \nonumber
    
\end{aligned}$$ with parameter
$v\in [0,1]^{\mathcal{O}\times\mathcal{A}'}$. We write $v^c$ for the
center of the parameter region, which is the matrix with entries
$(v^c)_{o,a}=\frac12$. Similarly, we write $v^\Delta$ for the matrix of
possible deviations from the center, $(v^\Delta)_{o,a}=\frac12$. This
system captures the linear equations $\sum_{a}p_{o,a}=1$,
$o\in\mathcal{O}$ that define $\Delta^\mathcal{O}_\mathcal{A}$ in
$\mathbb{R}^{\mathcal{O}\times \mathcal{A}}$.

::: proposition
[]{#prop:intersection label="prop:intersection"} The intersection of the
sets
$S_1 =\{ (A(p),b(p)) \colon p\in[0,1]^{\mathcal{O}\times\mathcal{A}}\}$
and
$S_2 = \{(B(v),c(v))\colon v\in[0,1]^{\mathcal{O}\times\mathcal{A}'}\}$
is equal to our set of interest
$S = \{(A(p),b(p)) \colon p\in\Delta^{\mathcal{O}}_{\mathcal{A}}\}$.
:::

::: proof
*Proof.* We first show that $S_1\cap S_2\supseteq S$. Clearly
$S_1\supseteq S$, because $S_1$ is a relaxation of $S$ with a larger set
of parameters. We further observe that $S_2\supseteq S$ as follows. Note
that $S_2$ contains all $(A(p),b(p))$ with $p$ a vertex of
$\Delta^\mathcal{O}_\mathcal{A}$. For any vertex
$p = \sum_o\mathds{1}_{(o,a_o)}$ of $\Delta^\mathcal{O}_\mathcal{A}$ we
can choose $v$ with entries $$v_{o,a}= 
        \begin{cases} 
        1, & (o,a) = (o,a_o),\\
        0, & \text{otherwise}, 
        \end{cases} 
        \quad (o,a)\in\mathcal{O}\times\mathcal{A}',$$ which produces
$$\begin{aligned}
        B(v) 
        &= A^0 + \sum_{o}A^{(o,a_0)} + 
        \\ &\quad\sum_{(o,a)\in\mathcal{O}\times\mathcal{A'}} (A^{(o,a)}-A^{(o,a_0)})v_{o,a}
        \\ &= A^0 + \sum_{o}A^{(o,a_0)} + 
        \\ &\quad \sum_{(o,a_o)\in\mathcal{O}\times\mathcal{A'}} (A^{(o,a)}-A^{(o,a_0)})
        \\ &= A^0 + \sum_{o}A^{(o,a_o)}  
        \\ &= A(p) 
        
\end{aligned}$$ and $$\begin{aligned}
        c(v) 
        &= \sum_o b^{(o,a_0)} + \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'}(b^{(o,a)}-b^{(o,a_0)})v_{o,a}\\
        &= \sum_{o} b^{(o,a_o)}\\
        &= b(p) . 
        
\end{aligned}$$ Since $S_2$ is linear in $v$ and the possible values of
$v$ form a convex set, we have that
$S_2\supseteq \operatorname{conv}\{ (A(p),b(p))\colon \text{$p$ is a vertex of $\Delta^\mathcal{O}_\mathcal{A}$}\} = S$.

Next we show the reverse inclusion $S_1\cap S_2 \subseteq S$. Note that
for any $v\in[0,1]^{\mathcal{O}\times\mathcal{A}'}$ we have
$$\begin{aligned}
    B(v) &= A^0 + \sum_{o}A^{(o,a_0)} + 
    \\ &\quad \sum_{(o,a)\in\mathcal{O}\times\mathcal{A'}} (A^{(o,a)}-A^{(o,a_0)})v_{o,a}
    \\&= A(p^0+p^v) , 
    
\end{aligned}$$ and $$\begin{aligned}
    c(v) = & \sum_o b^{(o,a_0)} + \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'}(b^{(o,a)}-b^{(o,a_0)})v_{o,a}\\
    =& b(p^0+p^v), 
    
\end{aligned}$$ where $p^0=\sum_o \mathds{1}_{(o,a_0)}$ is a vertex of
$\Delta^\mathcal{O}_\mathcal{A}$ and
$$p^v =\sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'} v_{o,a}(\mathds{1}_{(o,a)}-\mathds{1}_{(o,a_0)})$$
is a feasible direction of $\Delta^\mathcal{O}_\mathcal{A}$ at $p^0$.
Thus $(B(v),c(v)) \in \{ (A(p),b(p)) \colon \sum_a p_{o,a}=1\}$. In turn
$S_1\cap S_2 \subseteq \{ (A(p),b(p))\colon p_{o,a}\geq0, \sum_a p_{o,a}=1 \} = S$. ◻
:::

## Solution Set of the Bellman Equation

::: theorem
[]{#thm:value_functions label="thm:value_functions"} Consider a (PO)MDP.
Then $x\in\mathbb{R}^\mathcal{S}$ is a feasible value function, meaning
that it solves the Bellman equation $(I-\gamma P^\pi)x - r^\pi=0$ for
some $\pi\in\Delta^\mathcal{O}_\mathcal{A}$, if and only if it solves
$$\begin{aligned}
    y^\top (A(p^c)x -b(p^c)) \leq& \\
    \sum_{(o,a)\in \mathcal{O}\times\mathcal{A}}  p_{(o,a)}^\Delta |y^\top &(A^{(o,a)} x - b^{(o,a)})| \\ 
    y^\top (B(v^c)x -c(v^c)) \leq& \\
    \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'} v_{(o,a)}^\Delta |y^\top &(B^{(o,a)} x - c^{(o,a)})|  
\end{aligned}$$ for every $y\in\mathbb{R}^n$, where the matrices are
given in eqs. [\[eq:eqs1a\]](#eq:eqs1a){reference-type="ref"
reference="eq:eqs1a"}-[\[eq:eqs1b\]](#eq:eqs1b){reference-type="ref"
reference="eq:eqs1b"} and [\[eq:eqs2a\]](#eq:eqs2a){reference-type="ref"
reference="eq:eqs2a"}-[\[eq:eqs2b\]](#eq:eqs2b){reference-type="ref"
reference="eq:eqs2b"}.
:::

::: proof
*Proof.* This follows from
Proposition [\[prop:intersection\]](#prop:intersection){reference-type="ref"
reference="prop:intersection"} and
Theorem [\[thm:hladik-necessary-sufficient\]](#thm:hladik-necessary-sufficient){reference-type="ref"
reference="thm:hladik-necessary-sufficient"}. ◻
:::

We obtain an analog of
Theorem [\[thm:hladik-necessary\]](#thm:hladik-necessary){reference-type="ref"
reference="thm:hladik-necessary"} for (PO)MDPs as a special case of
Theorem [\[thm:value_functions\]](#thm:value_functions){reference-type="ref"
reference="thm:value_functions"}.

::: corollary
[]{#thm:enclosure_mdp label="thm:enclosure_mdp"} Consider a (PO)MDP.
Then the set of feasible value functions, i.e., solutions to the Bellman
equation $(I-\gamma P^\pi)x - r^\pi = 0$, is contained within the set
defined by the following constraints: $$\begin{aligned}
    \vert A(p^c)x -b(p^c) \vert \leq& \\
    \sum_{(o,a)\in \mathcal{O}\times\mathcal{A}}  p_{(o,a)}^\Delta \vert &(A^{(o,a)} x - b^{(o,a)})\vert \\ 
    \vert B(v^c)x -c(v^c)\vert \leq& \\
    \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'} v_{(o,a)}^\Delta \vert &(B^{(o,a)} x - c^{(o,a)})\vert
    
\end{aligned}$$
:::

::: proof
*Proof.* This follows from
Theorem [\[thm:value_functions\]](#thm:value_functions){reference-type="ref"
reference="thm:value_functions"} by making various special choices of
$y$. ◻
:::

## Converting to a finite system

Theorem [\[thm:value_functions\]](#thm:value_functions){reference-type="ref"
reference="thm:value_functions"} gives a necessary and sufficient
condition for a vector to be a valid value function of a POMDP. However,
it requires infinitely many piecewise linear inequalities to do so ---
we seek an alternative and equivalent characterization that requires
only finitely many equations and inequalities.

We first write the following criterion that we will then use to extract
such a finite characterization.

::: proposition
[]{#col:zonotope-parametric label="col:zonotope-parametric"} A vector
$x$ solves a parametric matrix system $A(p)x - b(p) = 0$ with
parametrization
([\[eq:parametrization\]](#eq:parametrization){reference-type="ref"
reference="eq:parametrization"}) if and only if there exist
$q_k \in [-1,1]$, $k=1,\ldots, K$ such that $$\begin{aligned}
    A(p^c)x - b(p^c) = \sum_{k=1}^K q_k p_k^\Delta (A^k x - b^k).
\end{aligned}$$
:::

::: proof
*Proof.* We start by deriving a necessary and sufficient condition for
$x$ to solve a parametric matrix equation $A(p)x - b(p) = 0$ in terms of
the quantities $A(p^c), b(p^c), p_k^\Delta, A^k, b^k$. The parametric
matrices under consideration depend affinely on the parameters $p_k$ via
$$A(p) = A^0 + \sum_{k=1}^K p_kA^k, \quad 
    b(p) = b^0 + \sum_{k=1}^K p_kb^k.$$ At the parameter midpoint $p^c$
they are given by $A(p^c) = A^0 + \sum_{k=1}^K p^c_kA^k$,
$b(p) = b^0 + \sum_{k=1}^K p^c_kb^k$. A general $p$ in the interval
vector $[p]$ can be expressed as $p = p^c + \delta p$, i.e., the
midpoint plus some deviation. Thus we can write the parametric
coefficients at any arbitrary $p$ as $$\begin{aligned}
    A(p) &= A(p^c + \delta p) = A^0 + \sum_{k=1}^K (p^c_k + \delta p_k)A^k\\ &= A(p^c) + \sum_{k=1}^K \delta p_k A^k,\\
    b(p) &= b^0 + \sum_{k=1}^K p_kb^k = b(p^c) + \sum_{k=1}^K \delta p_k b^k.
    
\end{aligned}$$ Let $q_k \in [-1,1]$ be the factor for each parameter
component, such that $\delta p_k = q_kp^\Delta$. Then $$\begin{aligned}
        A(p) &= A(p^c) + \sum_{k=1}^K q_k p_k^\Delta A^k \\
        b(p) &= b(p^c) + \sum_{k=1}^K q_k p_k^\Delta b^k. 
    
\end{aligned}$$ Therefore we can rewrite $A(p)x-b(p) = 0$ as
$$\begin{aligned}
        A(p)x - b(p) =& 
        A(p^c)x + \sum_{k=1}^K q_k p_k^\Delta A^kx \\ 
        &- ( b(p^c) + \sum_{k=1}^K q_k p_k^\Delta b^k ), \\
       % A(p)x - b(p) 
       =& A(p^c)x-b(p^c) + \sum_{k=1}^K q_k p_k^\Delta (A^kx-b^k).
    
\end{aligned}$$ Thus $x$ solves $A(p)x-b(p) = 0$ for some $p\in[p]$ iff
there exist $q_k \in [-1,1]$ such that
$$A(p^c)x-b(p^c) = \sum_{k=1}^K q_k p_k^\Delta (A^kx-b^k).
    \label{eq:iff-proof}$$ ◻
:::

The above
Proposition [\[col:zonotope-parametric\]](#col:zonotope-parametric){reference-type="ref"
reference="col:zonotope-parametric"} can be further elaborated on.

Let $D$ be the matrix of deviations that appear on the right hand side
of the statement, with the $k$th column of $D$ given as
$$D_k(x) := p_k^\Delta (A^kx-b^k).
\label{eq:def-D}$$ Similarly, let $R^c$ be the vector of midpoint
residuals that appear on the left hand side,
$$R^{c}(x) := A(p^c)x - b(p^c). 
\label{eq:def-R}$$

Then we can rephrase the necessary and sufficient condition stated in
Proposition [\[col:zonotope-parametric\]](#col:zonotope-parametric){reference-type="ref"
reference="col:zonotope-parametric"} in terms of these matrices as
$$R^c = Dq, \quad \text{for some } -1 \leq q \leq 1.$$ This means that
$R^c(x)$ is contained in the zonotope generated by the columns of
$D(x)$.

We interpret this as two conditions:

1.  Equations: $R^c(x)$ is contained in $\operatorname{col}(D(x))$.

2.  Inequalities: the projection of $R^c(x)$ onto
    $\operatorname{col}(D(x))$ satisfies the facet defining inequalities
    of the zonotope generated by the columns of $D(x)$.

#### Equality Condition

The first condition is equivalent to asking that all $(r+1)\times (r+1)$
sub-matrices of $[D|R^c]$ have determinant zero, where
$r = \operatorname{rank}(D)$. This gives us a system of polynomial
equations in $x$ of degree $\operatorname{rank}(D(x))+1$. Note that the
particular equations that need to be verified depend on $\rank(D(x))$
and thus on $x$ itself.

An equivalent condition that can be written independently of
$\operatorname{rank}(D)$ is that the projection of $R^c$ onto
$\operatorname{col}(D)^\bot$ vanishes,[^1]
$$R^{c,\perp} := (I-DD^+)R^c = 0 . 
\label{eq:orth-zero}$$

At those $x$ where $D$ has full column rank or full row rank, the
Moore-Penrose pseudo inverse $D^+$ can be written explicitly as
$(D^\top D)^{-1}D^\top$ and $D^\top(DD^\top)^{-1}$, respectively. In
these cases, using $A^{-1} = \det(A)^{-1}\operatorname{adj}(A)$, we can
write ([\[eq:orth-zero\]](#eq:orth-zero){reference-type="ref"
reference="eq:orth-zero"}) as a system of polynomial equations in $x$.

#### Inequality Condition

To elaborate the second condition, recall that the zonotope
$Z(g_1,\dots, g_K)$ generated by vectors $(g_1,\dots, g_K)$ is
$$Z = \Big\{\sum_{k=1}^K \alpha_kg_k : \alpha_k \in [-1, 1] \Big\}.$$

If a polyhedron $P \subseteq \mathbb{R}^n$ contains $n$ linearly
independent directions, it is said to be full dimensional. Full
dimensional polyhedra have unique and minimal descriptions in terms of a
finite number of linear inequalities, i.e.
$P \subseteq \mathbb{R}^n = \{ x \in \mathbb{R}^n: a_ix \leq b_i, i = 1, \dots, m\}$.
These descriptions are minimal in that if you remove any of them you no
longer have $P$ and they are unique up to positive scaling. Each of the
inequalities in the unique and minimal description of $P$ defines a
*facet* of $P$, a face of the polyhedron which has dimension equal to
one less than the dimension of $P$ [@wolsey_integer_2020 Prop 9.2].

Since zonotopes are polyhedra, we can consider the facets of a zonotope
and their defining inequalities. The facet-defining inequalities for $z$
to be in the zonotope take the form $c^\top z \leq \sum_k |c^\top g_k|$,
where $c$ is orthogonal to a maximal set of linearly independent
generators that span said facet (there are always to choices up to norm
with opposite sign). Thus we have inequalities of the form
$$c^\top (DD^+)R^c \leq \sum_k |c^\top D_k|, 
\label{eq:zon-ineq}$$ where $c$ is any column of $(I - D_ID_I^+)$, i.e.,
a vector orthogonal to the columns of $D_I$, and $D_I$ is a sub-matrix
of $D$ collecting $d-1$ linearly independent columns.

Again, at those points $x$ where $D(x)$ has full column rank or full row
rank, we can write $D^+$ explicitly and
([\[eq:zon-ineq\]](#eq:zon-ineq){reference-type="ref"
reference="eq:zon-ineq"}) is a polynomial inequality in $x$.

::: example
Consider a $2 \times 2$ real parametric system in 2 parameters with
$$\begin{aligned}
A(p) &= A^0 + p_1A^1 + p_2A^2
\\ b(p) &= b^0 + p_1b^1 + p_2b^2 . 
\end{aligned}$$

Let $$A^0 = \begin{pmatrix}
a^0_{11} & a^0_{12} \\[4pt]
a^0_{21} & a^0_{22}
\end{pmatrix}, \quad
b^0 = \begin{pmatrix}
b^0_1 \\[3pt]
b^0_2
\end{pmatrix},$$ and for $k=1,2$, $$A^k = \begin{pmatrix}
a^k_{11} & a^k_{12} \\[4pt]
a^k_{21} & a^k_{22}
\end{pmatrix},
b^k = \begin{pmatrix}
b^k_1 \\[3pt]
b^k_2
\end{pmatrix},
p_k^\Delta = \Delta_k, p^c = 0.$$

For $x=(x_1,x_2)$, the midpoint residual vector is $$\begin{aligned}
    R^c(x) &=  
    \begin{pmatrix}
    r_1(x) \\[4pt]
    r_2(x)
    \end{pmatrix} \\
    &= 
    \begin{pmatrix}
    a^0_{11}x_1 + a^0_{12}x_2 \;-\; b^0_1 \\[6pt]
    a^0_{21}x_1 + a^0_{22}x_2 \;-\; b^0_2
    \end{pmatrix} 
    .
\end{aligned}$$ The matrix
$D(x) = \bigl[\,d^1(x)\;\;d^2(x)\bigr] \in\mathbb{R}^{2\times2}$ has
columns $$\begin{aligned}
    d^k(x)
    &=\Delta_k\bigl(A^k x - b^k\bigr) \\
    &=\Delta_k
    \begin{pmatrix}
    a^k_{11}x_1 + a^k_{12}x_2 - b^k_1 \\[4pt]
    a^k_{21}x_1 + a^k_{22}x_2 - b^k_2
    \end{pmatrix}, \quad k=1,2. 
\end{aligned}$$

We have that $(I - D\,D^\dagger)R^c = 0$ iff the augmented matrix
$\bigl[D \mid R^c\bigr]$ has the same rank as $D$. The latter is the
case iff the $(r+1)\times(r+1)$ minors of $[D | R^c ]$ vanish, where $r$
is the rank of $D$.

We can distinguish a few cases here: first let $\rank(D) = 2$. Since
$D \in \mathbb{R}^{2\times 2}$, $\rank(D) = 2$ implies
$\col(D) = \mathbb{R}^{2}$, and hence $R^c$ must be contained within it.
Moreover, the above condition on the minors of the augmented matrix
still holds --- we need to ensure all $3 \times 3$ minors vanish, but
$[D \vert R^c] \in \mathbb{R}^{2 \times 3}$ does not have any such
minors and thus there is no constraint.

If $\rank(D) = 1$, then we need the $2 \times 2$ minors of
$\bigl[D \mid R^c\bigr]$ to vanish, meaning we will evaluate the
determinants $\det\begin{pmatrix}
d^1_1(x) & d^2_1(x) \\[4pt]
d^1_2(x) & d^2_2(x)
\end{pmatrix}$, $\det\begin{pmatrix}
d^1_1(x) & r_1(x) \\[4pt]
d^1_2(x) & r_2(x)
\end{pmatrix}$ and $\det\begin{pmatrix}
d^2_1(x) & r_1(x) \\[4pt]
d^2_2(x) & r_2(x)
\end{pmatrix}$.
:::

Finally, we rewrite the necessary and sufficient condition in terms of a
condition on the midpoint residuals
$R^c(V^\pi) = A(\frac{1}{2})V^\pi - b(\frac{1}{2})$ as existing inside a
zonotope in $\mathbb{R}^{|S|}$.

::: theorem
[]{#thm:zonotope_mdp label="thm:zonotope_mdp"}
$x\in\mathbb{R}^\mathcal{S}$ is a feasible value function, meaning that
it solves the Bellman equation $(I-\gamma P^\pi)x - r_\pi=0$ for some
$\pi\in\Delta^\mathcal{O}_\mathcal{A}$, if and only if there exist
vectors $q^{(1)} \in [-1,1]^{\mathcal{O}\times\mathcal{A}}$ and
$q^{(2)} \in [-1,1]^{\mathcal{O}\times\mathcal{A}'}$ such that
$$\begin{aligned}
    A(p^c)x - b(p^c) &= \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}} q^{(1)}_{o,a} (A^{(o,a)}x - b^{(o,a)}) \\
    B(v^c)x - c(v^c) &= \sum_{(o,a)\in\mathcal{O}\times\mathcal{A}'} q^{(2)}_{o,a} (B^{(o,a)}x - c^{(o,a)})
\end{aligned}$$ where $p^c$ and $v^c$ are the center points with all
entries equal to $\frac{1}{2}$ and where the matrices are given in
eqs. [\[eq:eqs1a\]](#eq:eqs1a){reference-type="ref"
reference="eq:eqs1a"}-[\[eq:eqs1b\]](#eq:eqs1b){reference-type="ref"
reference="eq:eqs1b"} and [\[eq:eqs2a\]](#eq:eqs2a){reference-type="ref"
reference="eq:eqs2a"}-[\[eq:eqs2b\]](#eq:eqs2b){reference-type="ref"
reference="eq:eqs2b"}.
:::

::: proof
*Proof.* We noted in
[\[prop:intersection\]](#prop:intersection){reference-type="ref"
reference="prop:intersection"} that the Bellman equation could be
reparametrized into the two systems described in
eqs. [\[eq:eqs1a\]](#eq:eqs1a){reference-type="ref"
reference="eq:eqs1a"}-[\[eq:eqs1b\]](#eq:eqs1b){reference-type="ref"
reference="eq:eqs1b"} and [\[eq:eqs2a\]](#eq:eqs2a){reference-type="ref"
reference="eq:eqs2a"}-[\[eq:eqs2b\]](#eq:eqs2b){reference-type="ref"
reference="eq:eqs2b"}.

Since the solution set for these two parametric systems are themselves
described by the result in
[\[col:zonotope-parametric\]](#col:zonotope-parametric){reference-type="ref"
reference="col:zonotope-parametric"}, we can obtain our desired result
by enforcing that they both hold simultaneously. ◻
:::

In Figure [2](#fig:2){reference-type="ref" reference="fig:2"} we show a
comparison of the results of
Theorem [\[thm:zonotope_mdp\]](#thm:zonotope_mdp){reference-type="ref"
reference="thm:zonotope_mdp"} with those of
Theorem [\[thm:hladik-necessary-sufficient\]](#thm:hladik-necessary-sufficient){reference-type="ref"
reference="thm:hladik-necessary-sufficient"}. In particular, we are able
to exactly cut out the feasible space of value functions for the POMDP
with only four curves --- the solutions to
$\vert q_1 \vert, \vert q_2 \vert = 1$.

![Describing the feasible region via infinitely many linear inequalities
vs finitely many nonlinear
inequalities.](linear_nonlinear_ineqs.png){#fig:2 width="100%"}

## Experiments

# Conclusion

In [@mullerGeometryMemorylessStochastic2022], the authors already noted
that value functions admitted rational parametrizations, and gave bounds
on the degree of these rational functions in terms of the cardinality of
states that could be observed at a given observation $\tilde{o}$. In
particular, this shows that the line theorem does not hold for POMDPs,
as the value functions on $Y^\pi_{S_i}$ do not lose dimensionality
linearly in the size of the subset $S_i$, but rather in a matter that
depends on the rank of the observation kernel.

Moreover, authors in [@dadashiValueFunctionPolytope2019] show that the
line theorem can also be used to justify the effectiveness of policy
improvement for fully observable MDPs. However, policy improvement is in
general not feasible for POMDPs, as the possibility of receiving the
same observation in two different states introduces a \"correlation\"
[@liFindingOptimalMemoryless2011]. As we have shown, the failure of a
line theorem to exist for POMDPs emerges from the non-linearity of the
relationship between policies which agree on many states and their value
functions. The rational characterization we have expounded on here
provides a better understanding of what drives the extreme complexity of
this class of models.

[^1]: For any matrix $A \in \mathbb{R}^{m \times n}$ and vector
    $b \in \mathbb{R}^n$, the component of $b$ orthogonal to the column
    space of $A$ is given by $(I-AA^+)b$, where $A^+$ denotes the
    Moore-Penrose pseudoinverse of $A$ [see, e.g., @axler Theorem 6.69].
