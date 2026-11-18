# Mathematical formulation

**Competing Platforms in Mobile Crowdsensing: Learning Platforms and Strategic Users**

## 1. Environment

Values are for the full D1 setup (`simulation_full.ipynb`). The earlier 10×10 synthetic pilot is in `simulation.ipynb`.

| Symbol | Meaning | D1 value |
|-|-|-|
| $\mathcal G$ | 500 m cells over central Beijing (lat 39.83–39.99, lon 116.27–116.49) | $36\times38$, $|\mathcal G|=1368$ |
| $\mathcal J=\{1,\dots,M\}$ | platforms (apps) | $M\in\{2,3,4\}$: air quality, traffic, noise, weather (first $M$) |
| $\mathcal N=\{1,\dots,N\}$ | users | $N=300$: 200 T-Drive taxis + 100 GeoLife pedestrians |
| $t=1,2,\dots$ | rounds (1 round = 1 hour); hour-of-week $h(t)=t \bmod 168$ | the week is replayed |
| $w_{jg}\in[0,1]$ | how much platform $j$ cares about block $g$ | built from observed activity: with $\tilde T_g,\tilde P_g\in[0,1]$ the log-normalised taxi and pedestrian user-hours, $w_{1g}=0.2+0.8\tilde P_g$, $w_{2g}=0.1+0.9\tilde T_g$, $w_{3g}=0.2+0.4(\tilde T_g+\tilde P_g)$, $w_{4g}=0.5$ |
| $B_j$ | hourly budget of platform $j$ | equal across apps; three levels (tight, base, loose) |
| $L\subset\mathcal G$ | low-traffic cells | the 25% of cells with the fewest user-hours in the week |
| $V$ | value of a fully covered weight-1 block | 20 |
| $\rho$ | probability that one reading is good | 0.5 |
| $q(n)=1-(1-\rho)^n$ | data quality of a block sensed by $n$ users (submodular) | |
| $b$ | base reward (money scale) | 5 |
| $P_i(t)$ | blocks on user $i$'s route in hour $t$ | cells the real trace passes through in hour-of-week $h(t)$; empty means user $i$ is inactive that hour |
| $\mathcal G_i(t)$ | blocks user $i$ can reach: $\{g:\operatorname{dist}(g,P_i(t))\le 1\}$ | |
| $d_i(g)$ | detour in km from the route to $g$ (0 if on the route) | 0 or 0.5 |

**User type** (private information): $\theta_i=(c_i,\kappa_i,h_i,\beta_i,\delta_i)$
- $c_i\sim U[1,3]$: cost of performing one task (time, battery, data)
- $\kappa_i\sim U[0.5,2]$ per km for taxis and $U[1,4]$ for pedestrians: detour cost
- behavioural type:
  - **honest**: $\beta_i=0$, $\delta_i=0$;
  - **loyal** (20% of users): home app $h_i$ and loyalty bonus $\beta_i=1$, reports honestly;
  - **strategic** (0, 25 or 50% of users): $\beta_i=0$, reports inflated by $\delta_i=\delta\in\{0.5,1\}$.

## 2. The stage game played every round (leader-follower)

**Stage 1: platforms (leaders) post bounties within budget.**
- Each platform has a candidate bounty $\tilde R_{jg}(t)$ from its pricing rule.
- It posts tasks only on cells that some active user can reach, $\mathcal C(t)=\cup_i\mathcal G_i(t)$.
- Cells are taken in decreasing order of $w_{jg}$, while the total posted bounty stays within the hourly budget. The posted set $S_j(t)$ is the longest prefix $g_1,g_2,\dots$ with $\sum_{k}\tilde R_{jg_k}\le B_j$.
- $R_{jg}=\tilde R_{jg}$ if $g\in S_j(t)$, else no task.
- Since a platform pays at most the posted bounty per cell, spending can never exceed $B_j$.

The pricing rule (mechanism) is one of:

| Mechanism | Rule |
|-|-|
| Fixed price (baseline) | $R_{jg}=\bar m\,w_{jg}\,b$ with constant $\bar m$ |
| Q-learning price | $R_{jg}(t)=m_j(t)\,w_{jg}\,b$, where $m_j(t)\in\mathcal M$ is a learned price level |
| Rival-offer learning | per-block update driven by user reports (Section 5) |

**Stage 2: users (followers) choose one task.**
- Action set: $A_i(t)=\{(j,g):j\in\mathcal J,\ g\in\mathcal G_i(t)\cap S_j(t)\}\cup\{\varnothing\}$.
- **One-task assumption.** A user performs at most one task per round. This follows PACMAB (Dongare et al.): a task costs time and energy, so a user can do only one per time step. When two platforms request the same data, we assume an exclusivity clause forbids handing the same reading to a rival.
- Load: $n_{jg}(a)=|\{i:a_i=(j,g)\}|$. A block's bounty is split equally among the users who sensed it for that platform.

**Utilities.**

$$u_i(a)=\begin{cases}\dfrac{R_{jg}}{n_{jg}(a)}-c_i-\kappa_i d_i(g)+\beta_i\mathbf 1[j=h_i] & a_i=(j,g)\\[4pt] 0 & a_i=\varnothing\end{cases}$$

$$\pi_j(a)=\sum_{g}w_{jg}\,V\,q\big(n_{jg}(a)\big)-\sum_g R_{jg}\,\mathbf 1[n_{jg}(a)\ge1]$$

$$W(a)=\sum_j\sum_g w_{jg}Vq(n_{jg})-\sum_{i:a_i\neq\varnothing}\big(c_i+\kappa_i d_i(a_i)\big)\qquad\text{(payments are transfers and cancel)}$$

## 3. Equilibrium of the users' game (E1)

**Proposition 1.** For fixed bounties $R$, the users' game is an exact potential game with

$$\Phi(a)=\sum_{(j,g)}\sum_{\ell=1}^{n_{jg}(a)}\frac{R_{jg}}{\ell}\;-\;\sum_{i:a_i\neq\varnothing}\big(c_i+\kappa_i d_i(a_i)-\beta_i\mathbf 1[j(a_i)=h_i]\big).$$

(The loyalty term depends only on user $i$'s own action, so it enters $\Phi$ the same way as the costs.)

*Proof.* Let user $i$ move from $e=(j,g)$ to $e'=(j',g')$, all others fixed.
- Load on $e$ falls from $n_e$ to $n_e-1$, so the first sum loses $R_e/n_e$.
- Load on $e'$ rises from $n_{e'}$ to $n_{e'}+1$, so it gains $R_{e'}/(n_{e'}+1)$.
- Only $i$'s private cost changes in the second sum.

So $\Delta\Phi=\frac{R_{e'}}{n_{e'}+1}-\frac{R_e}{n_e}-[c_i+\kappa_i d_i(g')]+[c_i+\kappa_i d_i(g)]=\Delta u_i$. Moves to or from $\varnothing$ work the same way with one sum unchanged. $\square$

**Consequences (Lectures 7 and 9).**
1. A pure Nash equilibrium exists: the maximizer of $\Phi$.
2. Asynchronous best response, and BR with inertia, reach a pure NE in finite time.
3. Under log-linear learning, the stationary distribution is $\propto e^{\Phi/T}$, so it selects potential maximizers as $T\to0$.

**Equilibrium selection.** The NE is generally not unique (about 6 per small instance in our tests). We fix the users' response $a^*(R)$ as the NE reached by asynchronous best response from the empty profile.

**Coverage threshold (toy).** One platform, blocks A and B, two users, zero cost. B is covered in equilibrium iff $R_B>R_A/2$.

## 4. Platforms' pricing game (E2) and learning

**Static price game.** $G^P=(\mathcal J,\mathcal M^M,\Pi)$, where

$$\Pi_j(m)=\frac1{168}\sum_{h=0}^{167}\pi_j\big(a^*_h(R(m))\big).$$

$\Pi_j$ is the average over the 168 real hours of the week, at the base budget. Price grids:
- $M=2$: $\{0.4,0.6,\dots,1.6\}$
- $M=3$: $\{0.6,0.8,\dots,1.4\}$
- $M=4$: $\{0.6,0.9,1.2,1.5\}$

Benchmarks:
- Nash: $m^N$, a pure NE if one exists. Otherwise a mixed NE, computed only for $M=2$ (support enumeration). Joint profit $\Pi^N=\sum_j\Pi_j(m^N)$.
- Collusive: $m^C=\arg\max_m\sum_j\Pi_j(m)$, with joint profit $\Pi^C$.

Platforms are buyers of data, so competition pushes bounties **up** and collusion keeps them **down**.

**Repeated game with Q-learning (Calvano et al. 2020).**
- State: $s_t=(m_1(t-1),\dots,m_M(t-1))$, i.e. one-period memory ($|\mathcal M|^M$ states). Each round replays a random hour of the week.
- Action: own $m_j(t)$.
- Reward: realised $\pi_j$.

$$Q_j(s,m)\leftarrow(1-\alpha)Q_j(s,m)+\alpha\Big[\pi_j+\gamma\max_{m'}Q_j(s',m')\Big]$$

- Parameters: $\alpha=0.15$, $\gamma=0.95$; exploration $\varepsilon_t=e^{-\beta t}$ with $\beta=10^{-5}$.
- Initialization: $Q_0(s,m)=\frac1{|\mathcal M|}\sum_{m'}\Pi_j(m,m')/(1-\gamma)$.
- Convergence: greedy policies unchanged for 50,000 rounds once $\varepsilon<0.01$.

**Collusion index.** Let $\bar\Pi$ be the average joint profit on the cycle the learned greedy strategies settle into.

$$\Delta=\frac{\bar\Pi-\Pi^N}{\Pi^C-\Pi^N}\qquad(0=\text{competitive},\ 1=\text{fully collusive})$$

**Punishment test.** Force one platform to deviate for one round to its static best response (the price that steals users), then let both follow their learned strategies. Check whether:
- the rival retaliates by raising pay (a bounty war), and
- the deviation was unprofitable over the following rounds.

## 5. Reporting game (E3): rival-offer learning with misreporting

**Reports.** After round $t$, a user with $a_i=(k,g)$ who earned $x_i=R_{kg}/n_{kg}$ sends every rejected platform $j\neq k$ a report

$$r_{i\to j}=(1+\delta_i)\,x_i .$$

PACMAB assumes $\delta_i=0$.

**Platform rule (simplified PACMAB, per block).** Let $\hat o_{jg}$ be platform $j$'s estimate of the best per-person pay a rival offers at $g$. Let $\bar R_{jg}=w_{jg}V q(1)$ be the value of one reading (break-even cap) and $\underline R_{jg}=0.2\,w_{jg}b$ the floor.

| Event at block $g$ in round $t$ | Update |
|-|-|
| received reports (lost users to a rival) | $\hat o_{jg}\leftarrow(1-\eta)\hat o_{jg}+\eta\max_i r_{i\to j,g}$, then $R_{jg}\leftarrow\min(\hat o_{jg}+\lambda,\ \bar R_{jg})$ |
| won users, received no reports | $R_{jg}\leftarrow\max((1-\zeta)R_{jg},\ \underline R_{jg})$ |
| nobody came, no reports | $R_{jg}\leftarrow\min((1+\zeta)R_{jg},\ \bar R_{jg})$ |

Parameters: $\eta=0.2$, $\lambda=0.25$, $\zeta=0.02$, start at $m=1$. Variant: PACMAB's $\hat o\leftarrow\max(\hat o,r)$.

**Proposition 2 (truthful reporting is not incentive compatible).**
- A report cannot change the payment already agreed, so in a single round every report is a best response. Truth-telling is weakly, but not strictly, dominant.
- Under the rule above, $R_{jg}(t+1)$ is non-decreasing in the reports received at $g$. So inflating ($\delta_i>0$) weakly raises future bounties on blocks the user visits, and lying is weakly better in the repeated game.
- The higher bounty is shared by everyone who senses $g$. Lying is therefore a private action with a *public* benefit: honest users free-ride, as in the Lecture 3 public goods game with a trusting mechanism.

**Measured.** Liar shares of 0%, 25% and 50% (nested sets, same people), with $\delta=0.5$:
- platform spending and profit;
- net earnings of the same people across scenarios (liars vs never-liars);
- coverage.

## 6. Coverage bonus (RQ3)
Each app reserves a share $\phi=0.25$ of $B_j$ for low-traffic cells.
- On those cells it posts $\tilde R_{jg}=(1+\text{bonus})\,m\,w_{jg}b$ for $g\in L$, highest weight first, until the reserve is used.
- The rest of the budget is posted as in Section 2.
- The total budget is unchanged, so bonus levels are compared at equal cost.

## 7. Metrics
| Metric | Definition |
|-|-|
| Coverage | share of all cells with at least one reading in the hour; also the count of cells covered |
| Low-traffic coverage | the same, restricted to $L$ |
| Spending, profit | $\sum_g R_{jg}\mathbf 1[n_{jg}\ge1]$ and $\pi_j$, per hour |
| Coverage per unit budget (RQ4) | cells covered per hour per 100 units spent (also for low-traffic cells) |
| Task completion | completed posted tasks / posted tasks, i.e. $\sum_{j,t}|\{g\in S_j(t):n_{jg}\ge1\}| / \sum_{j,t}|S_j(t)|$ |
| Net user earnings | $\sum_t u_i(a(t))$ over the measurement window, per user and per type |
| Payment inequality | Gini coefficient of net earnings over all 300 users (users who never earn count as 0) |
| Effect of misreporting (RQ1) | % change against the run with 0% strategic users and the same seed, i.e. the same routes, costs and loyal users |
