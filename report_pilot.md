# D1 results report: Competing Platforms in Mobile Crowdsensing

Files:
- `formulation.md`: the game, written formally.
- `simulation.ipynb`: all code and outputs (runtime about 5 minutes).
- `results/notebook_figs/`: the figures.

All numbers below are averages over 10 random seeds unless stated otherwise.

## 1. What we simulated

Setting:
- A 10×10-block city with 50 users and two apps that post bounties on blocks: P1 (air quality, values residential blocks) and P2 (traffic, values road blocks).
- Every hour, each user accepts **one task** (one app, one block) or sits out. This follows PACMAB's one-task-per-time-step assumption. Where both apps want the same data, we assume an exclusivity clause.
- Users share a block's bounty equally, so a crowded block pays less per person.

**The two-level game:**
- **Users (followers).** For fixed bounties they play a congestion game, which is a potential game (Proposition 1). A pure equilibrium always exists and best-response play reaches it.
- **Apps (leaders).** They set bounties ahead of the users, using one of three rules:
  - fixed price (baseline);
  - Q-learning price (Experiment 2);
  - rival-offer learning from user reports, where some users lie (Experiment 3).

## 2. Checks on the theory (Section 0 of the notebook)

| Check | Result |
|-|-|
| Potential identity ΔΦ = Δu | holds to 2.8×10⁻¹⁴ over 5,000 random moves |
| Best response reaches an equilibrium | always: in 2 to 4 sweeps on the full city, and in all 300 small test games |
| Log-linear learning | the lower the temperature, the higher the potential: T = 1.0: Φ = 114.5; T = 0.3: 141.5; T = 0.1: 147.0 (above the BR equilibrium's 145.9) |
| Number of equilibria in small games | 6 on average, up to 24, so we must state a selection rule |
| Price of anarchy (small games) | mean 1.22, worst 2.81; the Lecture 7 "≤ 2" bound does not apply to equal-split pay |

## 3. Experiment 1: baseline (both apps fixed at price level 1.0)

| Metric | Value |
|-|-|
| Blocks covered per hour | 49.8% |
| **Quiet blocks covered** | **6.2%** |
| Profit P1 / P2 | 183.3 / 65.8 |
| Users working for P1 / P2 | 36.7 / 13.2 |
| User net earnings per hour | 2.88 |
| Users' equilibrium welfare vs a greedy central planner | 1.004 (equal, in practice) |

**What it means:**
- With fixed prices, users spread themselves efficiently. Selfish choices do about as well as a planner.
- Coverage is very uneven: quiet blocks are almost never sensed.
- From the earlier price sweep: raising the price uniformly does not fix this.

## 4. Experiment 2: do learning apps learn to keep pay low?

**Benchmarks from the one-shot price game:**
- There is **no pure equilibrium**. The apps keep outbidding each other until one drops back.
- Competitive (mixed Nash) outcome: average price levels 1.00 (P1) and 1.24 (P2), joint profit **212.5**.
- "Secretly agree to keep pay low" (collusive) outcome: both at 0.6, joint profit **335.1**.

**Results after learning:**

| Setting | Price P1 | Price P2 | Joint profit | Δ (0 = competitive, 1 = collusive) | User earnings per hour |
|-|-|-|-|-|-|
| Q-learning (memory of last prices) | 0.76 | 0.88 | 286.7 | **0.61** (range 0.38 to 0.76) | **1.98** |
| Control: myopic (γ = 0) | 0.80 | 1.20 | 241.6 | 0.24 | 2.90 |
| Control: memoryless | 0.60 | 0.94 | 292.4 | 0.65 | 1.74 |

All 30 runs converged, in about 0.5 to 0.9 million simulated hours.

**Punishment test:**
- Right after an app deviates to steal users, the rival raises its bounty in 50% of cases.
- On average, deviating was **unprofitable** over the next 15 rounds (net −13.4 for P1, −15.3 for P2).
- But it was profitable in some seeds.
- In the clearest example (seed 0, P2 deviates), P1 hits back with a two-round bounty war (pay rises to 1.2) and then both return to their old low-pay cycle. That is the classic punish-then-forgive pattern.

**What it means:**
1. **Learning apps do drift towards "keep pay low".** They capture about 61% of the gap between the competitive and collusive profit. Users earn about **31% less** per hour than under competitive or fixed pricing (1.98 vs 2.88 to 2.90). That is in the 70 to 90% range Calvano et al. report, though a bit lower.
2. **It is only partly genuine collusion.**
   - The memoryless control can't see the rival's price, so it can't punish. Yet it reaches the same Δ (0.65).
   - Only the myopic control (apps that ignore the future) stays near competitive (Δ = 0.24).
   - So most of the low pay comes from *forward-looking learning* in a game with no stable price, not from reward-and-punish strategies.
   - The punishment test shows some enforcement, but not consistently.
   - This is an honest, defensible finding, and it is exactly the "falling pay alone doesn't prove collusion" caveat from our proposal.
3. **Coverage is unaffected.** Coverage stays at 49% and quiet-block coverage at 8%. Lower pay hurts users, not coverage, because users are still willing to work at these prices.

## 5. Experiment 3: users who lie about the rival's offer

Setup:
- Both apps use rival-offer learning: when an app loses a user, it raises its bounty on that block to just above what the user says the rival paid.
- Liars report 1.5× the true pay.
- We follow the **same people** across the 0%, 25% and 50% scenarios. Comparing different people would be misleading, because liars and honest users differ in location and costs.

**Changes relative to the 0%-liar world** (all 10 seeds point the same way unless noted):

| | 25% liars | 50% liars |
|-|-|-|
| P2 (traffic) spending | **+5.2%** | **+13.3%** |
| P1 (air quality) spending | +0.5% | +1.6% |
| P1 profit | −0.9% | −2.3% |
| P2 profit | +1.1% | −0.9% |
| Earnings, group A (liar in both runs) | +3.4% | +9.4% |
| Earnings, group B (liar only in the 50% run) | +4.1% | +12.1% |
| Earnings, **group C (always honest)** | **+3.8%** | **+9.6%** |
| Coverage | +0.9% | +1.8% |
| Quiet-block coverage | no change | no change |

Sensitivity checks (5 seeds, 50% liars vs 0%):

| Variant | Total app spending | Total app profit |
|-|-|-|
| Main rule, liars report 1.5× | +6.2% | −1.6% |
| Bigger lie (report 2×) | +10.5% | −4.1% |
| PACMAB's own "keep the max" update | +8.5% | −2.8% |

**What it means:**
1. **Users gain and the apps pay.** Every user group earns more as more people lie. The money comes from the apps, mainly from P2's extra spending. Total app profit falls.
2. **Lying is a public good, so honest users free-ride.** Honest users gain as much as liars, in percentage terms. A lie raises the bounty on a whole block, and everyone who works there shares it. So a liar gains almost nothing *compared with* the honest users next to them. This is the Lecture 3 public goods logic: the incentive to lie is positive but weak, and spread thin.
3. **The app you turn down is the one that gets hurt.** Reports only go to the rejected app. Most users work for P1, so most reports land at P2, and its spending rises 13% at 50% liars. P1 still loses profit, because P2's higher bounties pull users away from it.
4. **PACMAB's "keep the maximum" update is more exposed than a moving average.** One inflated report raises its estimate permanently. That's a concrete design lesson.
5. **The damage is capped.** An app never pays more than one reading is worth to it, so lying can only push spending up to that cap. Bigger lies cause more damage (2× lies: profit −4.1%).
6. **Side finding (RQ3).** With per-block learning, quiet-block coverage is **31%**, compared with 6% at a fixed price. That's because apps slowly raise the bounty on blocks nobody visits. Per-block pricing fixes coverage gaps that a single price level cannot.

## 6. Limitations to state honestly
- The city and movement are synthetic, and the parameters are chosen, not calibrated to Beijing data.
- When users have several possible equilibria, we pick one with a fixed rule (best response from the empty profile).
- The Q-learning profits use a bank of 200 hourly route draws, i.e. a week of "traces" replayed.
- The Δ results come from one price grid ({0.4, …, 1.6}) and one set of learning parameters.
- In Experiment 3, liars inflate by a fixed factor. They don't strategically decide when or how much to lie.

## 7. What we can do next
1. **Pin down the collusion question (RQ2).**
   - Rerun with deterministic (average) profits to test whether the noise from random hours drives the result.
   - Try slower exploration, more seeds and a finer price grid.
   - Run a proper best-response check on the learned strategies (Calvano's Q-loss).
   - If memoryless agents still reach low pay, the story is "learning in a game without a stable price produces low pay", which is a result in itself.
2. **Defences against lying (RQ1).** Test whether simple app-side fixes remove the overspending without losing users:
   - ignore reports above a percentile;
   - average reports instead of taking the max;
   - spot-check reports.
3. **Strategic liars.** Let users *learn* how much to inflate (a bandit over δ). Check whether lying spreads or dies out, given the free-riding.
4. **Coverage (RQ3).**
   - Measure the smallest quiet-block bonus that achieves a target coverage, which generalises the toy threshold R_B > R_A/2.
   - Compare equal-split pay with marginal-contribution pay (Lecture 7: PoS = 1, PoA ≤ 2).
5. **Robustness.**
   - More users (N = 100 to 150), so blocks get shared and data-quality returns matter.
   - Three apps.
   - Loyal users.
   - The send-to-both variant (where competition largely vanishes).
6. **Real data.** Map one day of T-Drive taxi traces onto 500 m blocks to replace the synthetic routes.

## 8. Three-sentence summary for the presentation
- Users' task choice is a potential game, so a stable outcome always exists and simple learning reaches it. But with fixed prices quiet areas are left out (6% coverage), and per-block adaptive pricing raises that to 31%.
- Apps that learn their prices pay users about 31% less than competition would (Δ ≈ 0.6). Our controls show this is only partly true collusion: forward-looking learning alone produces most of it.
- Users who exaggerate the rival's offer raise everyone's pay, honest users included, at the expense of the app they turned down (up to +13% spending at 50% liars). Lying is a public good with a weak private incentive.

## 9. Answers to the proposal's research questions (D1 status)

| RQ | Question | Answer so far | Strength |
|-|-|-|-|
| RQ1 | Do false rival offers raise users' net earnings or platform spending? | **Yes, both.** User earnings rise 3–4% (25% liars) and 9–12% (50% liars). The rejected app's spending rises 5% / 13%; total app spending rises 6% (lie = 1.5×) to 10.5% (lie = 2×). Honest users gain as much as liars, so lying is a shared benefit, not a private edge. | Supported in all 10 paired seeds. Proposition 2 proves lying can't hurt the liar under this rule. |
| RQ2 | Do learning platforms raise or lower payments over time? | **Lower.** Q-learning apps drift from about 1.0 down to 0.76 / 0.88, below the competitive benchmark (1.00 / 1.24); users earn 31% less. Rival-offer learning also spends less than the fixed price (209 vs 249 per hour). Collusion is **not** established: memoryless apps, which can't punish, reach the same low pay. | Pay falling: supported (10/10 seeds). Collusion: not supported, which matches the proposal's caution. |
| RQ3 | Are low-traffic areas left uncovered? Can a bonus help? | **Yes, they are left uncovered:** 6% under any uniform fixed price, 8% under Q-learning. **A bonus helps in theory** (toy threshold R_B > R_A/2, proven). Per-block learning, which acts like an automatic bonus, raises coverage to 31%. | First part: supported. Second part: indirect; an explicit bonus experiment is still to run. |
| RQ4 | Which core method gives the best coverage per unit budget? | **Not tested yet** (budgets not equalised). Early hint: rival-offer learning matches fixed-price overall coverage (47% vs 50%) with 16% less spending and 5× the quiet-area coverage. | Preliminary only. |
