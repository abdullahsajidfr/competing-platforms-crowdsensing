# D1 results report: full setup on real Beijing traces

Files:
- `simulation_full.ipynb`: executed notebook, about 11 minutes on 8 cores.
- `formulation.md`: the game.
- `assumptions.md`: every modelling choice.
- `results/full_figs/`: figures.
- The earlier synthetic pilot is kept in `simulation.ipynb` / `report_pilot.md`.

## 1. Setup (as in the proposal)
| Proposal | D1 implementation |
|-|-|
| Beijing, 500 m cells, hourly rounds | 36 × 38 = 1,368 cells over central Beijing; one real week replayed hour by hour |
| Several hundred users on T-Drive / GeoLife | 200 T-Drive taxis + 100 GeoLife walkers; 121 active per hour on average (16 at night, 232 at peak) |
| 2 to 4 platforms with budgets and cell weights | 4 apps (air quality, traffic, noise, weather); M = 2, 3, 4; hourly budgets of 150 / 300 / 600 |
| Honest, loyal, strategic users | 20% loyal; 0 / 25 / 50% strategic; the rest honest |
| Three core methods | fixed posted price, Q-learning pricing, rival-offer learning (PACMAB-inspired) |
| Metrics | user earnings, Gini, spending, profit, task completion, payment trends, overall and low-traffic coverage |
| Mesa | `UserAgent`, `PlatformAgent`, `CrowdsensingModel(mesa.Model)` |
| 10 seeds; truthful and lying runs share routes, costs and seeds | yes; strategic sets are nested, so the same people are compared |

**Check:** across all 168 hours, every user was at a best response after the hourly dynamics (0 users could gain by switching; 3.3 sweeps on average). This confirms Proposition 1 on real data.

## 2. Answers to the research questions

### RQ1 (main): Do false rival offers raise users' net earnings or platform spending?
**Yes, both, in every seed and every setting tested.**

Main setting (budget 300, liars report 1.5×), % change against the same seeds with no liars:

| | M = 2, 25% liars | M = 2, 50% | M = 3, 50% | M = 4, 50% |
|-|-|-|-|-|
| Platform spending | +3.2% | **+5.1%** | +4.2% | +4.1% |
| Platform profit | −4.0% | **−6.9%** | −10.2% | −11.6% |
| User earnings (all users) | +7.4% | **+11.9%** | +6.5% | +6.3% |
| Honest users | +7.3% | +11.3% | +6.7% | +6.4% |
| Liars | +9.2% | +12.8% | +6.7% | +6.6% |
| Cells covered | −2.3% | −4.0% | −4.2% | −3.9% |

Spending rose in 10/10 seeds and honest users' earnings rose in 10/10 seeds, for every M and share.

**What it means:**
- **Users gain, platforms lose.** Inflated reports push up the bounties of the apps a user turned down. Platform profit falls more than spending rises, because the extra money buys no extra data.
- **Lying helps everyone on the block, not just the liar.** With 25% liars (M = 2), a liar gains 20.2 over three weeks and an honest user 15.2. So liars keep a small personal edge, but most of the benefit goes to people who did not lie. That is the public-goods logic of Proposition 2.
- **Coverage falls slightly (about 4%).** With fixed budgets, higher bounties on contested cells mean fewer tasks can be posted elsewhere. This is a side effect the proposal did not anticipate.

Sensitivity at M = 2, 50% liars:
- A bigger lie (2×) makes it worse: spending +6.8%, profit −9.7%, honest users +15.4%.
- Tight budgets limit the damage to profit (−3.1%).
- PACMAB's "keep the maximum" update already overpays with honest users (spending 331 vs 314 per hour), so lying adds less on top (+2.6%).

### RQ2: Do learning platforms raise or lower payments over time?
**It depends on the method and on how many apps compete.**

**Q-learning (price levels after learning, 10 seeds):**

| | Q-learning | Myopic learners (competitive reference) | Collusive | Δ vs myopic |
|-|-|-|-|-|
| M = 2 | **0.75** (Nash benchmark 0.82) | 0.83 | 0.70 | 0.59 (Δ vs Nash: 0.58) |
| M = 3 | 0.99 | 1.17 | 0.67 | 0.32 |
| M = 4 | 1.01 | 1.23 | 0.68 | 0.41 |

- Q-learners always pay **less** than competitive (myopic) learners. User earnings are 14–24% lower.
- With 2 apps, pay falls below the Nash level and Δ ≈ 0.58. With 3–4 apps the drift is weaker (Δ 0.3–0.4): more competitors make keeping pay low harder.
- **Not proven to be collusion.** With M = 2, *memoryless* learners, which cannot punish, end exactly at the collusive prices (Δ = 1.0). So low pay comes from the learning process, not from reward-and-punish strategies. This matches the proposal's caution that falling payments alone do not establish collusion.

**Rival-offer learning (posted price level over five weeks, honest users):**
- With 2 apps, pay settles **below** the fixed price (about 0.85, with a daily cycle).
- With 3 or 4 apps it **climbs to about 1.5–1.6**: each lost user triggers an outbid, and with more rivals that becomes a bidding war.

### RQ3: Are low-traffic areas left uncovered? Can a bonus help?
**Yes, they are left uncovered; yes, a modest bonus helps a lot.**
- Under fixed pricing (M = 2, base budget), only **0.7 of the 342 low-traffic cells** are covered in an average hour (0.2%), compared with 80 of the 1,368 cells overall (5.8%). More money doesn't fix this: at the loose budget it is still 0.3%. The low-traffic cells are mostly at the edge of the study area (Fig 1).
- Reserving 25% of the budget for low-traffic cells, with the same total budget:

| Bonus on low-traffic cells | 0% | +50% | +100% | +200% | +300% |
|-|-|-|-|-|-|
| Low-traffic cells covered per hour | 0.7 | **13.6** | 12.6 | 9.9 | 7.4 |
| All cells covered per hour | 79.8 | 77.3 | 75.4 | 73.1 | 72.2 |
| Platform profit per hour | 377 | 344 | 317 | 288 | 278 |

- A **+50% bonus covers 18× more low-traffic cells**, at a cost of 3% of overall coverage and 9% of profit.
- **Bigger bonuses do worse.** With a fixed reserve, a higher bonus per cell means fewer cells can be posted. The best bonus is small.

### RQ4: Which core method gives the best coverage per unit budget?
Setup: equal hourly budget (300 per app), honest and loyal users.

| | Cells covered per 100 spent | Low-traffic cells per 100 | User earnings per active hour | Gini |
|-|-|-|-|-|
| M = 2: fixed | 22.7 | **0.21** | **1.40** | **0.37** |
| M = 2: Q-learning | **31.0** | 0.20 | 0.80 | 0.45 |
| M = 2: rival-offer | 26.3 | 0.15 | 1.09 | 0.40 |
| M = 3: fixed | 19.4 | **0.17** | 1.62 | 0.35 |
| M = 3: Q-learning | **20.7** | 0.08 | 1.55 | 0.35 |
| M = 3: rival-offer | 15.6 | 0.00 | 2.17 | 0.42 |
| M = 4: fixed | 20.3 | 1.57 | 1.66 | 0.35 |
| M = 4: Q-learning | **21.9** | **2.17** | 1.54 | 0.35 |
| M = 4: rival-offer | 15.7 | 1.96 | 2.43 | 0.37 |

- **For the platforms, Q-learning gives the most coverage per unit budget at every M.** It covers about as many cells as the other methods while paying less. The cost falls on users: with 2 apps they earn 43% less than under the fixed price, and earnings are more unequal (Gini 0.45 vs 0.37).
- **Rival-offer learning is efficient with 2 apps but wasteful with 3–4,** because of the bidding war under RQ2.
- **No core method reaches low-traffic cells.** Fixed price plus a +50% bonus (RQ3) gets about 3.8 low-traffic cells per 100 spent, about 18× the best M = 2 core method without a bonus (0.21).

## 3. Limitations
- **The T-Drive week is Spring Festival 2008.** Traffic is unusual, and there are no taxi traces from Thursday evening to Friday noon.
- **GeoLife walks come from 2007–2012** and are overlaid on that week by weekday and hour.
- **The four app weightings are proxies** built from taxi and pedestrian activity, not land-use data.
- **Q-learning uses one fixed user population** and a coarse price grid for M = 3, 4. The Nash benchmark exists only for M = 2; for M = 3, 4 the competitive reference is the myopic learner.
- **Strategic users inflate by a fixed factor**; they don't learn how much to lie.
- **MSensing and deferred acceptance are not implemented** (optional in the proposal).

## 4. Next steps
1. **RQ1 defences.** Ignore reports above a percentile, cap report growth, or spot-check reports; then measure whether lying stops paying.
2. **Strategic users who learn δ.** Use a bandit over inflation levels and check whether lying spreads given the free-riding.
3. **RQ2.** Rerun Q-learning with expected (noise-free) profits and finer grids, and run the punishment test from the pilot on real data.
4. **RQ3.** Tune the reserve share and bonus together, and add the bonus to the learning methods.
5. **RQ4.** Add MSensing and deferred acceptance if time allows; test a second T-Drive-like week if one can be found.
6. **Final demo (D4).** An animated coverage map of Beijing, round by round.
