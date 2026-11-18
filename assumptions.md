# Assumptions made for the D1 full setup

These are choices the proposal does not pin down. Each one is a parameter in the code, so it can be changed and rerun.

## A. Data and city
| # | Assumption | Why |
|-|-|-|
| A1 | **Area:** a 17.8 km × 18.7 km box over central Beijing (lat 39.83–39.99, lon 116.27–116.49, roughly the 2nd to 4th Ring), cut into 36 × 38 = 1,368 cells of 500 m. | Most T-Drive and GeoLife activity is here; the full city would be mostly empty cells. |
| A2 | **Taxis:** 200 T-Drive taxis, drawn at random (seed 0) from those with at least 40 active hours inside the box during the week of 2–8 Feb 2008. | "Several hundred users". Taxis that are rarely in the box add little. |
| A3 | **Pedestrians:** the 100 GeoLife users with the most walking hours in the box. An hour of data counts as walking if the median speed between GPS fixes is at most 6 km/h. | GeoLife has no full mode labels, and the proposal says "walking segments". |
| A4 | **One replayed week.** T-Drive covers one week; GeoLife spans 2007–2012. Each GeoLife walk is placed by weekday and hour (Beijing time; GeoLife's GMT + 8 h). For each weekly slot, a pedestrian's first recorded walk is used. | Overlay by time of day, not calendar date, as planned. |
| A4b | **The T-Drive week (2–8 Feb 2008) includes Chinese New Year's Eve (6 Feb) and Spring Festival (7 Feb).** Traffic on those days is unusual (many people leave the city), and the release has no taxi traces from Thursday about 17:00 to Friday about 12:00 (hours 113–132 of the week). We do not correct for this. | This is the only week T-Drive provides. Worth stating as a limitation. |
| A5 | **Routes:** the cells a user's GPS passes through in that hour. Taxi fixes are about 3 minutes apart, so the path is filled in every 100 m between fixes less than 10 minutes apart. Jumps faster than 120 km/h are dropped as GPS errors. | Without filling in, taxis would "skip" cells they drove through. |
| A6 | **Availability:** a user can work only in hours where its trace has data. Otherwise it is inactive. | We only know where people are when they are recorded. |
| A7 | **Reach:** a user can sense any cell on its route or one cell (500 m) away. The detour is 0 km on the route and 0.5 km otherwise. | The same rule as the pilot. |
| A8 | **Low-traffic cells:** the 25% of cells that users pass near least often, among cells that someone passes near at least once in the week. Cells nobody ever passes near would count as "unreachable", not "low-traffic". With the real traces there are none: every one of the 1,368 cells is passed near at least once in the week. That leaves 342 low-traffic cells. | No incentive can get data from a cell no user ever comes near; counting those would make every method look equally bad. |

## B. Apps (platforms)
| # | Assumption | Why |
|-|-|-|
| B1 | **Four apps**, used in this order for M = 2, 3, 4: air quality, traffic, noise, weather. | The proposal names air quality and traffic; noise and weather are natural third and fourth. |
| B2 | **Cell weights come from observed activity**, since we have no land-use map. Using log-normalised taxi activity $\tilde T$ and pedestrian activity $\tilde P$: air quality $0.2+0.8\tilde P$, traffic $0.1+0.9\tilde T$, noise $0.2+0.4(\tilde T+\tilde P)$, weather $0.5$ (uniform). | Where people walk stands in for residential/pedestrian areas; where taxis drive stands in for roads. |
| B3 | **Hourly budget**, equal for every app: tight 150, base 300, loose 600 per hour. With two apps, the tight budget posts about 70 tasks per hour for about 120 active users, base about 150, and loose about 310 (calibration run, fixed price m = 1). So budgets bind at tight and base and only loosely at loose. | The proposal has budgets but no values. These levels bracket "binding" and "rarely binding". |
| B4 | **Posting rule:** each hour an app sees which cells active users can reach (apps know users' locations). It posts tasks on those cells, highest weight first, until the total posted bounty would exceed its budget. Spending therefore can never exceed the budget. | Keeps every hour's game a potential game and guarantees the budget. It works like PACMAB's task quotas. |
| B5 | **Money scale:** base reward $b=5$, value of a fully covered weight-1 cell $V=20$, reading quality $\rho=0.5$. A platform never pays more than one reading is worth to it ($w_{jg}V\rho$). | Same as the pilot, so results stay comparable. |
| B6 | **Q-learning:** one-period memory of all apps' prices. $\alpha=0.15$, $\gamma=0.95$, $\varepsilon_t=e^{-10^{-5}t}$, Calvano initialisation. Each round replays a random hour of the week. Up to 1.5 million rounds; "converged" means the greedy strategy is unchanged for 25,000 rounds. Price grids: 7 levels (M = 2), 5 (M = 3), 4 (M = 4). Learning uses one fixed user population (seed 0). | The number of states grows as (levels)^M, so the grid shrinks as M grows. The hour-to-hour noise makes a stricter convergence rule impractical. |
| B7 | **Competitive benchmark for M = 3, 4:** no pure Nash equilibrium exists, and computing a mixed one for 3–4 players is out of scope. "Competitive" is therefore the profit of myopic learners ($\gamma=0$), and Δ is also reported relative to them. | Gives every M a reference point. |
| B8 | **Rival-offer learning:** moving average $\eta=0.2$, outbid margin $\lambda=0.25$, adjustment $\zeta=2\%$ per hour, start at price level 1. PACMAB's own "keep the maximum" update is a sensitivity run. | The same as the pilot. |

## C. Users
| # | Assumption | Why |
|-|-|-|
| C1 | **Costs:** task cost $c_i\sim U[1,3]$. Detour cost per km: $U[0.5,2]$ for taxis and $U[1,4]$ for pedestrians. | Walking a detour costs more time than driving one. |
| C2 | **Types:** 20% loyal (random home app, loyalty bonus $\beta=1$, honest reports). Strategic users are 0%, 25% or 50% of all users, drawn from the non-loyal users; the rest are honest. | The proposal lists the three types but not the shares. 50% strategic plus 20% loyal leaves 30% honest. |
| C3 | **Nested types:** within a seed, the 25% strategic users are a subset of the 50% strategic users, and loyal users are the same in every run. Truthful and lying runs share routes, costs and seeds. | Lets us compare the same people across scenarios, as the proposal requires. |
| C4 | **Reports:** a user who works for app $k$ reports to every app whose task it could have taken that hour (each rejected app): honest users report the pay they got, strategic users report $(1+\delta)$ times it, with $\delta=0.5$ (main) or 1.0. | PACMAB's feedback rule, with misreporting added. |
| C5 | **Equilibrium selection:** each hour, users settle by asynchronous best response from "nobody working", in random order (the notebook checks that the result is a Nash equilibrium). | Results depend on which equilibrium is reached, so the rule is fixed. |

## D. Experiments and measurement
| # | Assumption | Why |
|-|-|-|
| D1 | 10 seeds per setting. Seeds change user costs and types, the update order and learning randomness, not the routes (those are the real data). | The proposal's 10 seeds. |
| D2 | Run lengths: baseline and bonus 2 weeks; lying runs 5 weeks with the first 2 discarded as burn-in; Q-learning strategies evaluated for 2 weeks inside the Mesa model. | Rival-offer prices settle within about a week. |
| D3 | **Coverage per unit budget (RQ4)** = cells covered per hour per 100 units spent. All methods get the same hourly budget (base). | The proposal's "coverage per unit budget" at equal budgets. |
| D4 | **Bonus (RQ3):** each app reserves 25% of its budget for low-traffic cells and pays a bonus of 0–300% there; the total budget is unchanged. | Compares bonus levels at equal cost. |
| D5 | **Gini** of net earnings over all 300 users in the measurement window. Users who never earn count as 0. | Excluding non-earners hides exclusion. |
| D6 | **Task completion** = completed posted tasks / posted tasks, summed over apps and hours. | The proposal lists task completion. |

## E. Still not in D1
- MSensing and deferred acceptance (optional in the proposal).
- Strategic users who *learn* how much to lie, rather than using a fixed δ.
- Budgets that change over time (per day or per week rather than per hour).
