"""Builds simulation.ipynb (nbformat v4 JSON, no dependencies)."""
import json

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

formulation = open("formulation.md").read()

md(r"""
# Competing Platforms in Mobile Crowdsensing: D1 simulation

Abdullah Sajid, Mishal Sagar (LUMS, EE567/CS6315 Multiagent Systems)

Two apps (P1 = air quality, P2 = traffic) post bounties on city blocks. Each hour, 50 people choose **one task** (an app and a block) or skip the hour. The notebook runs three experiments:

| # | Experiment | Question |
|-|-|-|
| 1 | **Baseline:** fixed prices | What coverage and profits do we get with no learning at all? |
| 2 | **Learning prices:** Q-learning platforms | Do the apps learn to keep pay low ("tacit collusion") without communicating? |
| 3 | **Lying users:** rival-offer learning with 0/25/50% liars | Who gains and who loses when users inflate what the rival paid them? |

Sections 0 and 1 check the equilibrium results that the experiments rely on.
The full mathematical formulation follows; it is also in `formulation.md`.
""")
md(formulation)

md("## Setup")
code(r"""
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from crowdsim.env import Config, World, UserGame, platform_profit
from crowdsim.dynamics import async_best_response, br_with_inertia, log_linear
from crowdsim.equilibrium import random_tiny_instance, brute_force
from crowdsim.model import CrowdsensingModel, FixedPrice
from crowdsim.learning import (build_bank, benchmarks, run_qlearning, limit_outcome,
                               impulse_response, RivalOfferModel)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
pd.set_option("display.precision", 3)
cfg = Config()
SEEDS = range(10)
T0 = time.time()
pd.Series({k: v for k, v in vars(cfg).items()}, name="value").to_frame()
""")

md("### The simulated city\nLeft: what each app cares about. Middle: how often users pass through each block. Right: the 25% least-visited blocks ('quiet areas').")
code(r"""
w = World(cfg)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for j, name in enumerate(["P1 air quality weight", "P2 traffic weight"]):
    im = ax[j].imshow(w.W[j].reshape(10, 10), origin="lower", cmap="Greens", vmin=0, vmax=1); ax[j].set_title(name)
im = ax[2].imshow(w.traffic.reshape(10, 10), origin="lower", cmap="magma"); ax[2].set_title("user traffic per block"); plt.colorbar(im, ax=ax[2], fraction=.046)
ax[3].imshow(w.low_traffic.reshape(10, 10), origin="lower", cmap="Blues"); ax[3].set_title("quiet (low-traffic) blocks")
plt.tight_layout(); plt.show()
""")

md("""
## 0. Checks on the users' game (Proposition 1)
1. The potential identity $\\Delta\\Phi=\\Delta u_i$ holds for random unilateral moves.
2. Best-response dynamics reach a pure NE, and $\\Phi$ rises monotonically along the way.
3. Log-linear learning at low temperature finds higher-potential equilibria.
4. Brute force on small games: every instance has a pure NE, BR always finds one, and we compute PoA/PoS.
""")
code(r"""
rng = np.random.default_rng(7)
acts = w.build_actions(w.draw_routes(rng)); R = w.W * cfg.b
g = UserGame(R, acts)
a_br, sweeps, tr_br = async_best_response(g, np.random.default_rng(1), record=True, check_potential=True)
err = 0.0
for _ in range(5000):
    i = rng.integers(g.N); b = a_br.copy(); b[i] = rng.integers(-1, len(acts[i][0]))
    err = max(err, abs((g.potential(b) - g.potential(a_br)) - (g.utility(i, b) - g.utility(i, a_br))))
a_in, steps, tr_in = br_with_inertia(g, np.random.default_rng(2), p_inertia=0.7)
lll = {T: log_linear(g, np.random.default_rng(3), T=T, steps=30000)[1] for T in (1.0, 0.3, 0.1)}
fig, ax = plt.subplots(1, 3, figsize=(13, 3))
ax[0].plot(tr_br); ax[0].set(title=f"Async best response: NE after {sweeps} sweeps", xlabel="single-user updates", ylabel="potential Φ")
ax[1].plot(tr_in, ".-"); ax[1].set(title=f"BR with inertia (p=0.7): NE after {steps} periods", xlabel="period")
for T, tr in lll.items(): ax[2].plot(tr, lw=.7, label=f"T={T}")
ax[2].axhline(tr_br[-1], c="k", ls="--", lw=.8, label="BR NE"); ax[2].legend(); ax[2].set(title="Log-linear learning", xlabel="revisions")
plt.tight_layout(); plt.show()
print(f"max |ΔΦ - Δu| over 5000 random deviations = {err:.1e};  BR profile is NE: {g.is_nash(a_br)};  inertia profile is NE: {g.is_nash(a_in)}")
print("mean Φ under LLL (last 5000 steps):", {T: round(float(np.mean(tr[-5000:])), 1) for T, tr in lll.items()}, " BR NE Φ:", round(tr_br[-1], 1))
""")
code(r"""
rng = np.random.default_rng(11); rows = []
for s in range(300):
    W_, kappa, acts_, R_ = random_tiny_instance(rng)
    gg = UserGame(R_, acts_); ne, opt = brute_force(gg, W_, cfg.V, cfg.rho)
    a, _, _ = async_best_response(gg, np.random.default_rng(s))
    welf = [x[1] for x in ne]
    rows.append(dict(n_NE=len(ne), BR_finds_NE=gg.is_nash(a), PoA=opt / min(welf) if min(welf) > 0 else np.nan,
                     PoS=opt / max(welf) if max(welf) > 0 else np.nan))
bf = pd.DataFrame(rows)
print(f"300 random games (4 users, 3 blocks, 2 apps): every game has a pure NE: {(bf.n_NE >= 1).all()}, BR always reaches a NE: {bf.BR_finds_NE.all()}")
bf[["n_NE", "PoA", "PoS"]].describe().loc[["mean", "50%", "max"]]
""")

md("""
## 1. Experiment 1: baseline (fixed price $m=1$ for both apps)
We run 10 seeds × 500 hours. Each hour, users settle to a NE of the stage-2 game. No one learns.

**Benchmark:** a greedy welfare-maximizing assignment, i.e. what a central planner would roughly achieve.
""")
code(r"""
base_rows, covmaps = [], []
for s in SEEDS:
    mdl = CrowdsensingModel(cfg, [FixedPrice(1.0), FixedPrice(1.0)], seed=s)
    h = pd.DataFrame(mdl.run(500)); h["seed"] = s; base_rows.append(h); covmaps.append(mdl.cover_count / 500)
base = pd.concat(base_rows)
base["welfare_ratio"] = base.welfare / base.welfare_greedy
b_summary = base.groupby("seed")[["coverage", "low_traffic_coverage", "profit0", "profit1", "spend0", "spend1",
                                  "user_net_mean", "users0", "users1", "welfare_ratio"]].mean()
display(b_summary.agg(["mean", "std"]).T.rename(columns={"mean": "mean over seeds", "std": "sd"}))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
im = ax[0].imshow(np.mean(covmaps, 0).reshape(10, 10), origin="lower", vmin=0, vmax=1); plt.colorbar(im, ax=ax[0], fraction=.046)
ax[0].set_title("P(block covered in an hour), baseline")
ax[1].hist(base.welfare_ratio, bins=40); ax[1].set(title="Welfare of users' NE / greedy planner", xlabel="ratio")
plt.tight_layout(); plt.show()
""")

md("""
## 2. Experiment 2: do learning apps drift to 'keep pay low'?

**Step 1: stage-game benchmarks.** For each price pair on $\\mathcal M=\\{0.4,\\dots,1.6\\}$, users' equilibrium outcomes are computed on a bank of 200 hourly route draws. From this we get the static Nash prices (competitive) and the joint-profit prices (collusive).
""")
code(r"""
grid = np.round(np.arange(0.4, 1.61, 0.2), 2)
t = time.time(); bank = build_bank(cfg, grid, K=200); print(f"bank built in {time.time()-t:.0f}s")
bm = benchmarks(bank["Pi"]); Pi = bank["Pi"]; L = len(grid)
print("pure NE:", bm.get("all_pure", "none"), "->", bm["nash_type"], "equilibrium")
if bm["nash_type"] == "mixed":
    x, y = bm["nash"]
    print("  mixed NE  P1:", {grid[i]: round(p, 2) for i, p in enumerate(x) if p > 1e-6}, " P2:", {grid[i]: round(p, 2) for i, p in enumerate(y) if p > 1e-6})
    mN = (x @ grid, y @ grid)
else:
    mN = (grid[bm["nash"][0]], grid[bm["nash"][1]])
c = bm["collusive"]; mC = (grid[c[0]], grid[c[1]])
print(f"competitive (Nash):  mean price levels {mN[0]:.2f}, {mN[1]:.2f};  joint profit Π^N = {bm['pi_N']:.1f}")
print(f"collusive:           price levels {mC[0]:.2f}, {mC[1]:.2f};  joint profit Π^C = {bm['pi_C']:.1f}")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, (M_, ttl) in enumerate([(Pi[..., 0], "Π1 air quality"), (Pi[..., 1], "Π2 traffic"), (Pi.sum(-1), "joint profit")]):
    im = ax[k].imshow(M_, origin="lower"); plt.colorbar(im, ax=ax[k], fraction=.046)
    ax[k].set(xticks=range(L), yticks=range(L), xticklabels=grid, yticklabels=grid, xlabel="m2", ylabel="m1", title=ttl)
    ax[k].plot(range(L), Pi[..., 0].argmax(0), "w^", ms=5, label="P1 best response")
    ax[k].plot(Pi[..., 1].argmax(1), range(L), "r>", ms=5, label="P2 best response")
    ax[k].plot(c[1], c[0], "y*", ms=15, mec="k", label="collusive")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()
""")
md("""
**Step 2: Q-learning.** Two independent Q-learners run for 10 seeds each. Each run lasts until the learned strategies stop changing (up to $10^6$ hours, replaying the 200-hour bank).

**Controls:**
- *myopic* ($\\gamma=0$): apps only care about today.
- *memoryless*: apps can't see last round's prices, so they can't reward or punish.

Calvano et al. find that both controls remove most of the collusion.
""")
code(r"""
def delta(joint): return (joint - bm["pi_N"]) / (bm["pi_C"] - bm["pi_N"])
runs, qrows = {}, []
for label, kw in [("Q-learning", {}), ("control: myopic γ=0", {"gamma": 0.0}), ("control: memoryless", {"memory": False})]:
    for s in SEEDS:
        r = run_qlearning(bank, seed=s, **kw); lo = limit_outcome(r, bank); runs[(label, s)] = r
        qrows.append(dict(setting=label, seed=s, converged=r["converged"], rounds=r["t_end"], m1=lo["m_mean"][0], m2=lo["m_mean"][1],
                          profit1=lo["profit"][0], profit2=lo["profit"][1], joint=lo["joint"], Delta=delta(lo["joint"]),
                          coverage=lo["coverage"], low_cov=lo["low"], user_net=lo["usernet"], cycle_len=lo["cycle_len"]))
q = pd.DataFrame(qrows)
qsum = q.groupby("setting")[["converged", "rounds", "m1", "m2", "profit1", "profit2", "joint", "Delta", "coverage", "low_cov", "user_net"]].mean()
ref = pd.DataFrame({"m1": [mN[0], mC[0]], "m2": [mN[1], mC[1]], "joint": [bm["pi_N"], bm["pi_C"]], "Delta": [0, 1]},
                   index=["benchmark: static Nash", "benchmark: collusive"])
display(pd.concat([qsum, ref]))
display(q[q.setting == "Q-learning"][["seed", "rounds", "m1", "m2", "joint", "Delta", "cycle_len"]].set_index("seed").T)
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
tr = runs[("Q-learning", 0)]["trace"]; win = 50
for j, lab in [(1, "P1"), (2, "P2")]:
    sm = np.convolve(grid[tr[:, j].astype(int)], np.ones(win) / win, mode="valid")
    ax[0].plot(tr[win - 1:, 0], sm, label=lab)
ax[0].axhline(mN[0], c="C0", ls=":", lw=1); ax[0].axhline(mN[1], c="C1", ls=":", lw=1, label="Nash (each app)")
ax[0].axhline(mC[0], c="k", ls="--", lw=1, label="collusive")
ax[0].set(title="Price levels while learning (seed 0)", xlabel="hour", ylabel="price level m"); ax[0].legend(fontsize=7)
qq = q[q.setting == "Q-learning"]
ax[1].bar(range(len(qq)), qq.Delta); ax[1].axhline(0, c="k", lw=.8); ax[1].axhline(1, c="k", ls="--", lw=.8)
ax[1].set(title="Collusion index Δ per seed (0 = competitive, 1 = collusive)", xlabel="seed", ylim=(-0.3, 1.1))
order = ["control: myopic γ=0", "control: memoryless", "Q-learning"]
ax[2].boxplot([q[q.setting == o].Delta for o in order], tick_labels=["myopic", "memoryless", "Q-learning"]); ax[2].axhline(0, c="k", lw=.8)
ax[2].set(title="Δ: Q-learning vs controls")
plt.tight_layout(); plt.show()
""")
md("""
**Step 3: punishment test.** After learning, we force P1 (and separately P2) to deviate once to its one-shot most profitable price, the one that steals users from the rival. Then both apps follow their learned strategies again.

If the rival answers with a bounty war and both return to the old prices afterwards, the high-profit outcome is being *enforced*, not just stumbled into. We also check whether deviating paid off over the next 15 rounds.
""")
code(r"""
from crowdsim.learning import greedy_path
gains, punished = [], []
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for dvt in (0, 1):
    for s in SEEDS:
        r = runs[("Q-learning", s)]
        path = impulse_response(r, bank, deviator=dvt, horizon=15)
        # counterfactual: no deviation
        pre = greedy_path(r["Q"], L, r["last_state"], steps=120)
        s0 = pre[-1][0] * L + pre[-1][1]
        nodev = [pre[-1]] + greedy_path(r["Q"], L, s0, steps=15)
        idx = lambda m: (list(grid).index(round(m[0], 2)), list(grid).index(round(m[1], 2)))
        pd_ = [Pi[idx(m)][dvt] for m in path[1:]]; pn = [Pi[a][dvt] for a in nodev[1:]]
        disc = 0.95 ** np.arange(15)
        gains.append(dict(deviator=f"P{dvt+1}", seed=s, one_shot_gain=pd_[0] - pn[0], discounted_net_gain=float(disc @ (np.array(pd_) - np.array(pn)))))
        rival = 1 - dvt
        punished.append(path[2, rival] > path[0, rival])
        if s == 0:
            ax[dvt].plot(range(-1, 15), path[:, 0], "o-", label="P1"); ax[dvt].plot(range(-1, 15), path[:, 1], "s-", label="P2")
            ax[dvt].axvline(0, c="grey", ls=":"); ax[dvt].set(title=f"P{dvt+1} deviates at round 0 (seed 0)", xlabel="round", ylabel="price level"); ax[dvt].legend()
plt.tight_layout(); plt.show()
G = pd.DataFrame(gains)
print(f"rival raises its bounty right after the deviation in {np.mean(punished)*100:.0f}% of cases")
G.groupby("deviator")[["one_shot_gain", "discounted_net_gain"]].describe().loc[:, (slice(None), ["mean", "min", "max"])]
""")

md("""
## 3. Experiment 3: users who lie about the rival's offer

Both apps use rival-offer learning (Section 5 of the formulation). We run 0%, 25% and 50% liars with $\\delta=0.5$ (reporting 1.5× the true pay), for 10 seeds × 1500 hours.

The liar sets are nested: the 25% liars are also liars in the 50% run, so we can follow **the same people** across scenarios:
- **group A**: liars in both the 25% and 50% runs;
- **group B**: liars only in the 50% run;
- **group C**: never lie.

Outcomes are averaged over the last 1000 hours, after the transient.
""")
code(r"""
T3, BURN = 1500, 500
def run3(share, seed, **kw):
    m = RivalOfferModel(cfg, liar_share=share, seed=seed, **kw)
    m.earn_tail = np.zeros(cfg.N)
    for t in range(T3):
        m.step()
        if t >= BURN: m.earn_tail += m.last_net
    m.earn_tail /= (T3 - BURN)
    return m
t = time.time(); M3 = {(sh, s): run3(sh, s) for sh in (0.0, 0.25, 0.5) for s in SEEDS}; print(f"{time.time()-t:.0f}s")
A = M3[(0.25, 0)].liar; AB = M3[(0.5, 0)].liar; B = AB & ~A; C = ~AB
rows = []
for (sh, s), m in M3.items():
    h = pd.DataFrame(m.hist).iloc[BURN:]
    rows.append(dict(liars=f"{int(sh*100)}%", seed=s, spend_P1=h.spend0.mean(), spend_P2=h.spend1.mean(),
                     profit_P1=h.profit0.mean(), profit_P2=h.profit1.mean(), coverage=h.coverage.mean(), low_cov=h.low.mean(),
                     earn_A=m.earn_tail[A].mean(), earn_B=m.earn_tail[B].mean(), earn_C=m.earn_tail[C].mean(), earn_all=m.earn_tail.mean()))
E3 = pd.DataFrame(rows)
tab = E3.groupby("liars").mean(numeric_only=True).drop(columns="seed")
display(tab.T)
""")
code(r"""
# paired changes vs the 0% world (same seed, same people), mean ± sd over seeds
base0 = E3[E3.liars == "0%"].set_index("seed")
chg = []
for lv in ("25%", "50%"):
    d = E3[E3.liars == lv].set_index("seed")[base0.columns.drop("liars")] - base0[base0.columns.drop("liars")]
    rel = 100 * d / base0[base0.columns.drop("liars")].abs()
    for col in ["spend_P1", "spend_P2", "profit_P1", "profit_P2", "earn_A", "earn_B", "earn_C", "coverage", "low_cov"]:
        chg.append(dict(liars=lv, metric=col, change=d[col].mean(), change_pct=rel[col].mean(), sd=d[col].std(),
                        seeds_positive=int((d[col] > 0).sum())))
CH = pd.DataFrame(chg).pivot(index="metric", columns="liars", values=["change", "change_pct", "seeds_positive"])
display(CH)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
lbl = {"earn_A": "group A\n(liar at 25% & 50%)", "earn_B": "group B\n(liar at 50% only)", "earn_C": "group C\n(always honest)"}
for k, lv in enumerate(("25%", "50%")):
    vals = [CH.loc[c, ("change_pct", lv)] for c in lbl]
    ax[0].bar(np.arange(3) + k * 0.38, vals, 0.38, label=f"{lv} liars")
ax[0].set_xticks(np.arange(3) + 0.19); ax[0].set_xticklabels(lbl.values(), fontsize=7); ax[0].axhline(0, c="k", lw=.8)
ax[0].set(title="Users: % change in hourly earnings vs 0% liars", ylabel="%"); ax[0].legend()
for k, lv in enumerate(("25%", "50%")):
    vals = [CH.loc[c, ("change_pct", lv)] for c in ["spend_P1", "spend_P2", "profit_P1", "profit_P2"]]
    ax[1].bar(np.arange(4) + k * 0.38, vals, 0.38, label=f"{lv} liars")
ax[1].set_xticks(np.arange(4) + 0.19); ax[1].set_xticklabels(["P1 spend", "P2 spend", "P1 profit", "P2 profit"]); ax[1].axhline(0, c="k", lw=.8)
ax[1].set(title="Apps: % change vs 0% liars", ylabel="%"); ax[1].legend()
for sh in (0.0, 0.25, 0.5):
    sp = np.mean([pd.DataFrame(M3[(sh, s)].hist).spend1.values for s in SEEDS], 0)
    ax[2].plot(np.convolve(sp, np.ones(25) / 25, mode="valid"), label=f"{int(sh*100)}% liars")
ax[2].set(title="P2 (traffic app) spending per hour", xlabel="hour"); ax[2].legend()
plt.tight_layout(); plt.show()
""")
md("""
**Why do the two apps react differently?** Reports go only to the app a user *turned down*. Most users work for P1, so most reports land at P2, and P2's bounties are the ones that get pushed up.

**Sensitivity checks** (5 seeds each):
- a bigger lie, $\\delta=1$ (report 2× the true pay);
- PACMAB's own update $\\hat o\\leftarrow\\max(\\hat o,r)$, under which a single inflated report raises the estimate permanently.
""")
code(r"""
sens = []
for name, kw in [("δ=0.5, avg update", {}), ("δ=1.0, avg update", {"delta": 1.0}), ("δ=0.5, PACMAB max update", {"update": "max"})]:
    for sh in (0.0, 0.5):
        for s in range(5):
            m = run3(sh, s, **kw); h = pd.DataFrame(m.hist).iloc[BURN:]
            sens.append(dict(setting=name, liars=f"{int(sh*100)}%", seed=s, spend_P2=h.spend1.mean(), spend_total=h.spend0.mean() + h.spend1.mean(),
                             profit_total=h.profit0.mean() + h.profit1.mean(), earn_liars=m.earn_tail[AB].mean(), earn_honest=m.earn_tail[C].mean()))
S = pd.DataFrame(sens).groupby(["setting", "liars"]).mean(numeric_only=True).drop(columns="seed")
display(S)
print(f"total notebook runtime: {(time.time()-T0)/60:.1f} min")
""")

md("""
## 4. Summary
The interpretation of these numbers is in `report.md`. Short version:
1. **Baseline:** about half the city is covered, but quiet areas almost never are.
2. **Learning prices:** Q-learning apps settle well above the competitive profit level, i.e. pay users less. The *controls* tell us whether this is genuine collusion: it is only collusion if strategies that can't reward or punish don't get there.
3. **Lying:** lying users push up the bounties of the app they turn down. The money goes to everyone on those blocks, so honest users free-ride; the app that receives the reports pays for it.
""")

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for k, (typ, src) in enumerate(cells):
    c = {"cell_type": typ, "id": f"c{k}", "metadata": {}, "source": src}
    if typ == "code":
        c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)
json.dump(nb, open("simulation.ipynb", "w"), indent=1)
print(len(cells), "cells")
