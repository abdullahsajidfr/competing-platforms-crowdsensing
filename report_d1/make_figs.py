"""Vector (PDF) figures for the D1 report, built from cached runs in results/full/."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from crowdsim.mesa_model import City
R = os.path.join(ROOT, "results", "full"); F = os.path.join(HERE, "figs"); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1, "axes.linewidth": 0.6})
load = lambda n: pickle.load(open(os.path.join(R, n + ".pkl"), "rb"))
save = lambda fig, n: (fig.tight_layout(pad=0.3), fig.savefig(os.path.join(F, n + ".pdf")), plt.close(fig))
city = City()
shape = (city.rows, city.cols)

# ---------------------------------------------------------------- Fig. data
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9))
for k, (v, t) in enumerate([(np.log1p(city.taxi_visits), "taxi user-hours (log)"), (np.log1p(city.ped_visits), "pedestrian user-hours (log)")]):
    pc = ax[k].pcolormesh(v.reshape(shape), cmap="magma", rasterized=False); ax[k].set_title(t); fig.colorbar(pc, ax=ax[k], fraction=.046)
ax[2].pcolormesh(city.low_traffic.reshape(shape), cmap="Blues", vmin=0, vmax=1.3); ax[2].set_title("low-traffic cells (342)")
for a in ax[:3]: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
act = np.array([[len(r) > 0 for r in u["routes"]] for u in city.users]); kinds = np.array([u["kind"] for u in city.users])
ax[3].plot(act[kinds == "taxi"].sum(0), label="taxis"); ax[3].plot(act[kinds == "ped"].sum(0), label="pedestrians")
ax[3].set(xlabel="hour of week", ylabel="active users", title="active users per hour"); ax[3].legend(frameon=False)
save(fig, "data")

# ---------------------------------------------------------------- Fig. equilibrium diagnostics
eq = load("E6_equilibrium")
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
x0 = eq[0]
for h in (9, 33, 81, 150):
    tr = np.array(x0["traces"][h]); ax[0].plot(range(len(tr)), tr / tr[-1], "o-", ms=2.5, label=f"hour {h}")
ax[0].set(xlabel="best-response sweep", ylabel=r"$\Phi/\Phi_{\mathrm{final}}$", title="(a) potential along BR dynamics ($M{=}2$)"); ax[0].legend(frameon=False)
df = pd.DataFrame([r for x in eq for r in x["rows"]]); df["ratio"] = df.welfare_ne / df.welfare_greedy
for M in (2, 3, 4):
    ax[1].hist(df[df.M == M].sweeps, bins=np.arange(1.5, 7.5), alpha=0.55, label=f"$M{{=}}{M}$")
    ax[2].hist(df[df.M == M].ratio, bins=30, alpha=0.55, label=f"$M{{=}}{M}$")
ax[1].set(xlabel="sweeps to reach NE", ylabel="hours", title="(b) convergence time"); ax[1].legend(frameon=False)
ax[2].axvline(1, c="k", lw=0.6); ax[2].set(xlabel=r"$W(a^{NE})/W(a^{greedy})$", ylabel="hours", title="(c) welfare of NE vs. planner"); ax[2].legend(frameon=False)
save(fig, "equilibrium")

# ---------------------------------------------------------------- Fig. price game (M=2)
bank = load("E2_bank_M2"); g = bank["grid"]; Pi = bank["Pi"]; L = len(g)
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
for k, (Mx, t) in enumerate([(Pi[..., 0], r"$\Pi_1$ (air quality)"), (Pi[..., 1], r"$\Pi_2$ (traffic)"), (Pi.sum(-1), r"$\Pi_1+\Pi_2$")]):
    pc = ax[k].pcolormesh(np.arange(L + 1) - .5, np.arange(L + 1) - .5, Mx, cmap="viridis"); fig.colorbar(pc, ax=ax[k], fraction=.046)
    ax[k].plot(range(L), Pi[..., 0].argmax(0), "w^", ms=4, label="BR of app 1")
    ax[k].plot(Pi[..., 1].argmax(1), range(L), "r>", ms=4, label="BR of app 2")
    c = np.unravel_index(np.argmax(Pi.sum(-1)), (L, L)); ax[k].plot(c[1], c[0], "y*", ms=10, mec="k", label="joint-profit max")
    ax[k].set(xticks=range(L), yticks=range(L), xticklabels=g, yticklabels=g, xlabel="$m_2$", ylabel="$m_1$", title=t)
    plt.setp(ax[k].get_xticklabels(), rotation=90)
ax[0].legend(frameon=True, fontsize=5.5, loc="lower right")
save(fig, "price_game")

# ---------------------------------------------------------------- Fig. baseline ablations
def abl(name, key):
    res = load(f"E6_abl_{name}")
    return pd.DataFrame([dict(v=r["kw"][key], **r["summary"]) for r in res]).groupby("v").agg(["mean", "std"])
pm = abl("price_m", "m")
E1 = load("E1_baseline"); e1 = pd.DataFrame([dict(M=r["kw"]["M"], B=r["kw"]["budget"], **r["summary"]) for r in E1]).groupby(["M", "B"]).mean()
ls = abl("loyal_share", "loyal_share"); dk = abl("detour_step_km", "cell_km")
fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.0))
a = ax[0, 0]; a.errorbar(pm.index, pm[("covered_cells", "mean")], pm[("covered_cells", "std")], fmt="o-", ms=3, capsize=2)
a.set(xlabel="fixed price level $m$", ylabel="cells covered / h", title="(a) coverage vs. price")
a = ax[0, 1]; a.plot(pm.index, pm[("profit_total", "mean")], "o-", ms=3, label="platform profit")
a.plot(pm.index, pm[("spend_total", "mean")], "s--", ms=3, label="platform spend"); a.set(xlabel="$m$", title="(b) profit and spend vs. price"); a.legend(frameon=False)
a = ax[0, 2]; a.plot(pm.index, pm[("user_net_per_active_hour", "mean")], "o-", ms=3, label="user net / active h")
a2 = a.twinx(); a2.plot(pm.index, pm[("task_completion", "mean")], "s--", ms=3, c="C1", label="task completion"); a2.set_ylabel("task completion", color="C1")
a.set(xlabel="$m$", ylabel="user net / active h", title="(c) users and completion vs. price")
for M in (2, 3, 4):
    ax[1, 0].plot([150, 300, 600], [e1.loc[(M, B), "covered_cells"] for B in (150, 300, 600)], "o-", ms=3, label=f"$M{{=}}{M}$")
    ax[1, 1].plot([150, 300, 600], [e1.loc[(M, B), "gini_all"] for B in (150, 300, 600)], "o-", ms=3, label=f"$M{{=}}{M}$")
ax[1, 0].set(xticks=[150, 300, 600], xlabel="hourly budget $B$", ylabel="cells covered / h", title="(d) coverage vs. budget"); ax[1, 0].legend(frameon=False)
ax[1, 1].set(xticks=[150, 300, 600], xlabel="hourly budget $B$", ylabel="Gini of earnings", title="(e) inequality vs. budget"); ax[1, 1].legend(frameon=False)
bt = abl("loyalty_bonus_beta", "beta"); rh = abl("quality_rho", "rho")
rng = lambda df, met: 100 * (df[(met, "mean")].iloc[-1] / df[(met, "mean")].iloc[0] - 1)
names = ["loyal share 0$\\to$0.6", "loyalty $\\beta$ 0$\\to$4", "detour 0.25$\\to$1 km", "quality $\\rho$ 0.3$\\to$0.9"]
dfs = [ls, bt, dk, rh]
a = ax[1, 2]; yy = np.arange(4)
a.barh(yy + .2, [rng(df_, "covered_cells") for df_ in dfs], .4, label="cells covered")
a.barh(yy - .2, [rng(df_, "user_net_per_active_hour") for df_ in dfs], .4, label="user net / h")
a.axvline(0, c="k", lw=.5); a.set_yticks(yy); a.set_yticklabels(names, fontsize=6)
a.set(xlabel="% change over the range", title="(f) weak parameters"); a.legend(frameon=False, fontsize=6, loc="lower left")
save(fig, "ablations")

# ---------------------------------------------------------------- Fig. lying (RQ1)
E3 = load("E3_lying")
def pick(M, sh, s, B=300, d=0.5, u="avg"):
    return [r for r in E3 if r["kw"]["M"] == M and r["kw"]["budget"] == B and r["kw"].get("delta") == d and r["kw"].get("update", "avg") == u
            and r["kw"]["strategic_share"] == sh and r["kw"]["seed"] == s][0]
rows = []
for M in (2, 3, 4):
    for s in range(10):
        b0, r25, r50 = pick(M, 0, s), pick(M, .25, s), pick(M, .5, s)
        A = r25["utype"] == "strategic"; S50 = r50["utype"] == "strategic"; C = (b0["utype"] == "honest") & ~S50
        pc = lambda x, y: 100 * (x - y) / abs(y)
        for sh, r in ((25, r25), (50, r50)):
            rows.append(dict(M=M, sh=sh, spend=pc(r["summary"]["spend_total"], b0["summary"]["spend_total"]),
                             profit=pc(r["summary"]["profit_total"], b0["summary"]["profit_total"]),
                             honest=pc(r["earn"][C].mean(), b0["earn"][C].mean()), liars=pc(r["earn"][A].mean(), b0["earn"][A].mean()),
                             cover=pc(r["summary"]["covered_cells"], b0["summary"]["covered_cells"])))
d = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
mu = d[d.M == 2].groupby("sh").mean(); sd = d[d.M == 2].groupby("sh").std()
keys = ["liars", "honest", "spend", "profit", "cover"]; labs = ["liars' earnings", "honest earnings", "platform spend", "platform profit", "cells covered"]
for k, sh in enumerate((25, 50)):
    ax[0].bar(np.arange(5) + (k - .5) * .38, mu.loc[sh, keys], .38, yerr=sd.loc[sh, keys], capsize=2, error_kw={"lw": .6}, label=f"{sh}% strategic")
ax[0].axhline(0, c="k", lw=.5); ax[0].set_xticks(range(5)); ax[0].set_xticklabels(labs, fontsize=6.5)
ax[0].set(ylabel="% change vs. truthful", title="(a) effect of inflated reports ($M{=}2$, $B{=}300$)"); ax[0].legend(frameon=False)
for met, ls_ in (("spend", "-"), ("profit", "--")):
    for M in (2, 3, 4):
        mm = d[d.M == M].groupby("sh")[met].mean()
        ax[1].plot([0, 25, 50], [0] + list(mm.values), "o" + ls_, ms=3, c=f"C{M-2}", label=f"{met}, $M{{=}}{M}$")
ax[1].axhline(0, c="k", lw=.5); ax[1].set(xlabel="% strategic users", ylabel="% change", title="(b) platform spend and profit"); ax[1].legend(frameon=False, ncol=2, fontsize=6, loc="lower left")
save(fig, "lying")

# ---------------------------------------------------------------- Fig. Q-learning (RQ2)
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1)); Lr = 750
bench = {2: (0.82, 0.70), 3: (None, 0.667), 4: (None, 0.675)}
for k, M in enumerate((2, 3, 4)):
    gM = load(f"E2_bank_M{M}")["grid"]
    for lab, fn, ls_ in [("Q-learning", f"E2_q_M{M}", "-"), (r"myopic ($\gamma{=}0$)", f"E2_q_M{M}_myopic", ":"), ("memoryless", f"E2_q_M{M}_nomem", "--")]:
        def pad(r):
            xx = gM[r["trace"][:, 1:]].mean(1); return np.concatenate([xx, np.full(max(0, Lr - len(xx)), xx[-1])])[:Lr]
        tr = pd.Series(np.mean([pad(r) for r in load(fn)], 0)).rolling(10, min_periods=1).mean()
        ax[k].plot(np.arange(Lr) * 2000 / 1e6, tr, ls_, label=lab)
    if bench[M][0]: ax[k].axhline(bench[M][0], c="r", lw=.7, label="static Nash (mixed)")
    ax[k].axhline(bench[M][1], c="k", lw=.7, ls="-.", label="joint-profit max")
    ax[k].set(xlabel=r"hours of play ($\times10^6$)", ylabel="mean price level", title=f"$M{{=}}{M}$")
ax[0].legend(frameon=False, fontsize=5.8)
save(fig, "qlearning")

# ---------------------------------------------------------------- Fig. bonus (RQ3) + methods (RQ4)
E4 = load("E4_bonus"); e4 = pd.DataFrame([dict(b=r["kw"]["bonus"], **r["summary"]) for r in E4]).groupby("b").agg(["mean", "std"])
rows = []
for r in E1:
    if r["kw"]["budget"] == 300: rows.append(dict(M=r["kw"]["M"], meth="Fixed", **r["summary"]))
for M in (2, 3, 4):
    for e in load(f"E5_qeval_M{M}"): rows.append(dict(M=M, meth="Q-learning", **e["summary"]))
    for s in range(10): rows.append(dict(M=M, meth="Rival-offer", **pick(M, 0, s)["summary"]))
t5 = pd.DataFrame(rows).groupby(["M", "meth"]).mean(numeric_only=True)
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
ax[0].errorbar(100 * e4.index, e4[("low_covered_cells", "mean")], e4[("low_covered_cells", "std")], fmt="o-", ms=3, capsize=2, label="low-traffic cells")
a2 = ax[0].twinx(); a2.plot(100 * e4.index, e4[("covered_cells", "mean")], "s--", ms=3, c="C1", label="all cells"); a2.set_ylabel("all cells / h", color="C1"); a2.set_ylim(60, 85)
ax[0].set(xlabel="bonus on low-traffic cells (%)", ylabel="low-traffic cells / h", title="(a) RQ3: equal-budget bonus")
cols = {"Fixed": "#bdbdbd", "Q-learning": "#3182bd", "Rival-offer": "#e6550d"}
for k, meth in enumerate(cols):
    ax[1].bar(np.arange(3) + (k - 1) * .26, [t5.loc[(M, meth), "cells_per_100"] for M in (2, 3, 4)], .26, color=cols[meth], label=meth)
    ax[2].bar(np.arange(3) + (k - 1) * .26, [t5.loc[(M, meth), "user_net_per_active_hour"] for M in (2, 3, 4)], .26, color=cols[meth], label=meth)
for a, t in ((ax[1], "(b) RQ4: cells per 100 spent"), (ax[2], "(c) user net earnings / active h")):
    a.set_xticks(range(3)); a.set_xticklabels(["$M{=}2$", "$M{=}3$", "$M{=}4$"]); a.set_title(t)
ax[1].legend(frameon=False, fontsize=6)
save(fig, "bonus_methods")
print(t5[["cells_per_100", "user_net_per_active_hour", "gini_all", "covered_cells", "spend_total"]].round(2))
print(d.groupby(["M", "sh"]).mean().round(1))
