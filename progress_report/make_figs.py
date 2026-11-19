"""Vector (PDF) figures for the progress report, built from the cached runs in results/full/."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "full"); F = os.path.join(HERE, "figs")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 6,
                     "axes.titlesize": 6.5, "axes.labelsize": 6, "legend.fontsize": 5, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5, "lines.linewidth": 1.0, "axes.linewidth": 0.6})
load = lambda n: pickle.load(open(os.path.join(R, n + ".pkl"), "rb"))
W, H = 1.78, 1.05

# ---------------- (a) RQ1: % change vs 0% liars, M=2, base budget, delta=0.5
E3 = load("E3_lying")
pick = lambda sh, s: [r for r in E3 if r["kw"]["M"] == 2 and r["kw"]["budget"] == 300 and r["kw"].get("delta") == 0.5
                      and r["kw"].get("update", "avg") == "avg" and r["kw"]["strategic_share"] == sh and r["kw"]["seed"] == s][0]
rows = []
for s in range(10):
    b0, r25, r50 = pick(0.0, s), pick(0.25, s), pick(0.5, s)
    A = r25["utype"] == "strategic"; S50 = r50["utype"] == "strategic"; C = (b0["utype"] == "honest") & ~S50
    for sh, r in ((25, r25), (50, r50)):
        pc = lambda x, y: 100 * (x - y) / abs(y)
        rows.append(dict(sh=sh, spend=pc(r["summary"]["spend_total"], b0["summary"]["spend_total"]),
                         profit=pc(r["summary"]["profit_total"], b0["summary"]["profit_total"]),
                         honest=pc(r["earn"][C].mean(), b0["earn"][C].mean()),
                         liars=pc(r["earn"][A].mean(), b0["earn"][A].mean()),
                         cover=pc(r["summary"]["covered_cells"], b0["summary"]["covered_cells"])))
d = pd.DataFrame(rows); mu = d.groupby("sh").mean(); sd = d.groupby("sh").std()
labs = ["Liars'\nearn.", "Honest\nearn.", "Plat.\nspend", "Plat.\nprofit", "Cells\ncovered"]
keys = ["liars", "honest", "spend", "profit", "cover"]
fig, ax = plt.subplots(figsize=(W, H)); x = np.arange(5); w = 0.38
for k, sh in enumerate((25, 50)):
    ax.bar(x + (k - 0.5) * w, mu.loc[sh, keys], w, yerr=sd.loc[sh, keys], capsize=1.5, error_kw={"lw": 0.5},
           label=f"{sh}% strategic", color=["#9ecae1", "#3182bd"][k])
ax.axhline(0, c="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5); ax.set_ylabel("% change vs. truthful", labelpad=1)
ax.legend(frameon=False, loc="upper right", fontsize=4.6); ax.set_title("(a) RQ1: inflated reports")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(F, "rq1_lying.pdf")); plt.close(fig)

# ---------------- (b) RQ2: Q-learning price paths, M=2
bank = load("E2_bank_M2"); g = bank["grid"]
fig, ax = plt.subplots(figsize=(W, H)); L = 750
for lab, fn, ls in [("Q-learning", "E2_q_M2", "-"), ("myopic ($\\gamma{=}0$)", "E2_q_M2_myopic", ":"), ("memoryless", "E2_q_M2_nomem", "--")]:
    runs = load(fn)
    def pad(r):
        xx = g[r["trace"][:, 1:]].mean(1); return np.concatenate([xx, np.full(max(0, L - len(xx)), xx[-1])])[:L]
    tr = pd.Series(np.mean([pad(r) for r in runs], 0)).rolling(10, min_periods=1).mean()
    ax.plot(np.arange(L) * 2000 / 1e6, tr, ls, label=lab)
ax.axhline(0.82, c="r", lw=0.6, label="static Nash (mixed)"); ax.axhline(0.70, c="k", lw=0.6, ls="-.", label="joint-profit max")
ax.set(xlabel="hours of play ($\\times10^6$)", ylabel="mean price level $m$", ylim=(0.66, 1.2))
ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=4.2, handlelength=1.6, columnspacing=0.5); ax.set_title("(b) RQ2: learned pay, $M{=}2$")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(F, "rq2_qlearning.pdf")); plt.close(fig)

# ---------------- (c) RQ3: coverage bonus
E4 = load("E4_bonus")
e4 = pd.DataFrame([dict(b=r["kw"]["bonus"], low=r["summary"]["low_covered_cells"], all=r["summary"]["covered_cells"]) for r in E4])
m4 = e4.groupby("b").mean(); s4 = e4.groupby("b").std()
fig, ax = plt.subplots(figsize=(W, H))
ax.errorbar(100 * m4.index, m4["low"], yerr=s4["low"], fmt="o-", ms=2.5, capsize=1.5, lw=1, c="#3182bd", label="low-traffic cells")
ax.set(xlabel="bonus on low-traffic cells (%)", ylabel="low-traffic cells / h")
ax2 = ax.twinx(); ax2.plot(100 * m4.index, m4["all"], "s--", ms=2.5, c="#e6550d", lw=0.9, label="all cells")
ax2.set_ylabel("all cells / h", color="#e6550d"); ax2.tick_params(axis="y", colors="#e6550d"); ax2.set_ylim(60, 85)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right"); ax.set_title("(c) RQ3: coverage bonus")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(F, "rq3_bonus.pdf")); plt.close(fig)

# ---------------- numbers for the RQ4 table
E1 = load("E1_baseline"); rows = []
for r in E1:
    if r["kw"]["budget"] == 300: rows.append(dict(M=r["kw"]["M"], meth="Fixed", **r["summary"]))
for M in (2, 3, 4):
    for e in load(f"E5_qeval_M{M}"): rows.append(dict(M=M, meth="Q-learning", **e["summary"]))
    for r in E3:
        k = r["kw"]
        if k["M"] == M and k["budget"] == 300 and k.get("update", "avg") == "avg" and k["strategic_share"] == 0.0 and k.get("delta") == 0.5:
            rows.append(dict(M=M, meth="Rival-offer", **r["summary"]))
t = pd.DataFrame(rows).groupby(["M", "meth"])[["cells_per_100", "user_net_per_active_hour", "gini_all", "task_completion"]].mean()
print(t.round(2)); print(mu.round(1))
# ---------------- (d) RQ4: coverage per 100 spent and user earnings
fig, ax = plt.subplots(figsize=(W, H)); x = np.arange(3); w = 0.26
cols = {"Fixed": "#bdbdbd", "Q-learning": "#3182bd", "Rival-offer": "#e6550d"}
for k, meth in enumerate(cols):
    ax.bar(x + (k - 1) * w, [t.loc[(M, meth), "cells_per_100"] for M in (2, 3, 4)], w, color=cols[meth], label=meth)
ax.set_xticks(x); ax.set_xticklabels(["$M{=}2$", "$M{=}3$", "$M{=}4$"]); ax.set_ylabel("cells covered per 100 spent")
ax.set_ylim(0, 44); ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=4.6, handlelength=1, columnspacing=0.6)
ax.set_title("(d) RQ4: equal-budget efficiency")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(F, "rq4_methods.pdf")); plt.close(fig)
