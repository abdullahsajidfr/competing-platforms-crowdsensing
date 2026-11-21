"""Baseline ablations and equilibrium diagnostics on the real-trace Mesa model.

Run from the project root:  .venv/bin/python -m experiments.ablations
Writes results/full/E6_*.pkl (cached).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import crowdsim.full_experiments as fx
from crowdsim.mesa_model import CrowdsensingModel, q_of

SEEDS = range(10)
WEEK = 168


# ---------------------------------------------------------------- ablation grids (fixed price)
def ablations():
    base = dict(M=2, mechanism="fixed", m=1.0, budget=300.0, loyal_share=0.2, beta=1.0)
    grids = {
        "price m": [dict(base, m=m) for m in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)],
        "loyal share": [dict(base, loyal_share=s) for s in (0.0, 0.2, 0.4, 0.6)],
        "loyalty bonus beta": [dict(base, beta=b) for b in (0.0, 1.0, 2.0, 4.0)],
        "quality rho": [dict(base, rho=r) for r in (0.3, 0.5, 0.7, 0.9)],
        "detour step km": [dict(base, cell_km=k) for k in (0.25, 0.5, 1.0)],
    }
    out = {}
    for name, cfgs in grids.items():
        res = fx.cached(f"E6_abl_{name.replace(' ', '_')}", fx.run_grid, cfgs, SEEDS, 2 * WEEK, 0)
        key = {"price m": "m", "loyal share": "loyal_share", "loyalty bonus beta": "beta",
               "quality rho": "rho", "detour step km": "cell_km"}[name]
        out[name] = pd.DataFrame([dict(value=r["kw"][key], **r["summary"]) for r in res])
    return out


# ---------------------------------------------------------------- equilibrium diagnostics
def _welfare(model):
    W = np.stack([p.W for p in model.platforms])
    val = (W * model.V * q_of(model.n, model.rho)).sum()
    cost = sum(u.cost[u.choice] for u in model.users if u.active and u.choice >= 0)
    return val - cost


def _greedy(model):
    """Greedy planner: repeatedly add the (user, posted task) with the largest marginal
    welfare gain. Same posted tasks as the game; payments are transfers and ignored."""
    W = np.stack([p.W for p in model.platforms]); V, rho = model.V, model.rho
    n = np.zeros_like(model.n); free = [u for u in model.users if u.active and len(u.aj)]
    total = 0.0
    while free:
        best, bu, bk = 1e-12, None, -1
        for u in free:
            gain = W[u.aj, u.ag] * V * (q_of(n[u.aj, u.ag] + 1, rho) - q_of(n[u.aj, u.ag], rho)) - u.cost
            k = int(np.argmax(gain))
            if gain[k] > best:
                best, bu, bk = gain[k], u, k
        if bu is None:
            break
        n[bu.aj[bk], bu.ag[bk]] += 1; total += best; free.remove(bu)
    return total


def _potential(model):
    n = model.n; R = model.R
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, max(int(n.max()), 1) + 1))])
    phi = (R * H[n]).sum()
    phi -= sum(u.ac[u.choice] for u in model.users if u.active and u.choice >= 0)
    return float(phi)


def _eq_job(job):
    M, seed = job
    m = CrowdsensingModel(fx.city(), M=M, mechanism="fixed", budget=300.0, seed=seed, type_seed=seed)
    rows, traces = [], {}
    for h in range(WEEK):
        for u in m.users:
            u.prepare(h)
        reach = np.zeros(m.city.G, bool)
        for u in m.users:
            if u.active:
                reach[u.reach] = True
        for p in m.platforms:
            p.post(reach)
        for u in m.users:
            u.build_actions(m.posted)
        m.n[:] = 0
        act = [u for u in m.users if u.active]
        tr = [_potential(m)]
        for sweep in range(500):
            m.nprng.shuffle(act); moved = False
            for u in act:
                moved |= u.best_respond()
            tr.append(_potential(m))
            if not moved:
                break
        g = _greedy(m); w = _welfare(m)
        rows.append(dict(M=M, seed=seed, hour=h, active=len(act), sweeps=sweep + 1,
                         welfare_ne=w, welfare_greedy=g))
        traces[h] = tr
    return dict(rows=rows, traces=traces)


def equilibrium_diagnostics():
    jobs = [(M, s) for M in (2, 3, 4) for s in range(3)]
    return fx.pmap(_eq_job, jobs)


if __name__ == "__main__":
    a = ablations()
    for k, d in a.items():
        print(k); print(d.groupby("value")[["coverage", "low_cov", "spend_total", "profit_total",
                                             "task_completion", "participation", "user_net_per_active_hour",
                                             "gini_all", "cells_per_100"]].mean().round(3))
    eq = fx.cached("E6_equilibrium", equilibrium_diagnostics)
    df = pd.DataFrame([r for x in eq for r in x["rows"]])
    df["ratio"] = df.welfare_ne / df.welfare_greedy
    print(df.groupby("M")[["sweeps", "ratio"]].describe().round(3))
