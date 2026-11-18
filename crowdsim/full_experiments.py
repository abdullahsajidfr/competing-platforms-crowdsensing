"""Experiment drivers for the full D1 setup (real traces, Mesa model, 2-4 apps).
All heavy loops run in parallel (fork pool, 8 workers) and are cached to results/full/.
"""
from __future__ import annotations

import itertools, os, pickle, time
import multiprocessing as mp
import numpy as np
import pandas as pd

from .mesa_model import City, CrowdsensingModel, gini
from .equilibrium import pure_nash, mixed_nash_2p

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "full")
os.makedirs(OUT, exist_ok=True)
_CITY = None


def city():
    global _CITY
    if _CITY is None:
        _CITY = City()
    return _CITY


def pmap(fn, jobs, procs=8):
    city()                                        # load once before forking
    with mp.get_context("fork").Pool(procs) as p:
        return p.map(fn, jobs, chunksize=1)


def cached(name, fn, *a, **kw):
    path = os.path.join(OUT, name + ".pkl")
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception as e:                    # e.g. library-version mismatch: recompute
            print(f"[{name}] cache unreadable ({type(e).__name__}); recomputing")
    t = time.time(); res = fn(*a, **kw)
    pickle.dump(res, open(path, "wb")); print(f"[{name}] {time.time() - t:.0f}s")
    return res


# ======================================================================== ABM runs
def summarize(model, burn):
    df = pd.DataFrame(model.rec).iloc[burn:]
    M = model.M
    ut = model.user_table()
    spend = sum(df[f"spend{j}"].mean() for j in range(M))
    posted = sum(df[f"posted{j}"].sum() for j in range(M)); comp = sum(df[f"completed{j}"].sum() for j in range(M))
    G = model.city.G
    s = dict(coverage=df.coverage.mean(), low_cov=df.low_cov.mean(),
             covered_cells=df.coverage.mean() * G, low_covered_cells=df.low_cov.mean() * model.city.low_traffic.sum(),
             spend_total=spend, profit_total=sum(df[f"profit{j}"].mean() for j in range(M)),
             task_completion=comp / max(posted, 1), tasks_posted=posted / len(df),
             cells_per_100=100 * df.coverage.mean() * G / max(spend, 1e-9),
             reachable_cov=df.coverage.mean() * G / model.city.reachable_ever.sum(),
             low_cells_per_100=100 * df.low_cov.mean() * model.city.low_traffic.sum() / max(spend, 1e-9),
             gini_all=gini(ut.earn), gini_paid=gini(ut.earn[ut.earn > 0]),
             participation=df.participating.sum() / df.active_users.sum(),
             user_net_per_active_hour=ut.earn.sum() / df.active_users.sum())
    for j in range(M):
        s[f"spend{j}"] = df[f"spend{j}"].mean(); s[f"profit{j}"] = df[f"profit{j}"].mean()
        s[f"m{j}"] = df[f"m{j}"].mean()
        s[f"completion{j}"] = df[f"completed{j}"].sum() / max(df[f"posted{j}"].sum(), 1)
    for ty in ("honest", "loyal", "strategic"):
        sub = ut[ut.utype == ty]
        s[f"earn_{ty}"] = sub.earn.mean() if len(sub) else np.nan
    return s, ut.earn.values, ut.utype.values, df


def _abm(job):
    kw, T, burn, keep_ts = job
    m = CrowdsensingModel(city(), burn_in=burn, **kw)
    for _ in range(T):
        m.step()
    s, earn, utype, df = summarize(m, burn)
    ts = pd.DataFrame(m.rec) if keep_ts else None
    return dict(kw=kw, summary=s, earn=earn, utype=utype, ts=ts)


def run_grid(configs, seeds, T, burn, keep_ts=False):
    jobs = [(dict(cfg, seed=s, type_seed=s), T, burn, keep_ts) for cfg in configs for s in seeds]
    return pmap(_abm, jobs)


# ======================================================================== Q-learning
def _bank_job(job):
    M, grid, prof_idx, budget = job
    m = CrowdsensingModel(city(), M=M, mechanism="fixed", budget=budget, seed=0, type_seed=0)
    prices = [grid[k] for k in prof_idx]
    rows = []
    for h in range(168):
        out = m.stage(h, prices)
        net = 0.0
        for u in m.users:
            if u.active and u.choice >= 0:
                j, g = u.aj[u.choice], u.ag[u.choice]
                net += m.R[j, g] / m.n[j, g] - u.cost[u.choice]
        rows.append(np.concatenate([out["profit"], out["spend"],
                                    [out["coverage"], out["low_cov"], net, out["completed"].sum(), out["posted"].sum()]]))
    return np.array(rows)


def build_bank(M, grid, budget):
    profiles = list(itertools.product(range(len(grid)), repeat=M))
    arr = np.stack(pmap(_bank_job, [(M, grid, p, budget) for p in profiles]))   # (P, 168, F)
    K = len(grid)
    shape = (K,) * M
    bank = dict(grid=np.asarray(grid), M=M, profit=arr[..., :M].reshape(shape + (168, M)),
                spend=arr[..., M:2 * M].reshape(shape + (168, M)),
                coverage=arr[..., 2 * M].reshape(shape + (168,)), low=arr[..., 2 * M + 1].reshape(shape + (168,)),
                usernet=arr[..., 2 * M + 2].reshape(shape + (168,)),
                completion=(arr[..., 2 * M + 3] / np.maximum(arr[..., 2 * M + 4], 1)).reshape(shape + (168,)))
    bank["Pi"] = bank["profit"].mean(M)
    return bank


def benchmarks(bank):
    Pi = bank["Pi"]; M = bank["M"]; grid = bank["grid"]
    tot = Pi.sum(-1); c = np.unravel_index(np.argmax(tot), tot.shape)
    out = dict(collusive=tuple(grid[list(c)]), pi_C=float(tot[c]))
    ne = pure_nash(Pi)
    if ne:
        idx = min(ne, key=lambda x: Pi[x].sum())
        out.update(nash_type="pure", nash_m=tuple(grid[list(idx)]), pi_N=float(Pi[idx].sum()), all_pure=[tuple(grid[list(i)]) for i in ne])
    elif M == 2:
        x, y = mixed_nash_2p(Pi[..., 0], Pi[..., 1])[0]
        out.update(nash_type="mixed", nash_m=(float(x @ grid), float(y @ grid)),
                   pi_N=float(x @ Pi[..., 0] @ y + x @ Pi[..., 1] @ y),
                   mixed=({float(grid[i]): round(float(p), 2) for i, p in enumerate(x) if p > 1e-6},
                          {float(grid[i]): round(float(p), 2) for i, p in enumerate(y) if p > 1e-6}))
    else:
        out.update(nash_type="none (no pure NE; mixed NE not computed for M>2)", pi_N=np.nan)
    return out


def _sidx(a, K):
    s = 0
    for x in a:
        s = s * K + int(x)
    return s


def _qrun(job):
    bank, seed, alpha, gamma, beta, T_max, stable_for, memory = job
    rng = np.random.default_rng(seed)
    M = bank["M"]; K = len(bank["grid"]); prof = bank["profit"]; Pi = bank["Pi"]
    S = K ** M if memory else 1
    Q = np.zeros((M, S, K))
    for j in range(M):                                   # Calvano init vs uniform rivals
        axes = tuple(i for i in range(M) if i != j)
        Q[j] += Pi[..., j].mean(axis=axes) / (1 - gamma)
    s = int(rng.integers(S)); a_prev = np.unravel_index(s, (K,) * M) if memory else tuple(rng.integers(K, size=M))
    greedy_prev = Q.argmax(2).copy(); stable = 0; trace = []
    hours = rng.integers(168, size=T_max); u = rng.random((T_max, M)); ra = rng.integers(K, size=(T_max, M))
    for t in range(T_max):
        eps = np.exp(-beta * t)
        a = tuple(int(ra[t, j]) if u[t, j] < eps else int(Q[j, s].argmax()) for j in range(M))
        r = prof[a][hours[t]]
        s2 = _sidx(a, K) if memory else 0
        for j in range(M):
            Q[j, s, a[j]] += alpha * (r[j] + gamma * Q[j, s2].max() - Q[j, s, a[j]])
        s = s2
        if t % 2000 == 0:
            trace.append((t,) + a)
        if t % 1000 == 999:
            g = Q.argmax(2)
            if np.array_equal(g, greedy_prev):
                stable += 1000
            else:
                stable = 0; greedy_prev = g.copy()
            if stable >= stable_for and eps < 0.01:
                break
    return dict(Q=Q, t_end=t + 1, converged=stable >= stable_for, trace=np.array(trace), last_state=s, memory=memory, seed=seed)


def greedy_path(Q, M, K, s0, steps, memory=True):
    s = s0; path = []
    for _ in range(steps):
        a = tuple(int(Q[j, s].argmax()) for j in range(M))
        path.append(a); s = _sidx(a, K) if memory else 0
    return path


def limit_outcome(res, bank):
    M = bank["M"]; K = len(bank["grid"]); Pi = bank["Pi"]
    path = greedy_path(res["Q"], M, K, res["last_state"], 200, res["memory"])[100:]
    prof = np.array([Pi[a] for a in path])
    return dict(m=np.array([[bank["grid"][x] for x in a] for a in path]).mean(0),
                profit=prof.mean(0), joint=float(prof.sum(1).mean()),
                coverage=float(np.mean([bank["coverage"][a].mean() for a in path])),
                low=float(np.mean([bank["low"][a].mean() for a in path])),
                usernet=float(np.mean([bank["usernet"][a].mean() for a in path])),
                completion=float(np.mean([bank["completion"][a].mean() for a in path])),
                spend=float(np.mean([bank["spend"][a].mean(0).sum() for a in path])))


def run_q(bank, seeds, memory=True, gamma=0.95, alpha=0.15, beta=1e-5, T_max=1_500_000, stable_for=25_000):
    jobs = [(bank, s, alpha, gamma, beta, T_max, stable_for, memory) for s in seeds]
    return pmap(_qrun, jobs)


def _qeval(job):
    """Play the learned greedy price strategies inside the Mesa model for T hours."""
    Q, last_state, grid, M, budget, seed, T = job
    K = len(grid)
    m = CrowdsensingModel(city(), M=M, mechanism="fixed", budget=budget, seed=seed, type_seed=seed)
    s = last_state
    for _ in range(T):
        a = tuple(int(Q[j, s].argmax()) for j in range(M))
        m.next_prices = [grid[x] for x in a]; m.step(); s = _sidx(a, K)
    s_, earn, utype, df = summarize(m, 0)
    return dict(summary=s_, earn=earn, utype=utype)


def eval_q(runs, bank, budget, T=336):
    jobs = [(r["Q"], r["last_state"], bank["grid"], bank["M"], budget, r["seed"], T) for r in runs]
    return pmap(_qeval, jobs)
