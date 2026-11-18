"""D1 experiments: equilibrium checks (E1, E2) + fixed-price baseline simulation.

Run from the project root:   .venv/bin/python -m experiments.run_d1
Outputs go to results/ (figures, CSV tables, summary.json).
"""
from __future__ import annotations

import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crowdsim.env import Config, World, UserGame, welfare
from crowdsim.dynamics import async_best_response, br_with_inertia, log_linear
from crowdsim.equilibrium import (random_tiny_instance, brute_force, price_matrix,
                                  pure_nash, collusive, mixed_nash_2p)
from crowdsim.model import CrowdsensingModel, FixedPrice

OUT = "results"
os.makedirs(OUT, exist_ok=True)
summary = {}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)


# =============================================================== E1a: potential & dynamics
def e1_dynamics(cfg):
    w = World(cfg)
    rng = np.random.default_rng(7)
    acts = w.build_actions(w.draw_routes(rng))
    R = np.ones(cfg.M)[:, None] * w.W * cfg.b
    g = UserGame(R, acts)
    a_br, sweeps, tr_br = async_best_response(g, np.random.default_rng(1), record=True,
                                              check_potential=True)
    a_in, steps, tr_in = br_with_inertia(g, np.random.default_rng(2), p_inertia=0.7)
    lll = {T: log_linear(g, np.random.default_rng(3), T=T, steps=30000) for T in (1.0, 0.3, 0.1)}
    # random-deviation identity check
    r2 = np.random.default_rng(4)
    err = 0.0
    for _ in range(5000):
        i = r2.integers(g.N); b = a_br.copy(); b[i] = r2.integers(-1, len(acts[i][0]))
        err = max(err, abs((g.potential(b) - g.potential(a_br)) - (g.utility(i, b) - g.utility(i, a_br))))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(tr_br, lw=1.2); ax[0].set(title=f"Async best response ({sweeps} sweeps)",
                                          xlabel="single-user updates", ylabel="potential $\\Phi$")
    ax[1].plot(tr_in, marker=".", lw=1); ax[1].set(title=f"BR with inertia p=0.7 ({steps} periods)",
                                                    xlabel="period", ylabel="$\\Phi$")
    for T, (_, tr) in lll.items():
        ax[2].plot(tr, lw=0.8, label=f"T={T}")
    ax[2].axhline(tr_br[-1], color="k", ls="--", lw=0.8, label="BR NE")
    ax[2].set(title="Log-linear learning", xlabel="revisions", ylabel="$\\Phi$"); ax[2].legend()
    save(fig, "fig1_potential_dynamics")
    summary["E1_dynamics"] = dict(
        max_abs_dPhi_minus_dU=err, br_sweeps=sweeps, br_is_NE=bool(g.is_nash(a_br)),
        inertia_periods=steps, inertia_is_NE=bool(g.is_nash(a_in)),
        phi_br=tr_br[-1], phi_lll={str(T): float(np.mean(tr[-5000:])) for T, (_, tr) in lll.items()})


# =============================================================== E1b: toy example (RQ3)
def e1_toy():
    # one platform, cells A,B; two users who can reach both at zero cost
    RB = np.linspace(0, 10, 101)
    rows = []
    for rb in RB:
        R = np.array([[10.0, rb]])
        acts = [(np.array([0, 0]), np.array([0, 1]), np.zeros(2), np.zeros(2))] * 2
        g = UserGame(R, acts)
        ne, _ = brute_force(g, np.ones((1, 2)), 20.0, 0.5)
        covB = [((n := g.loads(a))[0, 1] >= 1) for a, _, _ in ne]
        rows.append((rb, len(ne), min(covB), max(covB)))
    rows = np.array(rows, float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.step(rows[:, 0], rows[:, 2], where="mid", label="B covered in every NE")
    ax.step(rows[:, 0], rows[:, 3], where="mid", ls="--", label="B covered in some NE")
    ax.axvline(5, color="grey", lw=0.8); ax.text(5.2, 0.75, "$R_B = R_A/2$", color="grey")
    ax.set(xlabel="reward on cell B ($R_A$ = 10)", ylabel="cell B covered", yticks=[0, 1],
           title="Toy: 2 users, 2 cells, equal split")
    ax.legend(loc="lower right")
    save(fig, "fig2_toy_bonus_threshold")
    summary["E1_toy"] = dict(threshold_RB=5.0, note="R_B<5: unique NE (A,A); R_B=5: both; R_B>5: (A,B)")


# =============================================================== E1c: brute-force PoA
def e1_bruteforce(n_inst=300):
    rng = np.random.default_rng(11)
    V, rho = 20.0, 0.5
    stats = []
    for s in range(n_inst):
        W, kappa, acts, R = random_tiny_instance(rng)
        g = UserGame(R, acts)
        ne, opt = brute_force(g, W, V, rho)
        a_br, _, _ = async_best_response(g, np.random.default_rng(s))
        br_ok = g.is_nash(a_br)
        welf = [x[1] for x in ne]
        phis = [x[2] for x in ne]
        pmax_w = welf[int(np.argmax(phis))]
        stats.append(dict(n_ne=len(ne), br_ok=br_ok, opt=opt, worst=min(welf), best=max(welf),
                          potmax_welfare=pmax_w))
    n_ne = np.array([d["n_ne"] for d in stats])
    ok = all(d["br_ok"] for d in stats)
    valid = [d for d in stats if d["worst"] > 1e-9]
    poa = np.array([d["opt"] / d["worst"] for d in valid])
    pos = np.array([d["opt"] / d["best"] for d in valid])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(n_ne, bins=np.arange(0.5, n_ne.max() + 1.5), rwidth=0.85)
    ax[0].set(title="# pure NE per instance", xlabel="number of NE")
    ax[1].hist(poa, bins=30, alpha=0.7, label="PoA (opt/worst NE)")
    ax[1].hist(pos, bins=30, alpha=0.7, label="PoS (opt/best NE)")
    ax[1].set(title="Efficiency, 4 users x 3 cells x 2 platforms", xlabel="ratio"); ax[1].legend()
    save(fig, "fig_e1_bruteforce_poa")
    summary["E1_bruteforce"] = dict(
        instances=n_inst, every_instance_has_pure_NE=bool((n_ne >= 1).all()),
        BR_always_reaches_NE=ok, mean_num_NE=float(n_ne.mean()), max_num_NE=int(n_ne.max()),
        PoA_mean=float(poa.mean()), PoA_max=float(poa.max()),
        PoS_mean=float(pos.mean()), PoS_max=float(pos.max()))


# =============================================================== E2: platform price game
def e2_price_game(cfg, grid, K=40):
    t0 = time.time()
    res = price_matrix(cfg, grid, K=K, seed=0)
    Pi = res["Pi"]
    ne = pure_nash(Pi)
    c = collusive(Pi)
    mixed = mixed_nash_2p(Pi[..., 0], Pi[..., 1])
    # best-response maps
    br1 = Pi[..., 0].argmax(axis=0)   # P1's BR (row) to each column
    br2 = Pi[..., 1].argmax(axis=1)   # P2's BR (col) to each row
    np.savetxt(f"{OUT}/E2_profit_P1.csv", Pi[..., 0], delimiter=",", fmt="%.2f",
               header="rows: m1, cols: m2, grid=" + " ".join(map(str, grid)))
    np.savetxt(f"{OUT}/E2_profit_P2.csv", Pi[..., 1], delimiter=",", fmt="%.2f",
               header="rows: m1, cols: m2, grid=" + " ".join(map(str, grid)))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["$\\pi_1$ (air quality)", "$\\pi_2$ (traffic)", "joint profit $\\pi_1+\\pi_2$"]
    mats = [Pi[..., 0], Pi[..., 1], Pi.sum(-1)]
    L = len(grid)
    for k in range(3):
        im = ax[k].imshow(mats[k], origin="lower", cmap="viridis")
        ax[k].set(xticks=range(L), yticks=range(L), xticklabels=grid, yticklabels=grid,
                  xlabel="$m_2$", ylabel="$m_1$", title=titles[k])
        plt.colorbar(im, ax=ax[k], fraction=0.046)
        ax[k].plot(range(L), br1, "w^", ms=5, label="P1 best response")
        ax[k].plot(br2, range(L), "r>", ms=5, label="P2 best response")
        ax[k].plot(c[1], c[0], "y*", ms=14, mec="k", label="collusive")
        for (i, j) in ne:
            ax[k].plot(j, i, "o", mfc="none", mec="orange", ms=14, mew=2, label="pure NE")
    ax[0].legend(fontsize=7, loc="upper left")
    for tl in ax: plt.setp(tl.get_xticklabels(), rotation=90)
    save(fig, "fig3_price_game")

    out = dict(grid=list(map(float, grid)), K=K, seconds=round(time.time() - t0, 1),
               pure_NE=[tuple(float(grid[i]) for i in idx) for idx in ne],
               collusive=(float(grid[c[0]]), float(grid[c[1]])),
               collusive_profits=Pi[c].round(2).tolist(),
               BR_P1_to_m2={float(grid[j]): float(grid[br1[j]]) for j in range(L)},
               BR_P2_to_m1={float(grid[i]): float(grid[br2[i]]) for i in range(L)})
    if mixed:
        x, y = mixed[0]
        ePi = np.array([x @ Pi[..., 0] @ y, x @ Pi[..., 1] @ y])
        out["mixed_NE_count"] = len(mixed)
        out["mixed_NE_P1"] = {float(grid[i]): round(float(p), 3) for i, p in enumerate(x) if p > 1e-6}
        out["mixed_NE_P2"] = {float(grid[i]): round(float(p), 3) for i, p in enumerate(y) if p > 1e-6}
        out["mixed_NE_expected_profits"] = ePi.round(2).tolist()
        out["mixed_NE_expected_coverage"] = float(x @ res["coverage"] @ y)
        out["mixed_NE_expected_m"] = [float(x @ grid), float(y @ grid)]
        # Calvano-style benchmark for later: Delta = (pi - pi_N) / (pi_C - pi_N), on joint profit
        out["Delta_benchmarks_joint"] = dict(pi_Nash=float(ePi.sum()), pi_coll=float(Pi[c].sum()))
        fig, ax = plt.subplots(figsize=(5, 2.8))
        wdt = 0.035
        ax.bar(grid - wdt / 2, x, wdt, label="P1"); ax.bar(grid + wdt / 2, y, wdt, label="P2")
        ax.set(xlabel="price level m", ylabel="probability", title="Mixed NE of the price game")
        ax.legend(); save(fig, "fig3b_mixed_ne")
    out["coverage_at_collusive"] = float(res["coverage"][c])
    summary["E2_price_game"] = out
    return res, out


# =============================================================== Baseline simulation
def baseline(cfg, m=(1.0, 1.0), T=500, seeds=10):
    hist = []
    covmaps = []
    for s in range(seeds):
        model = CrowdsensingModel(cfg, [FixedPrice(x) for x in m], seed=s)
        model.run(T)
        hist.append(model.history)
        covmaps.append(model.cover_count / T)
    world = model.world
    keys = ["coverage", "low_traffic_coverage", "participation", "user_net_mean",
            "profit0", "profit1", "spend0", "spend1", "wcov0", "wcov1", "qcov0", "qcov1",
            "welfare", "welfare_greedy", "sweeps", "users0", "users1"]
    arr = {k: np.array([[r[k] for r in h] for h in hist]) for k in keys}
    ratio = arr["welfare"] / arr["welfare_greedy"]

    import csv
    with open(f"{OUT}/baseline_rounds.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["seed", "t"] + keys)
        for s, h in enumerate(hist):
            for r in h:
                wr.writerow([s, r["t"]] + [r[k] for k in keys])

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    t = np.arange(T)
    def band(a, k, lab, data):
        mu = data.mean(0); sd = data.std(0)
        win = 20; kern = np.ones(win) / win
        sm = np.convolve(mu, kern, mode="valid")
        a.plot(t[win - 1:], sm, label=lab)
    band(ax[0, 0], None, "all cells", arr["coverage"]); band(ax[0, 0], None, "low-traffic cells", arr["low_traffic_coverage"])
    ax[0, 0].set(title="Coverage (20-round moving avg, 10 seeds)", ylim=(0, 1)); ax[0, 0].legend()
    band(ax[0, 1], None, "P1 air quality", arr["profit0"]); band(ax[0, 1], None, "P2 traffic", arr["profit1"])
    ax[0, 1].set(title="Platform profit per round"); ax[0, 1].legend()
    band(ax[0, 2], None, "P1", arr["spend0"]); band(ax[0, 2], None, "P2", arr["spend1"])
    ax[0, 2].set(title="Platform spending per round"); ax[0, 2].legend()
    ax[1, 0].hist(ratio.ravel(), bins=40); ax[1, 0].set(title="Welfare: user NE / greedy benchmark", xlabel="ratio")
    im = ax[1, 1].imshow(world.traffic.reshape(cfg.grid, cfg.grid), origin="lower", cmap="magma")
    lt = np.argwhere(world.low_traffic.reshape(cfg.grid, cfg.grid))
    ax[1, 1].scatter(lt[:, 1], lt[:, 0], marker="x", c="cyan", s=12, label="low-traffic")
    ax[1, 1].set(title="User traffic per cell (x = low-traffic)"); plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    ax[1, 1].legend(fontsize=7, loc="upper left")
    im = ax[1, 2].imshow(np.mean(covmaps, 0).reshape(cfg.grid, cfg.grid), origin="lower", cmap="viridis", vmin=0, vmax=1)
    road = np.argwhere(world.is_road.reshape(cfg.grid, cfg.grid))
    ax[1, 2].scatter(road[:, 1], road[:, 0], marker=".", c="w", s=6)
    ax[1, 2].set(title="P(cell covered) at fixed price (dots = roads)"); plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    save(fig, "fig4_baseline_fixed_price")

    summary["baseline_fixed_price"] = dict(
        m=list(m), T=T, seeds=seeds,
        coverage=float(arr["coverage"].mean()), low_traffic_coverage=float(arr["low_traffic_coverage"].mean()),
        participation=float(arr["participation"].mean()), user_net_mean=float(arr["user_net_mean"].mean()),
        profit=[float(arr["profit0"].mean()), float(arr["profit1"].mean())],
        spend=[float(arr["spend0"].mean()), float(arr["spend1"].mean())],
        users_per_platform=[float(arr["users0"].mean()), float(arr["users1"].mean())],
        weighted_coverage=[float(arr["wcov0"].mean()), float(arr["wcov1"].mean())],
        welfare_ratio_NE_over_greedy=dict(mean=float(ratio.mean()), min=float(ratio.min())),
        mean_BR_sweeps=float(arr["sweeps"].mean()), max_BR_sweeps=int(arr["sweeps"].max()))


def price_sweep(cfg, grid, T=150, seeds=5):
    rows = []
    for m in grid:
        vals = []
        for s in range(seeds):
            model = CrowdsensingModel(cfg, [FixedPrice(m), FixedPrice(m)], seed=100 + s,
                                      compute_benchmark=False)
            h = model.run(T)
            vals.append([np.mean([r[k] for r in h]) for k in
                         ("coverage", "low_traffic_coverage", "profit0", "profit1", "user_net_mean", "participation")])
        rows.append([m] + list(np.mean(vals, 0)))
    rows = np.array(rows)
    np.savetxt(f"{OUT}/fixed_price_sweep.csv", rows, delimiter=",", fmt="%.4f",
               header="m,coverage,low_traffic_coverage,profit_P1,profit_P2,user_net_mean,participation")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(rows[:, 0], rows[:, 1], "o-", label="all cells"); ax[0].plot(rows[:, 0], rows[:, 2], "s-", label="low-traffic")
    ax[0].set(xlabel="symmetric fixed price m", title="Coverage", ylim=(0, 1)); ax[0].legend()
    ax[1].plot(rows[:, 0], rows[:, 3], "o-", label="P1"); ax[1].plot(rows[:, 0], rows[:, 4], "s-", label="P2")
    ax[1].set(xlabel="m", title="Platform profit"); ax[1].legend()
    ax[2].plot(rows[:, 0], rows[:, 5], "o-"); ax[2].set(xlabel="m", title="Mean user net earnings")
    save(fig, "fig5_fixed_price_sweep")
    summary["fixed_price_sweep"] = rows.round(3).tolist()


if __name__ == "__main__":
    cfg = Config()
    grid = np.round(np.arange(0.2, 2.01, 0.2), 2)
    t0 = time.time()
    e1_dynamics(cfg); print("E1 dynamics done", round(time.time() - t0, 1))
    e1_toy(); print("E1 toy done")
    e1_bruteforce(); print("E1 brute force done", round(time.time() - t0, 1))
    e2_price_game(cfg, grid); print("E2 done", round(time.time() - t0, 1))
    baseline(cfg); print("baseline done", round(time.time() - t0, 1))
    price_sweep(cfg, grid); print("sweep done", round(time.time() - t0, 1))
    summary["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(summary, indent=1, default=str))
