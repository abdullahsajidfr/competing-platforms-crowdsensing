"""Experiment 2 (learning prices) and Experiment 3 (lying users).

Experiment 2: Q-learning platforms (Calvano et al. 2020 style)
    Each round: both platforms pick a price level m_j from a grid, users settle to a
    pure NE of the congestion game, platforms earn profit. To make ~10^6 rounds
    affordable, user outcomes are precomputed for a bank of K hourly route draws
    for every price profile ("replaying a week of traces"); each round draws one
    hour from the bank at random. This is exact: it's the same stage game.

Experiment 3: rival-offer learning (simplified PACMAB) + misreporting users
    Each platform keeps a per-block estimate o_hat[j,g] of the best per-person pay a
    rival offers there, built from users' rejection reports, and posts
        R[j,g] = min(o_hat[j,g] + lam, cap[j,g])     after losing a user at g,
        R[j,g] = max((1 - zeta) R[j,g], floor[j,g])   after winning g without losing anyone,
        R[j,g] = min((1 + zeta) R[j,g], cap[j,g])     if nobody sensed g for j and no report came.
    Honest users report the per-person pay x they got from the app they worked for;
    liars report (1 + delta) x.
"""
from __future__ import annotations

import itertools
import numpy as np

from .env import Config, World, UserGame, platform_profit, q
from .dynamics import async_best_response
from .equilibrium import pure_nash, collusive, mixed_nash_2p


# =========================================================================== Experiment 2
def build_bank(cfg: Config, grid, K=200, seed=0):
    """profit[i1, i2, k, j], coverage[i1,i2,k], low[i1,i2,k], usernet[i1,i2,k] for hour k."""
    world = World(cfg, np.random.default_rng(cfg.seed))
    rr = np.random.default_rng(seed)
    hours = [world.build_actions(world.draw_routes(rr)) for _ in range(K)]
    L = len(grid)
    prof = np.zeros((L, L, K, 2)); cov = np.zeros((L, L, K)); low = np.zeros((L, L, K))
    unet = np.zeros((L, L, K))
    for i1, i2 in itertools.product(range(L), repeat=2):
        R = np.array([grid[i1], grid[i2]])[:, None] * world.W * cfg.b
        for k, acts in enumerate(hours):
            g = UserGame(R, acts)
            a, _, _ = async_best_response(g, np.random.default_rng(10_000 + k))
            n = g.loads(a)
            prof[i1, i2, k] = platform_profit(world.W, R, n, cfg.V, cfg.rho)[0]
            c = n.sum(0) >= 1
            cov[i1, i2, k] = c.mean(); low[i1, i2, k] = c[world.low_traffic].mean()
            unet[i1, i2, k] = sum(g.utility(i, a, n) for i in range(g.N) if a[i] >= 0) / g.N
    return dict(grid=np.asarray(grid), profit=prof, coverage=cov, low=low, usernet=unet,
                Pi=prof.mean(2), world=world)


def benchmarks(Pi):
    """Static Nash (pure if it exists, else mixed) and joint-profit benchmarks."""
    ne = pure_nash(Pi)
    c = collusive(Pi)
    out = dict(collusive=c, pi_C=float(Pi[c].sum()), pi_C_each=Pi[c].tolist())
    if ne:
        # if several pure NE, use the one with the lowest joint profit (most competitive)
        idx = min(ne, key=lambda x: Pi[x].sum())
        out.update(nash_type="pure", nash=idx, pi_N=float(Pi[idx].sum()),
                   pi_N_each=Pi[idx].tolist(), all_pure=ne)
    else:
        x, y = mixed_nash_2p(Pi[..., 0], Pi[..., 1])[0]
        e = [float(x @ Pi[..., 0] @ y), float(x @ Pi[..., 1] @ y)]
        out.update(nash_type="mixed", nash=(x, y), pi_N=sum(e), pi_N_each=e)
    return out


def run_qlearning(bank, seed=0, alpha=0.15, gamma=0.95, beta=1e-5, T_max=1_000_000,
                  stable_for=50_000, memory=True, record_every=1000):
    """Two independent Q-learners. State = last round's (m1, m2) index pair
    (or a single dummy state if memory=False). eps_t = exp(-beta t)."""
    rng = np.random.default_rng(seed)
    L = len(bank["grid"]); K = bank["profit"].shape[2]
    prof = bank["profit"]; Pi = bank["Pi"]
    S = L * L if memory else 1
    Q = np.zeros((2, S, L))
    # Calvano init: discounted payoff against a uniformly random rival
    Q[0] += Pi[..., 0].mean(1) / (1 - gamma)
    Q[1] += Pi[..., 1].mean(0) / (1 - gamma)
    s = int(rng.integers(S))
    greedy_prev = Q.argmax(2).copy(); stable = 0
    trace = []
    eps_draw = rng.random((T_max, 2)); rand_act = rng.integers(L, size=(T_max, 2))
    hour = rng.integers(K, size=T_max)
    for t in range(T_max):
        eps = np.exp(-beta * t)
        a0 = rand_act[t, 0] if eps_draw[t, 0] < eps else int(Q[0, s].argmax())
        a1 = rand_act[t, 1] if eps_draw[t, 1] < eps else int(Q[1, s].argmax())
        r = prof[a0, a1, hour[t]]
        s2 = a0 * L + a1 if memory else 0
        Q[0, s, a0] += alpha * (r[0] + gamma * Q[0, s2].max() - Q[0, s, a0])
        Q[1, s, a1] += alpha * (r[1] + gamma * Q[1, s2].max() - Q[1, s, a1])
        s = s2
        if t % record_every == 0:
            trace.append((t, a0, a1, r[0], r[1]))
        if t % 1000 == 999:
            g = Q.argmax(2)
            if np.array_equal(g, greedy_prev):
                stable += 1000
            else:
                stable = 0; greedy_prev = g.copy()
            if stable >= stable_for and eps < 0.01:
                break
    return dict(Q=Q, t_end=t + 1, converged=stable >= stable_for, trace=np.array(trace),
                memory=memory, last_state=s)


def greedy_path(Q, L, s0, steps=60, memory=True, force=None):
    """Play greedy strategies from state s0. force = {step: (agent, action)} overrides."""
    s = s0; path = []
    for t in range(steps):
        a = [int(Q[0, s].argmax()), int(Q[1, s].argmax())]
        if force and t in force:
            ag, act = force[t]; a[ag] = act
        path.append(tuple(a))
        s = a[0] * L + a[1] if memory else 0
    return path


def limit_outcome(res, bank):
    """Average expected outcome on the cycle the greedy strategies settle into."""
    L = len(bank["grid"]); Pi = bank["Pi"]
    path = greedy_path(res["Q"], L, res["last_state"], steps=200, memory=res["memory"])[100:]
    prof = np.array([Pi[a] for a in path])
    cov = np.mean([bank["coverage"][a].mean() for a in path])
    low = np.mean([bank["low"][a].mean() for a in path])
    unet = np.mean([bank["usernet"][a].mean() for a in path])
    # cycle length
    cyc = next((p for p in range(1, 50) if path[-1] == path[-1 - p]), None)
    m = np.array([[bank["grid"][a[0]], bank["grid"][a[1]]] for a in path])
    return dict(profit=prof.mean(0), joint=float(prof.sum(1).mean()), coverage=float(cov),
                low=float(low), usernet=float(unet), m_mean=m.mean(0), cycle_len=cyc,
                path_tail=path[-cyc:] if cyc else path[-5:])


def q_loss(res, bank):
    """Calvano-style equilibrium check on the limit path: how much more a platform could
    earn (one-shot deviation, then continue greedy-on-path) — reported as % of profit,
    using each agent's own learned Q as continuation value."""
    L = len(bank["grid"]); Q = res["Q"]
    path = greedy_path(Q, L, res["last_state"], steps=150, memory=res["memory"])[100:]
    gaps = []
    for a in path:
        s = a[0] * L + a[1] if res["memory"] else 0
        for j in range(2):
            gaps.append((Q[j, s].max() - Q[j, s][Q[j, s].argmax()]) == 0)
    return float(np.mean(gaps))


def impulse_response(res, bank, deviator=0, horizon=15):
    """Force one platform to deviate for one round to its static best response that steals
    users (= the price that maximizes its one-shot profit given the rival's current price),
    then let both follow their learned greedy strategies."""
    L = len(bank["grid"]); Pi = bank["Pi"]; Q = res["Q"]
    pre = greedy_path(Q, L, res["last_state"], steps=120, memory=res["memory"])
    s0 = pre[-1][0] * L + pre[-1][1] if res["memory"] else 0
    base = greedy_path(Q, L, s0, steps=1, memory=res["memory"])[0]
    rival = base[1 - deviator]
    if deviator == 0:
        dev = int(Pi[:, rival, 0].argmax())
    else:
        dev = int(Pi[rival, :, 1].argmax())
    path = [pre[-1]] + greedy_path(Q, L, s0, steps=horizon, memory=res["memory"],
                                   force={0: (deviator, dev)})
    return np.array([[bank["grid"][a[0]], bank["grid"][a[1]]] for a in path])


# =========================================================================== Experiment 3
class RivalOfferModel:
    """Both platforms use rival-offer learning; users settle to NE each round."""

    def __init__(self, cfg: Config, liar_share=0.0, delta=0.5, eta=0.2, lam=0.25, zeta=0.02,
                 m0=1.0, floor_m=0.2, update="avg", seed=0, liar_order_seed=99):
        self.cfg = cfg
        self.world = World(cfg, np.random.default_rng(cfg.seed))
        self.rng = np.random.default_rng(seed)
        w = self.world
        # nested liar sets: the 25% liars are a subset of the 50% liars
        order = np.random.default_rng(liar_order_seed).permutation(cfg.N)
        self.liar = np.zeros(cfg.N, bool); self.liar[order[:int(round(liar_share * cfg.N))]] = True
        self.delta = np.where(self.liar, delta, 0.0)
        self.R = m0 * w.W * cfg.b                          # posted bounties R[j, g]
        self.cap = w.W * cfg.V * q(1, cfg.rho)              # value of a single reading
        self.floor = floor_m * w.W * cfg.b
        self.ohat = np.zeros_like(self.R)
        self.eta, self.lam, self.zeta, self.update = eta, lam, zeta, update
        self.t = 0
        self.hist = []
        self.earn = np.zeros(cfg.N)                         # cumulative net earnings

    def step(self):
        cfg, w = self.cfg, self.world
        acts = w.build_actions(w.draw_routes(self.rng))
        R = self.R.copy()
        game = UserGame(R, acts)
        a, _, _ = async_best_response(game, self.rng)
        n = game.loads(a)
        net = np.zeros(cfg.N)
        maxrep = np.zeros_like(R)                           # max report received per (j, g)
        for i, k in enumerate(a):
            if k < 0:
                continue
            aj, ag, ac, cost = acts[i]
            j, g = aj[k], ag[k]
            x = R[j, g] / n[j, g]
            net[i] = x - cost[k]
            for jr in range(cfg.M):
                if jr != j:
                    maxrep[jr, g] = max(maxrep[jr, g], (1 + self.delta[i]) * x)
        self.earn += net
        prof, value, spend = platform_profit(w.W, R, n, cfg.V, cfg.rho)

        # platform updates
        lost = maxrep > 0
        if self.update == "avg":
            first = lost & (self.ohat == 0)
            self.ohat = np.where(first, maxrep, self.ohat)
            self.ohat = np.where(lost & ~first, (1 - self.eta) * self.ohat + self.eta * maxrep, self.ohat)
        else:                                               # PACMAB-style: theta = max(theta, report)
            self.ohat = np.where(lost, np.maximum(self.ohat, maxrep), self.ohat)
        won_clean = (n >= 1) & ~lost
        newR = self.R.copy()
        newR[lost] = np.minimum(self.ohat[lost] + self.lam, self.cap[lost])
        newR[won_clean] = np.maximum((1 - self.zeta) * self.R[won_clean], self.floor[won_clean])
        empty = (n == 0) & ~lost
        newR[empty] = np.minimum((1 + self.zeta) * self.R[empty], self.cap[empty])
        self.R = newR

        cov = n.sum(0) >= 1
        mlev = (R / (w.W * cfg.b)).mean(1)                  # average price level posted
        self.hist.append(dict(
            t=self.t, profit0=prof[0], profit1=prof[1], spend0=spend[0], spend1=spend[1],
            m0=mlev[0], m1=mlev[1], coverage=cov.mean(), low=cov[w.low_traffic].mean(),
            net_liar=net[self.liar].mean() if self.liar.any() else np.nan,
            net_honest=net[~self.liar].mean(), net_all=net.mean(),
            overpay=float((R * (n >= 1)).sum())))
        self.t += 1
        self.last_net = net

    def run(self, T):
        for _ in range(T):
            self.step()
        return self
