"""Multi-homing variant: a user picks ONE block, takes ONE reading, and may upload it
to app 1, app 2, or both. Action = (block g, set S of apps). Uploading to an extra
app costs e (extra effort; e = 0 means sending twice is free).

u_i = sum_{j in S} R_jg / n_jg  -  c_i - kappa_i d_i(g) - e (|S| - 1)

Each action uses a SET of resources (j, g), so this is still a congestion game and
Rosenthal's potential still applies:  Phi = sum_(j,g) R_jg H(n_jg) - sum_i cost_i.
"""
from __future__ import annotations

import itertools
import numpy as np

from .env import Config, World, platform_profit, q


def build_actions_mh(world: World, routes, e=0.0):
    M = world.cfg.M
    subsets = [s for r in range(1, M + 1) for s in itertools.combinations(range(M), r)]
    acts = []
    for i, route in enumerate(routes):
        dist = world.D[route].min(axis=0)
        reach = np.flatnonzero(dist <= world.cfg.detour_radius)
        base = world.csense[i] + world.kappa[i] * dist[reach] * world.cfg.cell_km
        mask, ag, cost = [], [], []
        for S in subsets:
            mrow = np.zeros(M, bool); mrow[list(S)] = True
            for g, c in zip(reach, base):
                mask.append(mrow); ag.append(g); cost.append(c + e * (len(S) - 1))
        acts.append((np.array(mask), np.array(ag), np.array(cost)))
    return acts


class MHGame:
    def __init__(self, R, acts):
        self.R, self.acts, self.N = np.asarray(R, float), acts, len(acts)
        self.M, self.G = self.R.shape

    def _apply(self, n, i, k, s):
        if k >= 0:
            mask, ag, _ = self.acts[i]
            n[mask[k], ag[k]] += s

    def loads(self, a):
        n = np.zeros((self.M, self.G), int)
        for i, k in enumerate(a):
            self._apply(n, i, k, 1)
        return n

    def action_utils(self, i, n_minus):
        mask, ag, cost = self.acts[i]
        share = self.R[:, ag] / (n_minus[:, ag] + 1)          # M x K
        return (share.T * mask).sum(1) - cost

    def potential(self, a):
        n = self.loads(a)
        H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, max(n.max(), 1) + 1))])
        return float((self.R * H[n]).sum() - sum(self.acts[i][2][k] for i, k in enumerate(a) if k >= 0))

    def best_response(self, rng, max_sweeps=500, tol=1e-9):
        a = np.full(self.N, -1); n = self.loads(a)
        for sweep in range(1, max_sweeps + 1):
            moved = False
            for i in rng.permutation(self.N):
                k = a[i]; self._apply(n, i, k, -1)
                u = self.action_utils(i, n)
                ucur = 0.0 if k < 0 else u[k]
                kb = int(np.argmax(u)); ub = u[kb]
                if ub < 0: kb, ub = -1, 0.0
                if ub > ucur + tol:
                    a[i] = kb; moved = True
                self._apply(n, i, a[i], 1)
            if not moved:
                return a, sweep
        return a, max_sweeps


def evaluate(cfg: Config, m, K=40, e=0.0, seed=0):
    """Average outcomes over K hourly route draws at price profile m."""
    world = World(cfg, np.random.default_rng(cfg.seed))
    rr = np.random.default_rng(seed)
    R = np.asarray(m)[:, None] * world.W * cfg.b
    out = dict(profit=np.zeros(cfg.M), coverage=0.0, low=0.0, both=0.0, user_net=0.0)
    for kk in range(K):
        acts = build_actions_mh(world, world.draw_routes(rr), e)
        g = MHGame(R, acts)
        a, _ = g.best_response(np.random.default_rng(1000 + kk))
        n = g.loads(a)
        prof, _, _ = platform_profit(world.W, R, n, cfg.V, cfg.rho)
        out["profit"] += prof / K
        cov = n.sum(0) >= 1
        out["coverage"] += cov.mean() / K
        out["low"] += cov[world.low_traffic].mean() / K
        active = a >= 0
        out["both"] += (np.mean([acts[i][0][k].sum() == cfg.M for i, k in enumerate(a) if k >= 0])
                        if active.any() else 0.0) / K
        net = 0.0
        for i, k in enumerate(a):
            if k >= 0:
                nm = n.copy(); g._apply(nm, i, k, -1)
                net += g.action_utils(i, nm)[k]
        out["user_net"] += net / cfg.N / K
    return out


def price_matrix_mh(cfg: Config, grid, K=40, e=0.0):
    L = len(grid)
    Pi = np.zeros((L, L, 2))
    for i, j in itertools.product(range(L), repeat=2):
        Pi[i, j] = evaluate(cfg, (grid[i], grid[j]), K=K, e=e)["profit"]
    return Pi
