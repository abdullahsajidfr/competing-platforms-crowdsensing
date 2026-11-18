"""Round-by-round model: platforms (leaders) post rewards, users (followers) settle
to a pure NE of the congestion game via best-response dynamics, then payments,
reports and metrics are recorded. Baseline = fixed posted price for every platform."""
from __future__ import annotations

import numpy as np

from .env import Config, World, UserGame, platform_profit, welfare, q
from .dynamics import async_best_response


# ---------------------------------------------------------------- platform strategies
class FixedPrice:
    """Mechanism 1 (baseline): constant price level m_j."""
    name = "fixed"

    def __init__(self, m):
        self.m = float(m)

    def price(self, t, state):
        return self.m

    def update(self, *args, **kw):
        pass


class QLearningPricer:
    """Mechanism 2 (next step): state = last round's price levels of all platforms,
    action = own price index, reward = profit. eps_t = exp(-beta_eps * t)."""
    name = "qlearning"

    def __init__(self, grid, M, rng, alpha=0.1, gamma=0.95, beta_eps=1e-4, q0=0.0):
        self.grid = np.asarray(grid, float)
        self.K = len(self.grid)
        self.M = M
        self.rng = rng
        self.alpha, self.gamma, self.beta_eps = alpha, gamma, beta_eps
        self.Q = np.full((self.K ** M, self.K), q0)
        self.last_idx = None

    def _s(self, state_idx):
        s = 0
        for x in state_idx:
            s = s * self.K + int(x)
        return s

    def price_index(self, t, state_idx):
        eps = np.exp(-self.beta_eps * t)
        if self.rng.random() < eps:
            k = int(self.rng.integers(self.K))
        else:
            row = self.Q[self._s(state_idx)]
            k = int(self.rng.choice(np.flatnonzero(row == row.max())))
        self.last_idx = k
        return k

    def update(self, state_idx, k, reward, next_state_idx):
        s, s2 = self._s(state_idx), self._s(next_state_idx)
        target = reward + self.gamma * self.Q[s2].max()
        self.Q[s, k] += self.alpha * (target - self.Q[s, k])


# ---------------------------------------------------------------- greedy welfare benchmark
def greedy_welfare(world: World, acts, V, rho):
    """Greedy assignment maximizing W(a) (monotone submodular value minus modular cost,
    one action per user = partition matroid). Used as a scalable 'near-optimal' benchmark."""
    M, G = world.W.shape
    n = np.zeros((M, G), int)
    free = np.ones(len(acts), bool)
    total_cost = 0.0
    while True:
        best, bi, bk = 0.0, -1, -1
        for i in np.flatnonzero(free):
            aj, ag, _, cost = acts[i]
            gain = world.W[aj, ag] * V * (q(n[aj, ag] + 1, rho) - q(n[aj, ag], rho)) - cost
            k = int(np.argmax(gain))
            if gain[k] > best:
                best, bi, bk = gain[k], i, k
        if bi < 0:
            break
        aj, ag, _, cost = acts[bi]
        n[aj[bk], ag[bk]] += 1
        total_cost += cost[bk]
        free[bi] = False
    return welfare(world.W, n, V, rho, total_cost)


# ---------------------------------------------------------------- model
class CrowdsensingModel:
    def __init__(self, cfg: Config, strategies, delta=None, seed=0, compute_benchmark=True):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = World(cfg, np.random.default_rng(cfg.seed))   # same map/users per cfg.seed
        self.strategies = strategies
        # report inflation per user: 0 = honest (baseline)
        self.delta = np.zeros(cfg.N) if delta is None else np.asarray(delta, float)
        self.t = 0
        self.history = []
        self.reports = []           # (t, user, rejected platform j, cell g, reported pay)
        self.cover_count = np.zeros(self.world.G)
        self.compute_benchmark = compute_benchmark

    def step(self):
        cfg, w = self.cfg, self.world
        routes = w.draw_routes(self.rng)
        acts = w.build_actions(routes)
        m = np.array([s.price(self.t, None) for s in self.strategies])
        R = m[:, None] * w.W * cfg.b
        game = UserGame(R, acts)
        a, sweeps, _ = async_best_response(game, self.rng)
        n = game.loads(a)

        # payments, detours, reports
        pay = np.zeros(cfg.N); det = np.zeros(cfg.N)
        for i, k in enumerate(a):
            if k < 0:
                continue
            aj, ag, ac, cost = acts[i]
            j, g = aj[k], ag[k]
            pay[i] = R[j, g] / n[j, g]
            det[i] = cost[k]
            for jr in range(cfg.M):
                if jr != j:
                    self.reports.append((self.t, i, jr, g, (1 + self.delta[i]) * pay[i]))

        profit, value, spend = platform_profit(w.W, R, n, cfg.V, cfg.rho)
        covered = (n.sum(0) >= 1)
        self.cover_count += covered
        Wel = welfare(w.W, n, cfg.V, cfg.rho, det.sum())
        rec = dict(
            t=self.t, sweeps=sweeps,
            participation=float((a >= 0).mean()),
            coverage=float(covered.mean()),
            low_traffic_coverage=float(covered[w.low_traffic].mean()),
            user_net_mean=float((pay - det).mean()),
            welfare=Wel,
        )
        if self.compute_benchmark:
            rec["welfare_greedy"] = greedy_welfare(w, acts, cfg.V, cfg.rho)
        for j in range(cfg.M):
            cj = n[j] >= 1
            rec[f"m{j}"] = m[j]
            rec[f"profit{j}"] = profit[j]
            rec[f"spend{j}"] = spend[j]
            rec[f"wcov{j}"] = float((w.W[j] * cj).sum() / w.W[j].sum())
            rec[f"qcov{j}"] = float((w.W[j] * q(n[j], cfg.rho)).sum() / w.W[j].sum())
            rec[f"users{j}"] = int(n[j].sum())
        self.history.append(rec)
        self.t += 1
        return rec

    def run(self, T):
        for _ in range(T):
            self.step()
        return self.history
