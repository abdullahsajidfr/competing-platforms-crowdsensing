"""
Environment for the competing-platform crowdsensing game (D1 baseline).

Notation follows the D1 brief (after the renames):
  G        set of grid cells, g in G (cell_km x cell_km squares)
  M        number of platforms, j in {0..M-1}
  N        number of users, i in {0..N-1}
  w[j,g]   platform j's weight on cell g (0..1)
  b        base reward (money scale)
  m[j]     platform j's price level (multiplier) -> R[j,g] = m[j] * w[j,g] * b
  V        value of fully covering a weight-1 cell
  rho      prob. that a single reading is good;  q(n) = 1 - (1-rho)^n
  kappa_i  user i's private detour cost per km
  d_i(g)   detour (km) from user i's route to cell g (0 if on route)
  h_i, beta_i  loyal user's home platform and loyalty bonus
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    grid: int = 10                 # 10 x 10 grid
    cell_km: float = 0.5           # 500 m cells
    M: int = 2                     # platforms
    N: int = 50                    # users
    b: float = 5.0                 # base reward
    V: float = 20.0                # value of a fully covered weight-1 cell
    rho: float = 0.5               # single-reading quality
    kappa_low: float = 0.5         # kappa_i ~ U[kappa_low, kappa_high] per km
    kappa_high: float = 2.0
    sense_low: float = 1.0         # c_i ~ U[sense_low, sense_high]: per-task sensing cost
    sense_high: float = 3.0        # (battery, data, time); part of user i's private type
    route_len: int = 3             # random-walk steps per round (route has <= 4 cells)
    detour_radius: int = 1         # cells reachable within this Manhattan distance of route
    road_rows: tuple = (2, 7)
    road_cols: tuple = (2, 7)
    hotspots: tuple = ((3.0, 3.0), (6.5, 6.0))
    hotspot_sigma: float = 1.6
    loyal_share: float = 0.0       # baseline: no loyal users
    beta: float = 1.0
    low_traffic_quantile: float = 0.25
    seed: int = 0


MOVES = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])


def q(n, rho):
    """Data quality of a cell sensed by n users (submodular in n)."""
    return 1.0 - (1.0 - rho) ** np.asarray(n)


class World:
    """Static part of the environment: map, platform weights, user population."""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        L = cfg.grid
        self.G = L * L
        rr, cc = np.divmod(np.arange(self.G), L)
        self.rc = np.stack([rr, cc], 1)
        # Manhattan distance between cells (in cells)
        self.D = np.abs(rr[:, None] - rr[None, :]) + np.abs(cc[:, None] - cc[None, :])

        # cell types: road (arterial rows/cols) vs residential
        self.is_road = np.isin(rr, cfg.road_rows) | np.isin(cc, cfg.road_cols)

        # platform weights w[j, g]
        W = np.zeros((cfg.M, self.G))
        W[0] = np.where(self.is_road, 0.3, 1.0)          # air quality: residential
        if cfg.M >= 2:
            W[1] = np.where(self.is_road, 1.0, 0.2)      # traffic: roads
        for j in range(2, cfg.M):                         # extra platforms: random mix
            W[j] = rng.uniform(0.1, 1.0, self.G)
        self.W = W

        # users
        N = cfg.N
        hs = np.array(cfg.hotspots)
        which = rng.integers(len(hs), size=N)
        anchors = np.rint(hs[which] + rng.normal(0, cfg.hotspot_sigma, (N, 2)))
        anchors = np.clip(anchors, 0, L - 1).astype(int)
        self.anchor = anchors[:, 0] * L + anchors[:, 1]
        self.kappa = rng.uniform(cfg.kappa_low, cfg.kappa_high, N)
        self.csense = rng.uniform(cfg.sense_low, cfg.sense_high, N)
        self.loyal = rng.random(N) < cfg.loyal_share
        self.home = rng.integers(cfg.M, size=N)
        self.beta = np.where(self.loyal, cfg.beta, 0.0)

        # traffic map and low-traffic cells (estimated from the mobility model)
        mc_rng = np.random.default_rng(cfg.seed + 12345)
        visits = np.zeros(self.G)
        for _ in range(300):
            for r in self.draw_routes(mc_rng):
                visits[r] += 1
        self.traffic = visits / 300.0
        thr = np.quantile(self.traffic, cfg.low_traffic_quantile)
        self.low_traffic = self.traffic <= thr

    # ------------------------------------------------------------------ routes
    def draw_routes(self, rng):
        """R_i(t): random walk of route_len steps from each user's anchor."""
        L = self.cfg.grid
        routes = []
        for i in range(self.cfg.N):
            r, c = divmod(int(self.anchor[i]), L)
            cells = [r * L + c]
            for _ in range(self.cfg.route_len):
                dr, dc = MOVES[rng.integers(4)]
                r = min(max(r + dr, 0), L - 1)
                c = min(max(c + dc, 0), L - 1)
                cells.append(r * L + c)
            routes.append(np.unique(cells))
        return routes

    def build_actions(self, routes):
        """For each user: arrays (aj, ag, ac, cost) of feasible actions (j,g).
        cost = c_i + kappa_i * d_i(g)        (real resource cost, enters welfare)
        ac   = cost - beta_i * 1[j == h_i]   (private disutility used in u_i and Phi)
        The null action (do nothing) is handled separately (utility 0)."""
        M = self.cfg.M
        acts = []
        for i, route in enumerate(routes):
            dist = self.D[route].min(axis=0)
            reach = np.flatnonzero(dist <= self.cfg.detour_radius)
            d_km = dist[reach] * self.cfg.cell_km
            aj = np.repeat(np.arange(M), len(reach))
            ag = np.tile(reach, M)
            ad = np.tile(d_km, M)
            cost = self.csense[i] + self.kappa[i] * ad
            ac = cost - self.beta[i] * (aj == self.home[i])
            acts.append((aj, ag, ac, cost))
        return acts


class UserGame:
    """User subgame for fixed rewards R[j,g]: a congestion game.

    Resource e = (j, g); load n[j,g]; user payoff on e = R[e]/n[e] - kappa_i d_i(g)
    (+ beta_i if j is home). Profile a[i] = index into user i's action list, -1 = null.
    Rosenthal potential: Phi(a) = sum_e R[e] * H(n_e) - sum_i c_i(a_i),  H = harmonic number.
    """

    def __init__(self, R, acts):
        self.R = np.asarray(R, float)
        self.acts = acts
        self.N = len(acts)
        self.M, self.G = self.R.shape

    # -- helpers
    def loads(self, a):
        n = np.zeros((self.M, self.G), int)
        for i, k in enumerate(a):
            if k >= 0:
                aj, ag, _, _ = self.acts[i]
                n[aj[k], ag[k]] += 1
        return n

    def utility(self, i, a, n=None):
        n = self.loads(a) if n is None else n
        k = a[i]
        if k < 0:
            return 0.0
        aj, ag, ac, _ = self.acts[i]
        return self.R[aj[k], ag[k]] / n[aj[k], ag[k]] - ac[k]

    def action_utils(self, i, n_minus):
        """Utility of every action of user i given loads WITHOUT user i."""
        aj, ag, ac, _ = self.acts[i]
        return self.R[aj, ag] / (n_minus[aj, ag] + 1) - ac

    def potential(self, a, n=None):
        n = self.loads(a) if n is None else n
        nmax = int(n.max()) if n.size else 0
        H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, nmax + 1))])
        phi = float((self.R * H[n]).sum())
        for i, k in enumerate(a):
            if k >= 0:
                phi -= self.acts[i][2][k]
        return phi

    def is_nash(self, a, tol=1e-9):
        n = self.loads(a)
        for i in range(self.N):
            k = a[i]
            nm = n.copy()
            if k >= 0:
                aj, ag, _, _ = self.acts[i]
                nm[aj[k], ag[k]] -= 1
            u = self.action_utils(i, nm)
            ucur = 0.0 if k < 0 else u[k]
            best = max(0.0, u.max() if u.size else 0.0)
            if best > ucur + tol:
                return False
        return True


def platform_profit(W, R, n, V, rho):
    """pi_j = sum_g w_jg V q(n_jg) - sum_g R_jg 1[n_jg >= 1]   (budget not binding)."""
    value = (W * V * q(n, rho)).sum(axis=1)
    spend = (R * (n >= 1)).sum(axis=1)
    return value - spend, value, spend


def welfare(W, n, V, rho, detour_cost_total):
    """W(a) = sum_j sum_g w V q(n) - sum_i (c_i + kappa_i d_i)   (payments are transfers)."""
    return float((W * V * q(n, rho)).sum() - detour_cost_total)
