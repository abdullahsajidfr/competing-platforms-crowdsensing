"""Mesa agent-based model on real Beijing traces (D1 full setup).

Agents
  UserAgent      taxi (T-Drive) or pedestrian (GeoLife); type honest / loyal / strategic
  PlatformAgent  2-4 apps with their own cell weights, an hourly budget and a pricing
                 mechanism: 'fixed', 'qlearn' (price level set externally by a learner),
                 or 'rival' (rival-offer learning from user reports)

One model step = one hour of the replayed week:
  1. users load their route for this hour (inactive if they have no trace that hour)
  2. platforms post tasks (leaders): bounty R_jg on cells, within budget B_j
  3. users settle to a pure NE of the congestion game by asynchronous best response
  4. payments, rejection reports (honest or inflated), platform learning, data collection
"""
from __future__ import annotations

import os, pickle
import numpy as np
import mesa

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "beijing_week.pkl")


def q_of(n, rho):
    return 1.0 - (1.0 - rho) ** np.asarray(n)


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


class City:
    """Static data: grid, neighbours, traffic, app weights, low-traffic cells."""

    def __init__(self, path=DATA, low_q=0.25):
        d = pickle.load(open(path, "rb"))
        self.rows, self.cols = d["grid"]["rows"], d["grid"]["cols"]
        self.G = self.rows * self.cols
        self.users = d["users"]
        self.taxi_visits, self.ped_visits = d["taxi_visits"], d["ped_visits"]
        rr, cc = np.divmod(np.arange(self.G), self.cols)
        self.nbrs = []
        for g in range(self.G):
            r, c = rr[g], cc[g]
            nb = [(r + dr) * self.cols + (c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
            self.nbrs.append(np.array(nb))
        T = np.log1p(self.taxi_visits); T /= T.max()
        P = np.log1p(self.ped_visits); P /= P.max()
        self.T, self.P = T, P
        # app cell weights (assumption: derived from observed taxi / pedestrian activity)
        self.app_names = ["air quality", "traffic", "noise", "weather"]
        self.W_all = np.stack([0.2 + 0.8 * P,                 # air quality: where people walk
                               0.1 + 0.9 * T,                 # traffic: where cars drive
                               0.2 + 0.4 * (T + P),           # noise: busy places of either kind
                               np.full(self.G, 0.5)])         # weather: uniform city-wide
        # exposure: user-hours in which a cell is reachable (on a route or one cell away)
        expo = np.zeros(self.G)
        for u in self.users:
            for r in u["routes"]:
                if len(r):
                    expo[np.unique(np.concatenate([r] + [self.nbrs[g] for g in r]))] += 1
        self.exposure = expo
        self.reachable_ever = expo > 0
        # low-traffic = the least-exposed 25% of cells that users do pass near at least once;
        # cells nobody ever passes near are 'unreachable' and reported separately
        thr = np.quantile(expo[self.reachable_ever], low_q)
        self.low_traffic = self.reachable_ever & (expo <= thr)


class UserAgent(mesa.Agent):
    def __init__(self, model, idx, info, c, kappa, utype, home, beta, delta):
        super().__init__(model)
        self.idx, self.kind, self.routes = idx, info["kind"], info["routes"]
        self.c, self.kappa, self.utype = c, kappa, utype
        self.home, self.beta, self.delta = home, beta, delta
        self.choice = -1
        self.earn = 0.0; self.earn_window = 0.0; self.tasks_done = 0

    # -- hour setup
    def prepare(self, hour):
        route = self.routes[hour]
        self.active = len(route) > 0
        self.choice = -1
        if not self.active:
            return
        near = np.unique(np.concatenate([self.model.city.nbrs[g] for g in route]))
        near = np.setdiff1d(near, route)
        self.reach = np.concatenate([route, near])
        self.reach_d = np.concatenate([np.zeros(len(route)), np.full(len(near), self.model.cell_km)])

    def build_actions(self, posted):
        if not self.active:
            self.aj = np.zeros(0, int); return
        M = posted.shape[0]
        aj = np.repeat(np.arange(M), len(self.reach)); ag = np.tile(self.reach, M)
        d = np.tile(self.reach_d, M)
        ok = posted[aj, ag]
        self.aj, self.ag = aj[ok], ag[ok]
        self.cost = self.c + self.kappa * d[ok]                       # real resource cost
        self.ac = self.cost - self.beta * (self.aj == self.home)      # loyalty lowers disutility
        self.offered = np.unique(self.aj)

    # -- stage-2 play
    def best_respond(self, tol=1e-9):
        if not self.active or len(self.aj) == 0:
            return False
        n, R = self.model.n, self.model.R
        k = self.choice
        if k >= 0:
            n[self.aj[k], self.ag[k]] -= 1
        u = R[self.aj, self.ag] / (n[self.aj, self.ag] + 1) - self.ac
        ucur = 0.0 if k < 0 else u[k]
        kb = int(np.argmax(u)); ub = u[kb]
        if ub < 0:
            kb, ub = -1, 0.0
        moved = ub > ucur + tol
        if moved:
            self.choice = kb
        if self.choice >= 0:
            n[self.aj[self.choice], self.ag[self.choice]] += 1
        return moved

    # -- payment and reports
    def settle(self):
        self.net = 0.0
        if not self.active or self.choice < 0:
            return
        j, g = self.aj[self.choice], self.ag[self.choice]
        x = self.model.R[j, g] / self.model.n[j, g]
        self.net = x - self.cost[self.choice]
        self.earn += self.net; self.tasks_done += 1
        if self.model.measuring:
            self.earn_window += self.net
        rep = (1 + self.delta) * x
        for jr in self.offered:                      # every app whose offer was rejected
            if jr != j:
                self.model.maxrep[jr, g] = max(self.model.maxrep[jr, g], rep)


class PlatformAgent(mesa.Agent):
    def __init__(self, model, j, W, budget, mechanism, m=1.0, bonus=0.0, reserve=0.25,
                 eta=0.2, lam=0.25, zeta=0.02, update="avg"):
        super().__init__(model)
        self.j, self.W, self.budget, self.mech = j, W, budget, mechanism
        self.m, self.bonus, self.reserve = m, bonus, reserve
        b, V, rho = model.b, model.V, model.rho
        self.Rstate = m * W * b                      # per-cell bounty (rival-offer state)
        self.cap = W * V * q_of(1, rho)              # value of one reading = break-even cap
        self.floor = 0.2 * W * b
        self.ohat = np.zeros_like(W)
        self.eta, self.lam, self.zeta, self.update = eta, lam, zeta, update

    def post(self, reachable):
        """Post bounties on reachable cells, highest weight first, while the total posted
        bounty stays within the hourly budget (so spending can never exceed B_j)."""
        b = self.model.b
        if self.mech == "rival":
            R = self.Rstate.copy()
        else:
            R = self.m * self.W * b
        R = R.copy()
        posted = np.zeros(self.model.city.G, bool)
        cand = np.flatnonzero(reachable)
        spent = 0.0
        if self.bonus > 0:                           # RQ3: reserved budget for low-traffic cells
            low = cand[self.model.city.low_traffic[cand]]
            low = low[np.argsort(-self.W[low])]
            R[low] *= (1 + self.bonus)
            cs = np.cumsum(R[low]); take = low[cs <= self.reserve * self.budget]
            posted[take] = True; spent = R[take].sum()
            cand = np.setdiff1d(cand, take)
        cand = cand[np.argsort(-self.W[cand], kind="stable")]
        cs = spent + np.cumsum(R[cand])
        posted[cand[cs <= self.budget]] = True
        self.model.R[self.j] = np.where(posted, R, 0.0)
        self.model.posted[self.j] = posted

    def learn(self):
        if self.mech != "rival":
            return
        n = self.model.n[self.j]; posted = self.model.posted[self.j]
        rep = self.model.maxrep[self.j]
        lost = rep > 0
        if self.update == "avg":
            first = lost & (self.ohat == 0)
            self.ohat = np.where(first, rep, np.where(lost, (1 - self.eta) * self.ohat + self.eta * rep, self.ohat))
        else:                                        # PACMAB-style max update
            self.ohat = np.where(lost, np.maximum(self.ohat, rep), self.ohat)
        R = self.Rstate
        won = posted & (n >= 1) & ~lost
        empty = posted & (n == 0) & ~lost
        R[lost] = np.minimum(self.ohat[lost] + self.lam, self.cap[lost])
        R[won] = np.maximum((1 - self.zeta) * R[won], self.floor[won])
        R[empty] = np.minimum((1 + self.zeta) * R[empty], self.cap[empty])


class CrowdsensingModel(mesa.Model):
    def __init__(self, city: City, M=2, mechanism="fixed", m=1.0, budget=600.0,
                 loyal_share=0.2, strategic_share=0.0, delta=0.5, beta=1.0,
                 b=5.0, V=20.0, rho=0.5, cell_km=0.5, bonus=0.0, reserve=0.25,
                 update="avg", burn_in=0, seed=0, type_seed=None):
        super().__init__(rng=seed)
        self.city, self.M = city, M
        self.b, self.V, self.rho, self.cell_km = b, V, rho, cell_km
        self.t = 0; self.burn_in = burn_in; self.measuring = burn_in == 0
        G = city.G
        self.R = np.zeros((M, G)); self.posted = np.zeros((M, G), bool)
        self.n = np.zeros((M, G), int); self.maxrep = np.zeros((M, G))
        self.nprng = np.random.default_rng(seed)

        # ---- user population: costs depend on seed; type assignment is nested across shares
        prng = np.random.default_rng(seed if type_seed is None else type_seed)
        N = len(city.users)
        kinds = np.array([u["kind"] for u in city.users])
        c = prng.uniform(1.0, 3.0, N)
        kappa = np.where(kinds == "taxi", prng.uniform(0.5, 2.0, N), prng.uniform(1.0, 4.0, N))
        order = prng.permutation(N)
        n_loyal = int(round(loyal_share * N)); n_strat = int(round(strategic_share * N))
        utype = np.array(["honest"] * N, dtype=object)
        utype[order[:n_loyal]] = "loyal"
        utype[order[n_loyal:n_loyal + n_strat]] = "strategic"   # nested: 25% subset of 50%
        home = prng.integers(M, size=N)
        self.users = []
        for i in range(N):
            self.users.append(UserAgent(self, i, city.users[i], c[i], kappa[i], utype[i], home[i],
                                        beta if utype[i] == "loyal" else 0.0,
                                        delta if utype[i] == "strategic" else 0.0))
        self.platforms = [PlatformAgent(self, j, city.W_all[j], budget, mechanism, m=m, bonus=bonus,
                                        reserve=reserve, update=update) for j in range(M)]
        self.rec = []

    # ------------------------------------------------------------------ one hour
    def stage(self, hour, prices=None):
        """Run posting + user equilibrium for this hour (no learning, no recording)."""
        for u in self.users:
            u.prepare(hour)
        reachable = np.zeros(self.city.G, bool)
        for u in self.users:
            if u.active:
                reachable[u.reach] = True
        for p in self.platforms:
            if prices is not None:
                p.m = prices[p.j]
            p.post(reachable)
        for u in self.users:
            u.build_actions(self.posted)
        self.n[:] = 0
        active = [u for u in self.users if u.active]
        for sweep in range(500):
            self.nprng.shuffle(active)
            moved = False
            for u in active:
                moved |= u.best_respond()
            if not moved:
                break
        self.sweeps = sweep + 1
        return self.outcome()

    def outcome(self):
        W = np.stack([p.W for p in self.platforms])
        covered_j = self.n >= 1
        value = (W * self.V * q_of(self.n, self.rho)).sum(1)
        spend = (self.R * covered_j).sum(1)
        cov = covered_j.any(0)
        return dict(profit=value - spend, value=value, spend=spend,
                    posted=self.posted.sum(1), completed=(self.posted & covered_j).sum(1),
                    coverage=cov.mean(), low_cov=cov[self.city.low_traffic].mean(),
                    covered_cells=cov)

    def step(self):
        prices = getattr(self, "next_prices", None)    # set externally by a Q-learner
        hour = self.t % 168
        self.measuring = self.t >= self.burn_in
        out = self.stage(hour, prices)
        self.maxrep[:] = 0
        for u in self.users:
            u.settle()
        for p in self.platforms:
            p.learn()
        act = [u for u in self.users if u.active]
        rec = dict(t=self.t, hour=hour, active_users=len(act), sweeps=self.sweeps,
                   participating=sum(u.choice >= 0 for u in act),
                   coverage=out["coverage"], low_cov=out["low_cov"])
        for j in range(self.M):
            rec[f"profit{j}"] = out["profit"][j]; rec[f"spend{j}"] = out["spend"][j]
            rec[f"posted{j}"] = out["posted"][j]; rec[f"completed{j}"] = out["completed"][j]
            rec[f"m{j}"] = (self.R[j][self.posted[j]] / (self.platforms[j].W[self.posted[j]] * self.b)).mean() \
                if self.posted[j].any() else np.nan
        for ty in ("honest", "loyal", "strategic"):
            nets = [u.net for u in act if u.utype == ty]
            rec[f"net_{ty}"] = float(np.mean(nets)) if nets else np.nan
        self.rec.append(rec)
        self.last = out
        self.t += 1

    # ------------------------------------------------------------------ summaries
    def user_table(self):
        import pandas as pd
        return pd.DataFrame(dict(idx=[u.idx for u in self.users], kind=[u.kind for u in self.users],
                                 utype=[u.utype for u in self.users], earn=[u.earn_window for u in self.users],
                                 tasks=[u.tasks_done for u in self.users],
                                 active_hours=[sum(len(r) > 0 for r in u.routes) for u in self.users]))
