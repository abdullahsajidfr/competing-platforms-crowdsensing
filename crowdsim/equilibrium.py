"""Equilibrium analysis tools.

E1: brute-force NE / PoA / PoS of tiny user games.
E2: platform price game. For every price profile m on the grid, solve the user
    subgame (async BR from the empty profile; fixed seeds = common random numbers),
    average platform profits over K hourly route draws -> M-player matrix game.
    Then: pure NE (best-response check), mixed NE for M=2 (support enumeration),
    joint-profit ("collusive") profile.
"""
from __future__ import annotations

import itertools
import numpy as np

from .env import Config, World, UserGame, platform_profit, welfare, q
from .dynamics import async_best_response, enumerate_profiles


# ------------------------------------------------------------------ E1 tiny games
def random_tiny_instance(rng, n_users=4, n_cells=3, M=2, b=5.0, m=1.0):
    W = rng.uniform(0.1, 1.0, (M, n_cells))
    kappa = rng.uniform(0.5, 2.0, n_users)
    csense = rng.uniform(1.0, 3.0, n_users)
    acts = []
    for i in range(n_users):
        reach = np.flatnonzero(rng.random(n_cells) < 0.7)
        if reach.size == 0:
            reach = np.array([rng.integers(n_cells)])
        d = rng.choice([0.0, 0.5], size=reach.size)
        aj = np.repeat(np.arange(M), reach.size)
        ag = np.tile(reach, M)
        cost = csense[i] + kappa[i] * np.tile(d, M)
        acts.append((aj, ag, cost, cost))
    R = m * W * b
    return W, kappa, acts, R


def brute_force(game: UserGame, W, V, rho):
    """Enumerate every pure profile: NE set and welfare of each."""
    ne, wel_all = [], []
    best = -np.inf
    for prof in enumerate_profiles(game):
        a = np.array(prof)
        n = game.loads(a)
        cost = sum(game.acts[i][3][k] for i, k in enumerate(a) if k >= 0)
        Wv = welfare(W, n, V, rho, cost)
        best = max(best, Wv)
        if game.is_nash(a):
            ne.append((a, Wv, game.potential(a, n)))
    return ne, best


# ------------------------------------------------------------------ E2 price game
def price_matrix(cfg: Config, grid, K=30, seed=0):
    """Expected profits pi_j(m) for every profile m in grid^M (plus coverage, user surplus)."""
    world = World(cfg, np.random.default_rng(cfg.seed))
    M, L = cfg.M, len(grid)
    # common random numbers: the same K route draws for every price profile
    rr = np.random.default_rng(seed)
    draws = [world.build_actions(world.draw_routes(rr)) for _ in range(K)]
    shape = (L,) * M
    Pi = np.zeros(shape + (M,))
    Cov = np.zeros(shape); LowCov = np.zeros(shape); UserSurplus = np.zeros(shape)
    Wel = np.zeros(shape)
    for idx in itertools.product(range(L), repeat=M):
        m = np.array([grid[k] for k in idx])
        R = m[:, None] * world.W * cfg.b
        for kk, acts in enumerate(draws):
            game = UserGame(R, acts)
            a, _, _ = async_best_response(game, np.random.default_rng(1000 + kk))
            n = game.loads(a)
            prof, _, _ = platform_profit(world.W, R, n, cfg.V, cfg.rho)
            Pi[idx] += prof / K
            cov = n.sum(0) >= 1
            Cov[idx] += cov.mean() / K
            LowCov[idx] += cov[world.low_traffic].mean() / K
            us = sum(game.utility(i, a, n) for i in range(game.N) if a[i] >= 0)
            UserSurplus[idx] += us / K
            cost = sum(acts[i][3][k] for i, k in enumerate(a) if k >= 0)
            Wel[idx] += welfare(world.W, n, cfg.V, cfg.rho, cost) / K
    return dict(Pi=Pi, coverage=Cov, low_coverage=LowCov, user_surplus=UserSurplus,
                welfare=Wel, grid=np.asarray(grid), world=world)


def pure_nash(Pi, tol=1e-9):
    """All pure NE of an M-player matrix game Pi[idx..., j]."""
    M = Pi.shape[-1]
    out = []
    for idx in itertools.product(*[range(s) for s in Pi.shape[:-1]]):
        ok = True
        for j in range(M):
            sl = list(idx); sl[j] = slice(None)
            if Pi[tuple(sl) + (j,)].max() > Pi[idx + (j,)] + tol:
                ok = False; break
        if ok:
            out.append(idx)
    return out


def collusive(Pi):
    """Joint-profit maximizing profile."""
    tot = Pi.sum(-1)
    return np.unravel_index(np.argmax(tot), tot.shape)


def mixed_nash_2p(A, B, tol=1e-9):
    """Support enumeration for a bimatrix game (row payoff A, column payoff B).
    Returns list of (x, y). Equal-size supports (nondegenerate case)."""
    m, n = A.shape
    sols = []
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                # y on J makes rows in I indifferent; x on I makes cols in J indifferent
                def solve(P, S, T):
                    # find prob p on T s.t. P[S][:,T] p = v * 1, sum p = 1
                    Msub = np.zeros((k + 1, k + 1))
                    Msub[:k, :k] = P[np.ix_(S, T)]
                    Msub[:k, k] = -1
                    Msub[k, :k] = 1
                    rhs = np.zeros(k + 1); rhs[k] = 1
                    try:
                        sol = np.linalg.solve(Msub, rhs)
                    except np.linalg.LinAlgError:
                        return None
                    return sol
                sy = solve(A, I, J)
                sx = solve(B.T, J, I)
                if sy is None or sx is None:
                    continue
                y = np.zeros(n); y[list(J)] = sy[:k]
                x = np.zeros(m); x[list(I)] = sx[:k]
                if (y < -tol).any() or (x < -tol).any():
                    continue
                if (A @ y).max() > sy[k] + 1e-7 or (x @ B).max() > sx[k] + 1e-7:
                    continue
                sols.append((x.clip(0), y.clip(0)))
    return sols
