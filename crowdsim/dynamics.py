"""User-side learning dynamics on the congestion game (Lectures 7 and 9)."""
from __future__ import annotations

import itertools
import numpy as np

from .env import UserGame


def _remove(game, n, i, k):
    if k >= 0:
        aj, ag, _, _ = game.acts[i]
        n[aj[k], ag[k]] -= 1


def _add(game, n, i, k):
    if k >= 0:
        aj, ag, _, _ = game.acts[i]
        n[aj[k], ag[k]] += 1


def async_best_response(game: UserGame, rng, a0=None, max_sweeps=500, tol=1e-9,
                        record=False, check_potential=False):
    """Asynchronous (one user at a time) best response. In a potential game every
    strict improvement raises Phi by exactly the deviator's gain, so this terminates
    at a pure NE. Returns (a, sweeps, phi_trace)."""
    N = game.N
    a = np.full(N, -1) if a0 is None else np.array(a0)
    n = game.loads(a)
    trace = [game.potential(a, n)] if record else None
    for sweep in range(1, max_sweeps + 1):
        moved = False
        for i in rng.permutation(N):
            k = a[i]
            _remove(game, n, i, k)
            u = game.action_utils(i, n)
            ucur = 0.0 if k < 0 else u[k]
            kb = int(np.argmax(u)) if u.size else -1
            ub = u[kb] if u.size else -np.inf
            if ub <= 0.0 and 0.0 > ucur + tol:      # best is to do nothing
                kb, ub = -1, 0.0
            if ub > ucur + tol:
                if check_potential:
                    phi_before = game.potential(a)
                a[i] = kb
                moved = True
                if check_potential:
                    dphi = game.potential(a) - phi_before
                    assert abs(dphi - (ub - ucur)) < 1e-7, (dphi, ub - ucur)
            _add(game, n, i, a[i])
            if record:
                trace.append(game.potential(a, n))
        if not moved:
            return a, sweep, trace
    return a, max_sweeps, trace


def br_with_inertia(game: UserGame, rng, p_inertia=0.7, max_steps=2000, a0=None, tol=1e-9):
    """Simultaneous best response with inertia (Lecture 7): every period each user
    keeps its action w.p. p_inertia, otherwise switches to a best response to a_{t-1}.
    Phi need not increase every step, but play reaches a pure NE w.p. 1."""
    N = game.N
    a = np.full(N, -1) if a0 is None else np.array(a0)
    trace = [game.potential(a)]
    for t in range(max_steps):
        n = game.loads(a)
        new = a.copy()
        any_improving = False
        for i in range(N):
            k = a[i]
            nm = n.copy(); _remove(game, nm, i, k)
            u = game.action_utils(i, nm)
            ucur = 0.0 if k < 0 else u[k]
            kb = int(np.argmax(u)); ub = u[kb]
            if ub < 0:
                kb, ub = -1, 0.0
            if ub > ucur + tol:
                any_improving = True
                if rng.random() > p_inertia:
                    new[i] = kb
        a = new
        trace.append(game.potential(a))
        if not any_improving:
            break
    return a, t + 1, trace


def log_linear(game: UserGame, rng, T=0.5, steps=20000, a0=None):
    """Log-linear learning (Lecture 9): a random user revises, choosing action x
    w.p. proportional to exp(u(x)/T) (null action included). Stationary dist ~ exp(Phi/T)."""
    N = game.N
    a = np.full(N, -1) if a0 is None else np.array(a0)
    n = game.loads(a)
    trace = np.empty(steps + 1)
    trace[0] = game.potential(a, n)
    phi = trace[0]
    for s in range(steps):
        i = rng.integers(N)
        k = a[i]
        _remove(game, n, i, k)
        u = np.append(game.action_utils(i, n), 0.0)        # last = null
        ucur = 0.0 if k < 0 else u[k]
        p = np.exp((u - u.max()) / T); p /= p.sum()
        kn = int(rng.choice(len(u), p=p))
        kn = -1 if kn == len(u) - 1 else kn
        a[i] = kn
        _add(game, n, i, kn)
        phi += (0.0 if kn < 0 else u[kn]) - ucur            # exact potential update
        trace[s + 1] = phi
    return a, trace


def enumerate_profiles(game: UserGame):
    """All pure profiles of a tiny game (index -1 = null)."""
    choices = [list(range(-1, len(game.acts[i][0]))) for i in range(game.N)]
    return itertools.product(*choices)
