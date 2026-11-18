import time, numpy as np
from crowdsim.env import Config, World, UserGame
from crowdsim.dynamics import async_best_response, br_with_inertia, log_linear

cfg = Config()
w = World(cfg)
rng = np.random.default_rng(1)
print("road cells:", w.is_road.sum(), "low-traffic cells:", w.low_traffic.sum(), "zero-traffic:", (w.traffic==0).sum())
acts = w.build_actions(w.draw_routes(rng))
print("actions per user (mean):", np.mean([len(a[0]) for a in acts]))
R = np.ones(cfg.M)[:,None] * w.W * cfg.b
g = UserGame(R, acts)
# 1) potential identity checked on every move inside BR (assert)
t0=time.time()
a, sweeps, tr = async_best_response(g, rng, record=True, check_potential=True)
print(f"async BR: sweeps={sweeps}, NE={g.is_nash(a)}, Phi monotone={np.all(np.diff(tr)>=-1e-9)}, {time.time()-t0:.2f}s")
# 2) random unilateral deviations: dPhi == dU
errs=[]
for _ in range(2000):
    i = rng.integers(g.N); b = a.copy(); b[i] = rng.integers(-1, len(acts[i][0]))
    errs.append(abs((g.potential(b)-g.potential(a)) - (g.utility(i,b)-g.utility(i,a))))
print("max |dPhi - dU| over 2000 random deviations:", max(errs))
# 3) BR with inertia
a2, steps, tr2 = br_with_inertia(g, rng, p_inertia=0.7)
print("BR inertia: steps", steps, "NE", g.is_nash(a2))
a3, tr3 = log_linear(g, rng, T=0.1, steps=20000)
print("LLL T=0.1 final Phi", tr3[-1], "vs BR NE Phi", tr[-1], "recomputed", g.potential(a3))
t0=time.time()
for k in range(20):
    acts = w.build_actions(w.draw_routes(rng)); gg=UserGame(R,acts); async_best_response(gg, rng)
print(f"20 rounds: {time.time()-t0:.2f}s")
