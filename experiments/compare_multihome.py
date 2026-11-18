import numpy as np, json
from crowdsim.env import Config
from crowdsim.multihome import evaluate, price_matrix_mh
from crowdsim.equilibrium import pure_nash, collusive, mixed_nash_2p
cfg = Config(); grid = np.round(np.arange(0.2, 2.01, 0.2), 2)
print("fixed price (1,1):")
for e in (0.0, 0.5, 2.0):
    r = evaluate(cfg, (1.0, 1.0), e=e)
    print(f"  extra-upload cost e={e}: coverage={r['coverage']:.3f} low={r['low']:.3f} sent-to-both={r['both']:.2f} profit={r['profit'].round(1)} user_net={r['user_net']:.2f}")
for e in (0.0,):
    Pi = price_matrix_mh(cfg, grid, K=30, e=e)
    ne = pure_nash(Pi); c = collusive(Pi)
    print(f"e={e} pure NE:", [(grid[i],grid[j],Pi[i,j].round(1)) for i,j in ne], "collusive", grid[c[0]], grid[c[1]], Pi[c].round(1), "joint", Pi[c].sum().round(1))
    br1 = Pi[...,0].argmax(0); br2 = Pi[...,1].argmax(1)
    print("  P1 best reply to each m2:", dict(zip(grid, grid[br1])))
    print("  P2 best reply to each m1:", dict(zip(grid, grid[br2])))
    if not ne:
        x,y = mixed_nash_2p(Pi[...,0],Pi[...,1])[0]; print("  mixed", x.round(2), y.round(2))
    for (i,j) in ne:
        r = evaluate(cfg, (grid[i],grid[j]), e=e); print("  at NE coverage", round(r['coverage'],3), "low", round(r['low'],3), "user_net", round(r['user_net'],2))
