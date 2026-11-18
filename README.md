# Competing Platforms in Mobile Crowdsensing

**An agent-based simulation of learning platforms and strategic users**

Abdullah Sajid, Mishal Sagar. LUMS, EE567/CS6315 Multiagent Systems, Deliverable 1.

Several crowdsensing apps (air quality, traffic, noise, weather) post paid sensing tasks on 500 m cells of central Beijing. Each hour, 300 real users choose one task:
- 200 taxis from T-Drive;
- 100 walkers from GeoLife.

We study four questions:
- whether users who exaggerate rival offers gain, and what it costs the apps (RQ1);
- whether learning apps raise or lower pay (RQ2);
- whether low-traffic areas are left uncovered and whether a bonus helps (RQ3);
- which pricing method gives the most coverage per unit budget (RQ4).

## Open in Colab
| Notebook | |
|-|-|
| **Full D1 simulation** (real traces, Mesa, 2–4 apps) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdullahsajidfr/competing-platforms-crowdsensing/blob/main/simulation_full.ipynb) |
| Synthetic pilot (10×10 grid) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdullahsajidfr/competing-platforms-crowdsensing/blob/main/simulation.ipynb) |

In Colab, run the "Colab setup" cell first. It clones this repo and installs Mesa. Cached results in `results/full/` make the full notebook load in about a minute. Delete that folder to recompute everything (about 40 minutes on Colab, 11 minutes on 8 local cores).

## What's here
| Path | Contents |
|-|-|
| `simulation_full.ipynb` | Executed D1 notebook: data, equilibrium check, and Experiments 1–5 (baseline, Q-learning, lying users, coverage bonus, method comparison) |
| `formulation.md` | The game: players, actions, utilities, Propositions 1–2, learning rules, metrics |
| `assumptions.md` | Every modelling choice, each with a reason |
| `report.md` | Results per research question, limitations, next steps |
| `crowdsim/mesa_model.py` | Mesa model: `UserAgent`, `PlatformAgent`, `CrowdsensingModel` |
| `crowdsim/full_experiments.py` | Parallel experiment drivers with caching |
| `crowdsim/env.py`, `dynamics.py`, `equilibrium.py`, `learning.py` | Pilot model, best-response and log-linear dynamics, Nash and collusive benchmarks |
| `data/prepare.py` | Builds `data/beijing_week.pkl` from the raw T-Drive and GeoLife files |
| `results/` | Cached runs and figures |
| `simulation.ipynb`, `report_pilot.md` | Earlier synthetic pilot |

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook simulation_full.ipynb
```

## Data
`data/beijing_week.pkl` already holds the processed week, so the raw data is only needed to rebuild it. To rebuild:
1. Download the [T-Drive sample](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) to `data/raw/tdrive_full.zip`.
2. Download [GeoLife 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367) and unzip it into `data/raw/geolife/`.
3. Run `python -m data.prepare --taxis 200 --peds 100`.

Please cite:
- J. Yuan et al., "T-Drive: Driving directions based on taxi trajectories," ACM SIGSPATIAL GIS, 2010.
- Y. Zheng, X. Xie, W.-Y. Ma, "GeoLife: A collaborative social networking service among user, location and trajectory," IEEE Data Eng. Bull., 2010.
