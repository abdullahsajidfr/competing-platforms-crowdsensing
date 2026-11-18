"""Turn raw T-Drive (taxis) and GeoLife (pedestrians) traces into one replayable week
on a 500 m Beijing grid.

Output: data/beijing_week.pkl with
  grid: rows, cols, lat0, lon0, dlat, dlon
  users: list of dicts {kind: 'taxi'|'ped', src_id, routes: list of 168 int arrays (cell ids)}
  taxi_visits, ped_visits: (G,) number of user-hours that pass through each cell

Slots: slot = weekday*24 + hour (Beijing local time, Monday = 0).
T-Drive timestamps are Beijing time (week of 2-8 Feb 2008).
GeoLife timestamps are GMT (+8 h to local); recordings from 2007-2012 are overlaid by
weekday and hour, not by calendar date.

Usage: .venv/bin/python -m data.prepare [--taxis 200] [--peds 100]
"""
from __future__ import annotations

import argparse, glob, io, os, pickle, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# ~17.8 km x 18.7 km box around central Beijing (2nd-4th ring)
LAT0, LAT1 = 39.83, 39.99
LON0, LON1 = 116.27, 116.49
DLAT = 0.5 / 111.0                                    # 500 m in latitude
DLON = 0.5 / (111.32 * np.cos(np.radians(39.91)))     # 500 m in longitude at Beijing
ROWS = int(np.ceil((LAT1 - LAT0) / DLAT))
COLS = int(np.ceil((LON1 - LON0) / DLON))
G = ROWS * COLS


def to_cell(lat, lon):
    r = ((lat - LAT0) / DLAT).astype(int)
    c = ((lon - LON0) / DLON).astype(int)
    ok = (lat >= LAT0) & (lat < LAT1) & (lon >= LON0) & (lon < LON1)
    return np.where(ok, r * COLS + c, -1)


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def densify(lat, lon, t, max_gap_s=600, step_km=0.1):
    """Linear interpolation between consecutive fixes (gaps <= 10 min) so that a vehicle
    passing through a cell between two GPS fixes is recorded in that cell."""
    out_lat, out_lon, out_t = [lat[:1]], [lon[:1]], [t[:1]]
    for k in range(1, len(lat)):
        dt = (t[k] - t[k - 1]) / np.timedelta64(1, "s")
        dist = haversine_km(lat[k - 1], lon[k - 1], lat[k], lon[k])
        if 0 < dt <= max_gap_s and dist > step_km:
            n = int(dist / step_km)
            f = np.arange(1, n + 1) / (n + 1)
            out_lat.append(lat[k - 1] + f * (lat[k] - lat[k - 1]))
            out_lon.append(lon[k - 1] + f * (lon[k] - lon[k - 1]))
            out_t.append(t[k - 1] + (f * dt * 1e3).astype("timedelta64[ms]"))
        out_lat.append(lat[k:k + 1]); out_lon.append(lon[k:k + 1]); out_t.append(t[k:k + 1])
    return np.concatenate(out_lat), np.concatenate(out_lon), np.concatenate(out_t)


def slot_of(ts):
    ts = pd.DatetimeIndex(ts)
    return (ts.weekday * 24 + ts.hour).values


# ------------------------------------------------------------------------------ taxis
def load_taxis(n_taxis, min_hours=40):
    zips = [z for z in glob.glob(os.path.join(RAW, "tdrive*.zip"))]
    full = [z for z in zips if "full" in z]
    zips = full if full else zips
    files = []
    for z in zips:
        zf = zipfile.ZipFile(z)
        inner = [n for n in zf.namelist() if n.endswith(".zip")]
        if inner:                                   # the official release nests zips
            for n in inner:
                sub = zipfile.ZipFile(io.BytesIO(zf.read(n)))
                files += [(sub, m) for m in sub.namelist() if m.endswith(".txt")]
        else:
            files += [(zf, m) for m in zf.namelist() if m.endswith(".txt")]
    print(f"T-Drive files found: {len(files)}")
    taxis = []
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    for k in order:
        zf, name = files[k]
        raw = zf.read(name)
        if len(raw) < 20_000:
            continue
        df = pd.read_csv(io.BytesIO(raw), header=None, names=["id", "t", "lon", "lat"])
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna().sort_values("t")
        lat, lon, t = df.lat.values, df.lon.values, df.t.values
        # drop GPS jumps (> 120 km/h)
        if len(lat) > 1:
            sp = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.maximum((np.diff(t) / np.timedelta64(1, "h")), 1e-6)
            keep = np.concatenate([[True], sp < 120])
            lat, lon, t = lat[keep], lon[keep], t[keep]
        lat, lon, t = densify(lat, lon, t)
        cells = to_cell(lat, lon); s = slot_of(t)
        ok = cells >= 0
        routes = [np.unique(cells[ok & (s == h)]) for h in range(168)]
        active = sum(len(r) > 0 for r in routes)
        if active >= min_hours:
            taxis.append(dict(kind="taxi", src_id=name, routes=routes))
        if len(taxis) >= n_taxis:
            break
    return taxis


# ------------------------------------------------------------------------------ pedestrians
def load_peds(n_peds, min_hours=8, walk_kmh=6.0):
    base = glob.glob(os.path.join(RAW, "geolife", "*", "Data"))[0]
    peds = []
    for udir in sorted(glob.glob(os.path.join(base, "*"))):
        frames = []
        for f in glob.glob(os.path.join(udir, "Trajectory", "*.plt")):
            d = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 5, 6], names=["lat", "lon", "date", "time"])
            frames.append(d)
        if not frames:
            continue
        d = pd.concat(frames)
        d = d[(d.lat >= LAT0) & (d.lat < LAT1) & (d.lon >= LON0) & (d.lon < LON1)]
        if len(d) < 100:
            continue
        d["t"] = pd.to_datetime(d.date + " " + d.time) + pd.Timedelta(hours=8)    # GMT -> Beijing
        d = d.sort_values("t")
        d["dayhour"] = d.t.dt.floor("h")
        routes = [np.array([], int) for _ in range(168)]
        taken = set()
        for dh, seg in d.groupby("dayhour"):
            if len(seg) < 20:
                continue
            lat, lon = seg.lat.values, seg.lon.values
            dt_h = np.diff(seg.t.values) / np.timedelta64(1, "h")
            sp = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.maximum(dt_h, 1e-6)
            if np.median(sp) > walk_kmh:            # not walking (bus, car, bike, ...)
                continue
            s = int(dh.weekday() * 24 + dh.hour)
            if s in taken:                          # keep the first recorded walk per weekly slot
                continue
            cells = np.unique(to_cell(lat, lon)); cells = cells[cells >= 0]
            if len(cells):
                routes[s] = cells; taken.add(s)
        if len(taken) >= min_hours:
            peds.append(dict(kind="ped", src_id=os.path.basename(udir), routes=routes))
    peds.sort(key=lambda u: -sum(len(r) > 0 for r in u["routes"]))
    return peds[:n_peds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxis", type=int, default=200)
    ap.add_argument("--peds", type=int, default=100)
    a = ap.parse_args()
    peds = load_peds(a.peds); print(f"pedestrians kept: {len(peds)}")
    taxis = load_taxis(a.taxis); print(f"taxis kept: {len(taxis)}")
    users = taxis + peds
    tv = np.zeros(G); pv = np.zeros(G)
    for u in users:
        for r in u["routes"]:
            (tv if u["kind"] == "taxi" else pv)[r] += 1
    out = dict(grid=dict(rows=ROWS, cols=COLS, lat0=LAT0, lon0=LON0, dlat=DLAT, dlon=DLON),
               users=users, taxi_visits=tv, ped_visits=pv)
    pickle.dump(out, open(os.path.join(HERE, "beijing_week.pkl"), "wb"))
    act = np.array([[len(r) > 0 for r in u["routes"]] for u in users])
    print(f"grid {ROWS}x{COLS} = {G} cells; users {len(users)}; active users per hour: "
          f"mean {act.sum(0).mean():.0f}, min {act.sum(0).min()}, max {act.sum(0).max()}; "
          f"cells ever visited: {(tv + pv > 0).sum()}")


if __name__ == "__main__":
    main()
