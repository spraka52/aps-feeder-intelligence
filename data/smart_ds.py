"""NREL SMART-DS per-customer load profile fetcher.

Pulls a small, curated sample of per-unit hourly load profiles from the
public SMART-DS dataset (Austin TX, P1R substation, 2018) — the closest hot-
climate analog to Phoenix in the SMART-DS coverage. Each profile is one
real synthetic-but-realistic customer's annual electricity demand from
NREL's grid-modeling-and-analysis simulation.

Why SMART-DS instead of our prior procedural shapes:
  * Real customer-class diversity. Residential profiles peak in the
    evening; commercial profiles peak during business hours; mixed
    customers have flatter profiles. Our prior model treated every bus
    as a phase-shifted residential cluster.
  * Real-life HVAC, occupancy, and weekend patterns baked in.
  * Public (no auth required) — direct HTTPS GETs from
    s3://oedi-data-lake/SMART-DS/v1.0/2018/AUS/P1R/profiles/.

The downloaded full-year, 15-minute resolution CSVs are aggregated to
hourly resolution and sliced to the requested summer window (typically
Jun–Aug 2018). All profiles are normalized per-unit (peak ≤ 1.0).
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "smart_ds_cache"

# Curated sample of SMART-DS Austin P1R customer IDs.
# Roughly diverse mix to give the IEEE 34 buses different per-bus shapes.
RESIDENTIAL_IDS: List[int] = [
    2096, 3908, 5828, 16121, 17473, 23868, 33323, 35415,
    38274, 40187, 40688, 41702, 42956, 43287, 44719,
]

COMMERCIAL_IDS: List[int] = [
    # A subset of commercial customers; will be filled at first fetch
    # if these IDs aren't available we'll list and pick.
]

BASE_URL = (
    "https://oedi-data-lake.s3.amazonaws.com/"
    "SMART-DS/v1.0/2018/AUS/P1R/profiles/"
)


@dataclass(frozen=True)
class Profile:
    customer_class: str        # "res" or "com"
    customer_id: int
    hourly_pu: np.ndarray      # length 8760, per-unit (peak ≤ 1.0)


def _profile_url(customer_class: str, customer_id: int) -> str:
    return f"{BASE_URL}{customer_class}_kw_{customer_id}_pu.csv"


def _list_commercial_ids(n: int = 8) -> List[int]:
    """List a sample of commercial customer IDs from the S3 bucket."""
    import re
    keys: List[str] = []
    marker = ""
    for _ in range(20):
        url = f"https://oedi-data-lake.s3.amazonaws.com/?prefix=SMART-DS/v1.0/2018/AUS/P1R/profiles/com_kw&marker={marker}"
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
        page = re.findall(r'<Key>([^<]+)</Key>', text)
        if not page:
            break
        keys.extend(page)
        marker = page[-1]
        if '<IsTruncated>true</IsTruncated>' not in text:
            break
        if len(keys) >= 200:
            break
    ids = []
    for k in keys:
        fname = k.rsplit("/", 1)[-1]
        m = re.match(r"com_kw_(\d+)_pu\.csv", fname)
        if m:
            ids.append(int(m.group(1)))
    # Deterministic sample
    return sorted(ids)[:n]


def _fetch_one_pu_year(customer_class: str, customer_id: int) -> np.ndarray:
    """Download a single annual 15-min per-unit profile and aggregate to hourly."""
    url = _profile_url(customer_class, customer_id)
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    text = raw.decode()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    arr = np.array([float(x) for x in lines], dtype=np.float32)
    if arr.size != 35040:
        # Some profiles may have slightly different lengths; pad/trim
        target = 35040
        if arr.size > target:
            arr = arr[:target]
        else:
            arr = np.pad(arr, (0, target - arr.size), mode="edge")
    # 15-min → 1-hour mean
    hourly = arr.reshape(8760, 4).mean(axis=1).astype(np.float32)
    return hourly


def fetch_profiles(
    n_residential: int = 8,
    n_commercial: int = 5,
    cache: bool = True,
) -> List[Profile]:
    """Fetch and cache a sample set of residential + commercial profiles."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"profiles_r{n_residential}_c{n_commercial}.parquet"
    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return [
            Profile(
                customer_class=row["customer_class"],
                customer_id=int(row["customer_id"]),
                hourly_pu=np.array(row["hourly_pu"], dtype=np.float32),
            )
            for _, row in df.iterrows()
        ]

    profiles: List[Profile] = []
    for cid in RESIDENTIAL_IDS[:n_residential]:
        try:
            arr = _fetch_one_pu_year("res", cid)
            profiles.append(Profile("res", cid, arr))
            print(f"  fetched res_kw_{cid}: peak={arr.max():.3f}  mean={arr.mean():.3f}")
        except Exception as e:
            print(f"  skip res_kw_{cid}: {e}")

    com_ids = COMMERCIAL_IDS or _list_commercial_ids(n_commercial * 2)
    for cid in com_ids[:n_commercial]:
        try:
            arr = _fetch_one_pu_year("com", cid)
            profiles.append(Profile("com", cid, arr))
            print(f"  fetched com_kw_{cid}: peak={arr.max():.3f}  mean={arr.mean():.3f}")
        except Exception as e:
            print(f"  skip com_kw_{cid}: {e}")

    if cache and profiles:
        df = pd.DataFrame([
            {"customer_class": p.customer_class,
             "customer_id": p.customer_id,
             "hourly_pu": p.hourly_pu.tolist()}
            for p in profiles
        ])
        df.to_parquet(cache_file, index=False)
    return profiles


# ----------------- Per-bus customer mix for the IEEE 34 case ---------------- #

# (n_residential, n_commercial) per bus — sized so that the customer mix sums
# roughly to the IEEE 34 nominal kW with res ≈ 4 kW peak each, com ≈ 60 kW peak each.
# Every bus is then deterministically scaled to *exactly* match its IEEE nominal,
# so the case parameters are preserved.
def _mix_for(nominal_kw: float) -> Tuple[int, int]:
    if nominal_kw <= 0:
        return (1, 0)
    if nominal_kw < 15:
        return (3, 0)
    if nominal_kw < 35:
        return (8, 0)
    if nominal_kw < 80:
        return (12, 1)
    if nominal_kw < 200:
        return (20, 2)
    # Large pockets: mixed commercial-dominated
    return (max(8, int(nominal_kw / 25)), max(2, int(nominal_kw / 90)))


def synth_bus_loads_smart_ds(
    bus_list: List[str],
    nominal_kw: dict[str, float],
    times: pd.DatetimeIndex,
    profiles: List[Profile],
    seed: int = 7,
) -> np.ndarray:
    """Build a [N_bus, T] hourly kW matrix using SMART-DS profiles per bus.

    For each bus we deterministically draw N_res residential + N_com commercial
    profiles, sum them across the requested time window, then scale so that
    the bus's annual peak matches its IEEE nominal kW. This gives realistic
    customer-mix-aware shapes while preserving the IEEE 34 case parameters.
    """
    rng = np.random.default_rng(seed)
    res = [p for p in profiles if p.customer_class == "res"]
    com = [p for p in profiles if p.customer_class == "com"]
    if not res or not com:
        raise RuntimeError("Need at least 1 residential and 1 commercial profile")

    # Convert times to hour-of-year offsets (0..8759) in Austin LOCAL time.
    # SMART-DS profiles are 8760 hourly samples for calendar 2018 AUS local.
    # The simulation index (Phoenix local) happens to share the same
    # calendar — we just take the same hour-of-year for both.
    aus_2018_start = pd.Timestamp("2018-01-01", tz="America/Chicago")
    sim_local = times.tz_convert("America/Chicago") if times.tz is not None else \
        times.tz_localize("America/Phoenix").tz_convert("America/Chicago")
    # Map by month/day/hour (year-agnostic) so 2024 simulations can still index 2018 profiles.
    hour_of_year = (
        (sim_local.month - 1) * 31 * 24
        + (sim_local.day - 1) * 24
        + sim_local.hour
    )
    # More accurate: cumulative day-of-year * 24 + hour
    hour_of_year = (sim_local.dayofyear - 1) * 24 + sim_local.hour
    hour_of_year = np.clip(hour_of_year.values, 0, 8759)

    N = len(bus_list)
    T = len(times)
    out = np.zeros((N, T), dtype=np.float32)
    for i, bus in enumerate(bus_list):
        nominal = float(nominal_kw.get(bus, 0.0))
        n_res, n_com = _mix_for(nominal)
        # deterministic per-bus picks via local rng
        bus_rng = np.random.default_rng(int(bus) if bus.isdigit() else 0)
        res_picks = bus_rng.choice(len(res), size=n_res, replace=True)
        com_picks = bus_rng.choice(len(com), size=n_com, replace=True) if com and n_com else np.array([], dtype=int)
        # Sum the chosen customers' annual hourly profiles, then index by hour-of-year
        annual = np.zeros(8760, dtype=np.float32)
        for j in res_picks:
            annual += res[j].hourly_pu
        for j in com_picks:
            annual += com[j].hourly_pu
        # Slice the simulation window and scale to IEEE nominal
        window = annual[hour_of_year]
        peak = float(window.max()) if window.size else 1.0
        if peak <= 0:
            peak = 1.0
        scale = max(0.01, nominal / peak) if nominal > 0 else 0.5 / max(peak, 1.0)
        out[i] = (window * scale).astype(np.float32)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_res", type=int, default=8)
    p.add_argument("--n_com", type=int, default=5)
    args = p.parse_args()
    profs = fetch_profiles(args.n_res, args.n_com)
    print(f"\nfetched {len(profs)} profiles total")
    res = [p for p in profs if p.customer_class == "res"]
    com = [p for p in profs if p.customer_class == "com"]
    print(f"residential: {len(res)}  mean-peak={np.mean([p.hourly_pu.max() for p in res]):.3f}")
    print(f"commercial : {len(com)}  mean-peak={np.mean([p.hourly_pu.max() for p in com]):.3f}")
    # Show 24-hour shape for one residential and one commercial
    if res:
        r0 = res[0].hourly_pu
        # Average across each hour-of-day across the year
        shape_r = r0.reshape(365, 24).mean(axis=0)
        print(f"residential hour-of-day shape (sample): peak hour={int(shape_r.argmax())}, "
              f"trough hour={int(shape_r.argmin())}, peak/trough={shape_r.max()/max(shape_r.min(), 1e-3):.2f}x")
    if com:
        c0 = com[0].hourly_pu
        shape_c = c0.reshape(365, 24).mean(axis=0)
        print(f"commercial  hour-of-day shape (sample): peak hour={int(shape_c.argmax())}, "
              f"trough hour={int(shape_c.argmin())}, peak/trough={shape_c.max()/max(shape_c.min(), 1e-3):.2f}x")
