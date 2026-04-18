"""NREL ResStock + ComStock load profile fetcher (Phoenix climate zone 2B).

Pulls aggregated hourly electricity-demand profiles from the public OEDI
End-Use Load Profiles datasets:

  * ResStock (residential): single-family detached, multi-family 5+ units,
    mobile home, single-family attached  — climate zone 2B (hot-dry, Phoenix).
    Path: nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/
          2024/resstock_amy2018_release_2/timeseries_aggregates/
          by_ashrae_iecc_climate_zone_2004/upgrade=0/
          ashrae_iecc_climate_zone_2004=2B/up00-2b-<type>.csv

  * ComStock (commercial): small office, medium office, retail strip mall,
    warehouse, retail standalone — climate zone 2B (uses 2006 IECC zoning).
    Path: nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/
          2023/comstock_amy2018_release_2/timeseries_aggregates/
          by_ashrae_iecc_climate_zone_2006/upgrade=0/
          ashrae_iecc_climate_zone_2006=2B/up00-2b-<type>.csv

Each raw CSV is ~40 MB with 35040 quarter-hour rows × many end-use columns.
We stream it, sum every `out.electricity.*` column to get total electricity,
aggregate to hourly, normalize per-unit (peak = 1.0), and cache as a tiny
(~10 KB) parquet under data/resstock_cache/.

Why ResStock instead of SMART-DS Austin: ResStock is a per-county, per-
climate-zone simulation of every building in the US housing stock, so the
load shapes encode Phoenix-specific HVAC sensitivity, occupancy patterns,
and Maricopa-county appliance-mix assumptions. The previous SMART-DS
profiles encoded Austin equivalents — close climatologically, but not
Phoenix.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "resstock_cache"


# --- Catalog --------------------------------------------------------------- #

# Residential building types from ResStock 2024 r2, climate zone 2B.
RES_BUILDING_TYPES: List[str] = [
    "single-family_detached",
    "multi-family_with_5plus_units",
    "mobile_home",
    "single-family_attached",
]

# Commercial building types from ComStock 2023 r2, climate zone 2B.
# These five cover the bulk of distribution-feeder commercial load.
COM_BUILDING_TYPES: List[str] = [
    "smalloffice",
    "mediumoffice",
    "retailstripmall",
    "retailstandalone",
    "warehouse",
]


def _res_url(building_type: str) -> str:
    return (
        "https://oedi-data-lake.s3.amazonaws.com/"
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2024/resstock_amy2018_release_2/timeseries_aggregates/"
        "by_ashrae_iecc_climate_zone_2004/upgrade=0/"
        "ashrae_iecc_climate_zone_2004=2B/"
        f"up00-2b-{building_type}.csv"
    )


def _com_url(building_type: str) -> str:
    return (
        "https://oedi-data-lake.s3.amazonaws.com/"
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2023/comstock_amy2018_release_2/timeseries_aggregates/"
        "by_ashrae_iecc_climate_zone_2006/upgrade=0/"
        "ashrae_iecc_climate_zone_2006=2B/"
        f"up00-2b-{building_type}.csv"
    )


@dataclass(frozen=True)
class BuildingProfile:
    sector: str          # "res" or "com"
    building_type: str
    hourly_pu: np.ndarray   # length 8760, normalized so peak = 1.0


# --- Parser --------------------------------------------------------------- #

def _parse_aggregate_csv(raw: bytes) -> pd.DataFrame:
    """Parse a ResStock or ComStock aggregate CSV into a tidy hourly dataframe.

    Returns columns: timestamp, kw, hour_of_year. The kw column is per-unit
    (per-building, for ResStock) or per-1000-sqft (for ComStock).
    """
    text = raw.decode("utf-8", errors="replace")
    head = pd.read_csv(io.StringIO(text), nrows=1)
    cols = list(head.columns)

    # Total electricity: prefer the pre-computed `out.electricity.total.*` if present
    # (ComStock), otherwise sum every `out.electricity.*.energy_consumption.kwh` (ResStock).
    total_col = next((c for c in cols if c.startswith("out.electricity.total")), None)
    if total_col:
        elec_cols = [total_col]
    else:
        elec_cols = [c for c in cols if c.startswith("out.electricity.") and c.endswith(".kwh")]
    if not elec_cols:
        raise ValueError("no electricity column found in CSV")

    # Normalization column: units_represented (ResStock) or floor_area_represented (ComStock)
    norm_col = "units_represented" if "units_represented" in cols else "floor_area_represented"
    if norm_col not in cols:
        raise ValueError(f"no normalization column found (looked for units_represented / floor_area_represented)")

    usecols = ["timestamp", norm_col, *elec_cols]
    df = pd.read_csv(io.StringIO(text), usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["total_kwh"] = df[elec_cols].sum(axis=1) if len(elec_cols) > 1 else df[elec_cols[0]]
    norm_val = df[norm_col].iloc[0]
    if norm_col == "floor_area_represented":
        # ComStock: divide by floor area (sqft) and rescale to per-1000-sqft so
        # the per-unit shape has comparable amplitude to ResStock per-home.
        df["per_unit_kwh"] = df["total_kwh"] / max(norm_val, 1.0) * 1000.0
    else:
        df["per_unit_kwh"] = df["total_kwh"] / max(norm_val, 1.0)

    # Aggregate quarter-hour energy values to hourly: sum of 4 quarter-hour
    # kWh = total kWh in that hour = average kW (since each is 1-hour wide).
    df = df.set_index("timestamp")
    hourly = df["per_unit_kwh"].resample("h").sum()
    out = hourly.reset_index()
    out.columns = ["timestamp", "kw"]
    out["hour_of_year"] = (out["timestamp"].dt.dayofyear - 1) * 24 + out["timestamp"].dt.hour
    out = out.iloc[:8760].reset_index(drop=True)
    return out


def _fetch_one(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aps-feeder-intelligence/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch_profile(sector: str, building_type: str, cache: bool = True) -> BuildingProfile:
    """Fetch one building-type aggregate, normalize to per-unit, return BuildingProfile."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{sector}_{building_type}.parquet"
    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return BuildingProfile(sector, building_type, df["pu"].to_numpy().astype(np.float32))

    url = _res_url(building_type) if sector == "res" else _com_url(building_type)
    print(f"  fetching {sector}/{building_type} ...")
    raw = _fetch_one(url)
    parsed = _parse_aggregate_csv(raw)
    arr = parsed["kw"].to_numpy().astype(np.float64)
    if arr.size != 8760:
        # Pad/trim to exactly 8760 hours
        if arr.size > 8760:
            arr = arr[:8760]
        else:
            arr = np.concatenate([arr, np.full(8760 - arr.size, arr.mean())])
    peak = float(arr.max()) if arr.max() > 0 else 1.0
    pu = (arr / peak).astype(np.float32)

    if cache:
        out = pd.DataFrame({"hour_of_year": np.arange(8760), "pu": pu})
        out.to_parquet(cache_file, index=False)
    print(f"    peak={peak:.3f} kW/unit  mean={arr.mean():.3f}  pu_mean={pu.mean():.3f}")
    return BuildingProfile(sector, building_type, pu)


def fetch_all(cache: bool = True) -> List[BuildingProfile]:
    profiles = []
    for bt in RES_BUILDING_TYPES:
        try:
            profiles.append(fetch_profile("res", bt, cache=cache))
        except Exception as e:
            print(f"  skip res/{bt}: {e}")
    for bt in COM_BUILDING_TYPES:
        try:
            profiles.append(fetch_profile("com", bt, cache=cache))
        except Exception as e:
            print(f"  skip com/{bt}: {e}")
    return profiles


# --- Per-bus customer mix using ResStock/ComStock profiles ----------------- #

def _mix_for(nominal_kw: float) -> Tuple[List[str], List[str]]:
    """Return (res_types, com_types) for a bus of the given nominal-kW size.

    Each entry is a building-type name. The synthesizer draws the actual
    weights and counts from these lists.
    """
    if nominal_kw <= 0:
        return (["single-family_detached"], [])
    if nominal_kw < 15:
        # Small residential pocket: a single home
        return (["single-family_detached"], [])
    if nominal_kw < 35:
        # 5-10 homes mixed types
        return (["single-family_detached"] * 4 + ["multi-family_with_5plus_units"] * 2, [])
    if nominal_kw < 80:
        # Mostly residential, hint of small commercial
        return (["single-family_detached"] * 6 + ["multi-family_with_5plus_units"] * 4
                + ["mobile_home"] * 2,
                ["smalloffice"])
    if nominal_kw < 200:
        # Mixed neighborhood
        return (["single-family_detached"] * 10 + ["multi-family_with_5plus_units"] * 6
                + ["single-family_attached"] * 4,
                ["smalloffice", "retailstripmall"])
    # Large pocket — commercial-dominated mall / office park / industrial
    return (["multi-family_with_5plus_units"] * 4 + ["single-family_detached"] * 6,
            ["mediumoffice", "retailstripmall", "retailstandalone", "warehouse"])


def synth_bus_loads_resstock(
    bus_list: List[str],
    nominal_kw: dict[str, float],
    times: pd.DatetimeIndex,
    profiles: List[BuildingProfile],
    seed: int = 7,
) -> np.ndarray:
    """Build a [N_bus, T] hourly kW matrix using ResStock + ComStock profiles."""
    by_key: Dict[str, BuildingProfile] = {f"{p.sector}/{p.building_type}": p for p in profiles}

    def _get(sector: str, bt: str) -> Optional[BuildingProfile]:
        return by_key.get(f"{sector}/{bt}")

    # Hour-of-year mapping (use day-of-year × 24 + hour, year-agnostic so
    # 2024/2025/2026 all index into the same 2018 calendar curves).
    sim_local = times if times.tz is not None else times.tz_localize("America/Phoenix")
    hour_of_year = (sim_local.dayofyear - 1) * 24 + sim_local.hour
    hour_of_year = np.clip(hour_of_year.values, 0, 8759)

    N = len(bus_list)
    T = len(times)
    out = np.zeros((N, T), dtype=np.float32)
    for i, bus in enumerate(bus_list):
        nominal = float(nominal_kw.get(bus, 0.0))
        res_types, com_types = _mix_for(nominal)
        annual = np.zeros(8760, dtype=np.float32)
        # Sum all residential picks
        for bt in res_types:
            p = _get("res", bt)
            if p is not None:
                annual += p.hourly_pu
        # Sum commercial picks
        for bt in com_types:
            p = _get("com", bt)
            if p is not None:
                annual += p.hourly_pu

        if annual.max() <= 0:
            continue
        window = annual[hour_of_year]
        # Scale so the bus's annual peak hits its IEEE nominal kW
        peak = float(window.max())
        scale = max(0.01, nominal / peak) if nominal > 0 else 0.5 / max(peak, 1.0)
        out[i] = (window * scale).astype(np.float32)
    return out


# --- CLI ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true", help="re-fetch even if cached")
    args = p.parse_args()
    profs = fetch_all(cache=not args.no_cache)
    print(f"\nfetched {len(profs)} profiles total")
    for kind in ("res", "com"):
        sub = [p for p in profs if p.sector == kind]
        if not sub:
            continue
        print(f"\n{kind.upper()}:")
        for p in sub:
            shape = p.hourly_pu.reshape(365, 24).mean(axis=0)
            peak_h = int(shape.argmax())
            trough_h = int(shape.argmin())
            print(f"  {p.building_type:<35}  peak hour={peak_h:>2}  trough={trough_h:>2}  "
                  f"peak/trough={shape.max()/max(shape.min(), 1e-3):.2f}x")
