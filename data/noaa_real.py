"""Real Phoenix Sky Harbor (KPHX) hourly weather from NOAA NCEI ISD-Lite.

NCEI ISD-Lite is the public, no-auth fixed-width hourly dataset for ~5,000
weather stations worldwide. Each year's file is a gzipped text grid; we
parse it once and cache the result as a Parquet next to the synthetic data.

Schema (12 fixed-width columns, all integers, missing = -9999):
   1  YYYY
   2  MM
   3  DD
   4  HH (UTC)
   5  Air temp (°C × 10)
   6  Dew point (°C × 10)
   7  Sea-level pressure (hPa × 10)
   8  Wind direction (°)
   9  Wind speed (m/s × 10)
  10  Sky condition (cloud cover) coded 0–9 + special
  11  Precip 1-hr (mm × 10)
  12  Precip 6-hr (mm × 10)

Cloud-cover code (col 10) interpretation we use for the irradiance model:
  0 = clear, 1 = few, 2 = scattered, 3 = broken, 4 = overcast,
  5 = obscured, 6 = partial obscured, 7-9 / -9999 = missing → assume clear.

KPHX station: USAF 722780, WBAN 23183, lat 33.434, lon -112.012, elev 337 m.
"""
from __future__ import annotations

import gzip
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "noaa_cache"


KPHX_USAF = "722780"
KPHX_WBAN = "23183"
KPHX_TZ = "America/Phoenix"  # AZ does not observe DST


@dataclass(frozen=True)
class Station:
    name: str
    usaf: str
    wban: str
    lat: float
    lon: float
    elev_m: int
    tz: str = "UTC"


KPHX = Station("Phoenix Sky Harbor", KPHX_USAF, KPHX_WBAN, 33.434, -112.012, 337, KPHX_TZ)


def _isd_url(usaf: str, wban: str, year: int) -> str:
    return f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"


# Translate ISD-Lite cloud-cover codes to a [0, 1] cloud fraction. Higher = cloudier.
_CLOUD_FRAC = {
    0: 0.05,   # clear
    1: 0.15,   # few
    2: 0.30,   # scattered
    3: 0.65,   # broken
    4: 0.95,   # overcast
    5: 1.00,   # obscured (treat as overcast)
    6: 0.50,   # partial obscured
    7: 0.50,   # any-other "missing-with-precip" type
    8: 0.50,
    9: 0.30,
}


def _parse_isd_lite(text: str, year: int) -> pd.DataFrame:
    """Parse the fixed-width body of an ISD-Lite file into a typed DataFrame."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            yyyy, mm, dd, hh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            temp_t10 = int(parts[4])
            dew_t10 = int(parts[5])
            slp_t10 = int(parts[6]) if len(parts) > 6 else -9999
            wdir = int(parts[7]) if len(parts) > 7 else -9999
            wspd_t10 = int(parts[8]) if len(parts) > 8 else -9999
            sky = int(parts[9]) if len(parts) > 9 else -9999
        except ValueError:
            continue
        rows.append((yyyy, mm, dd, hh, temp_t10, dew_t10, slp_t10, wdir, wspd_t10, sky))

    df = pd.DataFrame(rows, columns=["yyyy", "mm", "dd", "hh", "temp_t10", "dew_t10",
                                     "slp_t10", "wdir", "wspd_t10", "sky"])
    df["time_utc"] = pd.to_datetime(df[["yyyy", "mm", "dd", "hh"]].rename(columns={
        "yyyy": "year", "mm": "month", "dd": "day", "hh": "hour"}), utc=True)
    # Convert tenths
    df["temp_c"] = np.where(df["temp_t10"] == -9999, np.nan, df["temp_t10"] / 10.0)
    df["dew_c"] = np.where(df["dew_t10"] == -9999, np.nan, df["dew_t10"] / 10.0)
    df["wspd_mps"] = np.where(df["wspd_t10"] == -9999, np.nan, df["wspd_t10"] / 10.0)
    df["cloud_frac"] = df["sky"].map(_CLOUD_FRAC).astype(float)
    df["cloud_frac"] = df["cloud_frac"].fillna(0.30)  # mild default if missing
    return df[["time_utc", "temp_c", "dew_c", "wspd_mps", "cloud_frac"]].dropna(subset=["temp_c"])


def fetch_station_year(station: Station, year: int, cache: bool = True) -> pd.DataFrame:
    """Download (and cache) one year of ISD-Lite data for a station."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{station.usaf}-{station.wban}-{year}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    url = _isd_url(station.usaf, station.wban, year)
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    text = gzip.decompress(raw).decode("utf-8")
    df = _parse_isd_lite(text, year)
    if cache:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_hourly(
    station: Station,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache: bool = True,
) -> pd.DataFrame:
    """Return a tidy hourly frame on the station's local timezone, with no gaps.

    Columns: time (tz-aware, station tz), temp_c, cloud_frac, dew_c, wspd_mps.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    years = sorted({y for y in range(start.year, end.year + 1)})

    frames = []
    for y in years:
        try:
            frames.append(fetch_station_year(station, y, cache=cache))
        except Exception as e:
            print(f"[noaa_real] year {y} for {station.usaf}-{station.wban} failed: {e}")

    if not frames:
        raise RuntimeError(f"no NOAA data fetched for {station.name}")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time_utc")
    df = df.sort_values("time_utc")
    # Reindex to a complete hourly UTC range, then convert to local
    full = pd.DataFrame({"time_utc": pd.date_range(df["time_utc"].min(), df["time_utc"].max(), freq="h", tz="UTC")})
    df = full.merge(df, on="time_utc", how="left")
    # Forward+back fill small gaps
    for col in ["temp_c", "cloud_frac", "dew_c", "wspd_mps"]:
        df[col] = df[col].interpolate(limit=4).ffill().bfill()
    df["time"] = df["time_utc"].dt.tz_convert(station.tz)
    df = df.set_index("time").sort_index()

    # Slice the requested local-time range
    if start.tzinfo is None:
        start = start.tz_localize(station.tz)
    else:
        start = start.tz_convert(station.tz)
    if end.tzinfo is None:
        end = end.tz_localize(station.tz)
    else:
        end = end.tz_convert(station.tz)
    return df.loc[start:end].reset_index()


def get_phoenix(start: str, end: str) -> pd.DataFrame:
    """Convenience: hourly Phoenix Sky Harbor weather for [start, end] local time."""
    return fetch_hourly(KPHX, start, end)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--end", default="2024-08-31 23:00")
    args = p.parse_args()
    df = get_phoenix(args.start, args.end)
    print(f"rows: {len(df)}  range: {df['time'].min()} .. {df['time'].max()}")
    print(f"temp_c: min={df['temp_c'].min():.1f}  max={df['temp_c'].max():.1f}  mean={df['temp_c'].mean():.1f}")
    print(f"cloud_frac mean={df['cloud_frac'].mean():.2f}")
    # Identify heatwave runs (3+ consecutive days with daily max >= 41°C)
    daily_max = df.set_index("time")["temp_c"].resample("D").max()
    hot_days = daily_max[daily_max >= 41].index
    print(f"days with high ≥ 41°C: {len(hot_days)}")
    # Detect runs
    if len(hot_days):
        runs = []
        cur_start = hot_days[0]; prev = hot_days[0]
        for d in hot_days[1:]:
            if (d - prev).days == 1:
                prev = d
            else:
                if (prev - cur_start).days >= 2:
                    runs.append((cur_start.date(), prev.date()))
                cur_start = d; prev = d
        if (prev - cur_start).days >= 2:
            runs.append((cur_start.date(), prev.date()))
        print(f"heatwave runs (≥3 consecutive ≥41°C days):")
        for r in runs:
            print(f"  {r[0]}  →  {r[1]}")
