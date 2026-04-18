"""NREL NSRDB hourly irradiance fetcher (GOES Aggregated PSM v4.0.0).

Fetches GHI / DNI / DHI / cloud type for a point (lat/lon) and a year, caches
the result as a Parquet so Streamlit Cloud deploys don't need to re-fetch
(or know the API key).

The active NSRDB endpoint is now:
    https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv

The PSM v3 / `psm3-2-2-download` paths from older NSRDB integrations are
deprecated and now return 404.

API key is read from the ``NREL_API_KEY`` environment variable. For
Streamlit Cloud, set it under app secrets (``[secrets]\nNREL_API_KEY = "..."``).
If no key is available we silently fall back to the synthetic clear-sky model.
"""
from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "nsrdb_cache"

NSRDB_ENDPOINT = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-aggregated-v4-0-0-download.csv"
)


def _api_key() -> Optional[str]:
    return os.environ.get("NREL_API_KEY")


@dataclass(frozen=True)
class NSRDBPoint:
    name: str
    lat: float
    lon: float
    tz: str = "America/Phoenix"


PHOENIX_KPHX = NSRDBPoint("Phoenix Sky Harbor", 33.434, -112.012)


def _fetch_year_csv(point: NSRDBPoint, year: int, email: str) -> bytes:
    key = _api_key()
    if not key:
        raise RuntimeError("NREL_API_KEY not set; cannot fetch NSRDB")
    params = dict(
        api_key=key,
        wkt=f"POINT({point.lon} {point.lat})",
        names=str(year),
        interval="60",
        attributes="ghi,dni,dhi,clearsky_ghi,air_temperature,cloud_type,solar_zenith_angle",
        utc="true",
        leap_day="false",
        email=email,
        full_name="APS Hackathon User",
        affiliation="ASU APS AI for Energy Hackathon",
        reason="distribution-feeder spatio-temporal forecasting",
        mailing_list="false",
    )
    url = NSRDB_ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "aps-feeder-intelligence/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _parse_nsrdb_csv(raw: bytes) -> pd.DataFrame:
    """NSRDB CSV has 2 metadata rows then a header row then hourly data."""
    text = raw.decode("utf-8")
    # Skip the first 2 metadata lines (NSRDB header + values), then read
    df = pd.read_csv(io.StringIO(text), skiprows=2)
    df = df.rename(columns={
        "Year": "year", "Month": "month", "Day": "day",
        "Hour": "hour", "Minute": "minute",
        "GHI": "ghi", "DNI": "dni", "DHI": "dhi",
        "Clearsky GHI": "clearsky_ghi",
        "Temperature": "temp_c",
        "Cloud Type": "cloud_type",
        "Solar Zenith Angle": "solar_zenith",
    })
    df["time_utc"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]],
        utc=True,
    )
    keep = ["time_utc", "ghi", "dni", "dhi", "clearsky_ghi", "temp_c", "cloud_type", "solar_zenith"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df


def fetch_year(point: NSRDBPoint, year: int, email: str = "user@example.com",
               cache: bool = True) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = point.name.replace(" ", "_").lower()
    cache_file = CACHE_DIR / f"{safe_name}_{year}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    raw = _fetch_year_csv(point, year, email)
    df = _parse_nsrdb_csv(raw)
    if cache:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_hourly(
    point: NSRDBPoint,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    email: str = "user@example.com",
    cache: bool = True,
) -> pd.DataFrame:
    """Hourly irradiance + temperature for [start, end] in the point's local tz."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    years = sorted({y for y in range(start.year, end.year + 1)})
    frames = []
    for y in years:
        try:
            frames.append(fetch_year(point, y, email=email, cache=cache))
        except Exception as e:
            print(f"[nsrdb_real] year {y} for {point.name} failed: {e}")
    if not frames:
        raise RuntimeError(f"no NSRDB data for {point.name}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time_utc").sort_values("time_utc")
    # NSRDB v4 timestamps the centre of each hour (e.g. 00:30 = the 00:00-01:00
    # average). Floor to the hour so it aligns with hour-start grids.
    df["time_utc"] = df["time_utc"].dt.floor("h")
    df["time"] = df["time_utc"].dt.tz_convert(point.tz)
    df = df.set_index("time").sort_index()
    if start.tzinfo is None:
        start = start.tz_localize(point.tz)
    else:
        start = start.tz_convert(point.tz)
    if end.tzinfo is None:
        end = end.tz_localize(point.tz)
    else:
        end = end.tz_convert(point.tz)
    return df.loc[start:end].reset_index()


def get_phoenix(start: str, end: str, email: str = "user@example.com") -> pd.DataFrame:
    return fetch_hourly(PHOENIX_KPHX, start, end, email=email)


def is_available() -> bool:
    """True if either we have a cached parquet, or an API key is set."""
    if any(CACHE_DIR.glob("*.parquet")):
        return True
    return _api_key() is not None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--end", default="2024-08-31 23:00")
    p.add_argument("--email", default="user@example.com")
    args = p.parse_args()
    df = get_phoenix(args.start, args.end, email=args.email)
    print(f"rows: {len(df)}  range: {df['time'].min()} .. {df['time'].max()}")
    print(f"GHI W/m^2: max={df['ghi'].max():.0f}  daytime mean={df.loc[df['ghi']>0, 'ghi'].mean():.0f}")
    print(f"NSRDB temp °C: min={df['temp_c'].min():.1f}  max={df['temp_c'].max():.1f}  mean={df['temp_c'].mean():.1f}")
    print(f"clearsky GHI peak: {df['clearsky_ghi'].max():.0f} W/m^2  → cloud loss avg: "
          f"{(1 - df.loc[df['clearsky_ghi']>0, 'ghi'].sum() / df.loc[df['clearsky_ghi']>0, 'clearsky_ghi'].sum()) * 100:.1f}%")
