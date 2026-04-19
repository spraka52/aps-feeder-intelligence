"""Generate Arizona-realistic spatio-temporal feeder data.

Produces an hourly dataset for a multi-month horizon with:
  * Air temperature (°C) — Phoenix-like diurnal + multi-day heatwave events.
  * GHI irradiance (W/m^2) — clear-sky model with stochastic cloud dimming.
  * Per-bus active load (kW) preserving spot-load proportions, scaled by:
        - climatology (HVAC sensitivity to temperature)
        - hour-of-day residential shape
        - day-of-week + holiday seasonality
        - bus-specific noise + small phase shift (geographic variation)
        - optional EV evening-peak growth (NREL EVI-Pro inspired curve)
        - optional behind-the-meter PV offset proportional to irradiance

This is synthetic but climatologically motivated — Phoenix sees July highs
near 41–46°C with overnight lows near 28–32°C. Heatwaves push HVAC load up
sharply and shift evening peaks later. Real NOAA / NSRDB / EVI-Pro inputs
can be dropped in by replacing the corresponding generator function.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .topology import SPOT_LOADS_KW, build_graph


RNG = np.random.default_rng(7)


@dataclass
class SimConfig:
    start: str = "2024-06-01"
    days: int = 92               # Jun-Aug peak summer
    heatwave_windows: List[Tuple[str, str]] = None  # (start, end) inclusive — auto if None+real
    ev_growth_pct: float = 0.0   # additional fleet evening-peak load as % of baseline
    bm_pv_kw_per_bus: float = 0.0  # behind-meter PV nameplate per bus (avg)
    seed: int = 7
    weather_source: str = "noaa"  # "noaa" or "synthetic"
    customer_source: str = "resstock"  # "resstock" (Phoenix), "smart_ds" (Austin), or "synthetic"

    def __post_init__(self):
        if self.heatwave_windows is None and self.weather_source == "synthetic":
            # Two procedural Phoenix-like heatwave events (synthetic mode only)
            self.heatwave_windows = [
                ("2024-07-08", "2024-07-14"),
                ("2024-08-12", "2024-08-18"),
            ]


# --- Weather: temperature & irradiance ---------------------------------------

def synth_temperature(idx: pd.DatetimeIndex, heatwaves: List[Tuple[str, str]]) -> np.ndarray:
    """Phoenix-like hourly temperature in °C."""
    hours = np.arange(len(idx))
    day_of_year = idx.dayofyear.values
    hour_of_day = idx.hour.values

    # Seasonal amplitude (warmer in mid-summer)
    seasonal = 4.0 * np.sin(2 * np.pi * (day_of_year - 172) / 365.0)
    base_high = 39.0 + seasonal           # daily high
    base_low = 27.0 + 0.6 * seasonal      # daily low
    amp = (base_high - base_low) / 2.0
    mid = (base_high + base_low) / 2.0
    # Peak temp ~ 16:00 local
    diurnal = -np.cos(2 * np.pi * (hour_of_day - 16) / 24.0)
    t = mid + amp * diurnal

    # Heatwave bumps: +5 to +8°C with a smooth ramp
    tz = idx.tz
    for hs, he in heatwaves:
        hs_ts = pd.Timestamp(hs).tz_localize(tz) if tz is not None else pd.Timestamp(hs)
        he_ts = (pd.Timestamp(he) + pd.Timedelta(hours=23))
        if tz is not None:
            he_ts = he_ts.tz_localize(tz)
        mask = (idx >= hs_ts) & (idx <= he_ts)
        if mask.any():
            n = mask.sum()
            ramp = np.sin(np.linspace(0, np.pi, n))  # parabola-shaped
            bump = 5.0 + 3.0 * ramp
            t[mask] += bump

    # Small synoptic noise
    t += RNG.normal(0.0, 0.6, size=len(t))
    return t


def synth_irradiance(idx: pd.DatetimeIndex, lat: float = 33.45,
                     cloud_frac: np.ndarray | None = None) -> np.ndarray:
    """Approximate hourly GHI (W/m^2) using a clear-sky model.

    If `cloud_frac` (0..1, 1=overcast) is provided — typically from NOAA sky
    coverage — we attenuate clear-sky GHI by (1 - 0.75*cloud_frac). Otherwise
    we use a stochastic AR(1) cloud factor.
    """
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60.0

    # Solar declination
    decl = 23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0))
    h_angle = 15.0 * (hour - 12.0)
    sin_alpha = (
        np.sin(np.radians(lat)) * np.sin(np.radians(decl))
        + np.cos(np.radians(lat)) * np.cos(np.radians(decl)) * np.cos(np.radians(h_angle))
    )
    sin_alpha = np.clip(sin_alpha, 0.0, None)
    ghi_clear = 1100.0 * sin_alpha

    if cloud_frac is not None:
        atten = 1.0 - 0.75 * np.clip(cloud_frac, 0.0, 1.0)
        return (ghi_clear * atten).astype(np.float32)

    # Synthetic cloud field: AR(1) in [0.4, 1.0]
    cloud = np.ones(len(idx))
    rho = 0.85
    for i in range(1, len(cloud)):
        cloud[i] = rho * cloud[i - 1] + (1 - rho) * RNG.uniform(0.5, 1.0)
    return ghi_clear * np.clip(cloud, 0.3, 1.0)


def _detect_heatwaves(idx: pd.DatetimeIndex, temp: np.ndarray,
                      threshold_c: float = 41.0, min_days: int = 3) -> List[Tuple[str, str]]:
    """Find runs of ≥ ``min_days`` consecutive calendar days with daily max ≥ threshold."""
    s = pd.Series(temp, index=idx)
    daily_max = s.resample("D").max()
    hot = daily_max >= threshold_c
    runs: List[Tuple[str, str]] = []
    if not hot.any():
        return runs
    in_run = False
    run_start: pd.Timestamp | None = None
    prev_day: pd.Timestamp | None = None
    for day, is_hot in hot.items():
        if is_hot:
            if not in_run:
                in_run, run_start = True, day
            prev_day = day
        else:
            if in_run and (prev_day - run_start).days + 1 >= min_days:
                runs.append((str(run_start.date()), str(prev_day.date())))
            in_run = False
            run_start = None
    if in_run and (prev_day - run_start).days + 1 >= min_days:
        runs.append((str(run_start.date()), str(prev_day.date())))
    return runs


# --- Demand ------------------------------------------------------------------

# Residential normalized hour-of-day shape (sums ~ 24).
HOURLY_SHAPE = np.array([
    0.78, 0.72, 0.68, 0.66, 0.66, 0.72, 0.85, 1.00,  # 0-7
    1.05, 1.02, 0.96, 0.92, 0.93, 0.96, 1.02, 1.10,  # 8-15
    1.18, 1.30, 1.45, 1.55, 1.50, 1.30, 1.05, 0.88,  # 16-23
])
HOURLY_SHAPE = HOURLY_SHAPE / HOURLY_SHAPE.mean()

# EV evening-peak charging shape (NREL EVI-Pro residential-dominated curve).
EV_SHAPE = np.array([
    0.30, 0.25, 0.18, 0.14, 0.12, 0.12, 0.10, 0.08,
    0.05, 0.04, 0.04, 0.04, 0.05, 0.06, 0.08, 0.12,
    0.30, 0.65, 0.95, 1.00, 0.90, 0.75, 0.55, 0.40,
])
EV_SHAPE = EV_SHAPE / EV_SHAPE.mean()


def hvac_multiplier(temp_c: np.ndarray) -> np.ndarray:
    """HVAC sensitivity: cooling load grows ~ quadratically above 24°C."""
    excess = np.clip(temp_c - 24.0, 0.0, None)
    return 1.0 + 0.018 * excess + 0.0009 * excess ** 2


def synth_loads(cfg: SimConfig) -> Dict[str, np.ndarray | pd.DatetimeIndex]:
    fg = build_graph()
    idx = pd.date_range(cfg.start, periods=cfg.days * 24, freq="h", tz="America/Phoenix")
    n = len(idx)

    if cfg.weather_source == "noaa":
        from .noaa_real import get_phoenix as get_noaa
        end_ts = idx[-1]
        wx = get_noaa(str(idx[0]), str(end_ts))
        wx = wx.set_index("time").reindex(idx).interpolate(limit=4).ffill().bfill()
        temp = wx["temp_c"].to_numpy().astype(np.float32)

        # Try NSRDB for real GHI; fall back to cloud-attenuated clear-sky model.
        try:
            from .nsrdb_real import get_phoenix as get_nsrdb, is_available as _nsrdb_ok
            if _nsrdb_ok():
                solar = get_nsrdb(str(idx[0]), str(end_ts))
                solar = solar.set_index("time").reindex(idx).interpolate(limit=4).ffill().bfill()
                ghi = solar["ghi"].to_numpy().astype(np.float32)
                irradiance_source = "nsrdb"
            else:
                raise RuntimeError("NSRDB unavailable")
        except Exception as e:
            print(f"[synthesize] NSRDB GHI unavailable ({e}); using NOAA cloud-attenuated clear-sky model")
            ghi = synth_irradiance(idx, cloud_frac=wx["cloud_frac"].to_numpy())
            irradiance_source = "noaa_cloud_attenuated"

        # Auto-detect heatwave windows from real temperature.
        heatwaves = _detect_heatwaves(idx, temp)
        cfg.heatwave_windows = heatwaves
        print(f"[synthesize] weather_source=noaa  irradiance_source={irradiance_source}  "
              f"heatwave_windows={len(heatwaves)}")
    else:
        temp = synth_temperature(idx, cfg.heatwave_windows or [])
        ghi = synth_irradiance(idx)
    hvac = hvac_multiplier(temp)
    hod = idx.hour.values
    dow = idx.dayofweek.values
    weekend = (dow >= 5).astype(float)
    week_factor = 1.0 - 0.06 * weekend

    base_shape = HOURLY_SHAPE[hod]
    ev_shape = EV_SHAPE[hod]

    # Use ALL buses from the topology graph so the GNN sees the full network.
    bus_list = sorted(fg.g.nodes())
    n_bus = len(bus_list)
    nominal_map = {b: SPOT_LOADS_KW.get(b, 0.5) for b in bus_list}

    # Behind-meter PV offset (kW) per bus — proportional to nameplate * GHI/1000
    pv_nameplate = RNG.uniform(0.0, 2.0 * cfg.bm_pv_kw_per_bus, size=n_bus) if cfg.bm_pv_kw_per_bus > 0 else np.zeros(n_bus)

    # Base per-bus shape: per Phoenix-specific source.
    #   "resstock"  → NREL ResStock + ComStock for climate zone 2B (Phoenix).
    #   "smart_ds"  → NREL SMART-DS Austin P1R 2018 (closest hot-climate analog).
    #   "synthetic" → procedural residential shape with phase shifts.
    customer_label = "procedural"
    base_kw = None

    if cfg.customer_source == "resstock":
        try:
            from .resstock_real import fetch_all, synth_bus_loads_resstock
            profiles = fetch_all(cache=True)
            if not profiles:
                raise RuntimeError("no ResStock profiles available")
            base_kw = synth_bus_loads_resstock(bus_list, nominal_map, idx, profiles, seed=cfg.seed)
            customer_label = "resstock"
        except Exception as e:
            print(f"[synthesize] ResStock unavailable ({e}); trying SMART-DS fallback")

    if base_kw is None and cfg.customer_source in {"resstock", "smart_ds"}:
        try:
            from .smart_ds import fetch_profiles, synth_bus_loads_smart_ds
            profiles = fetch_profiles(n_residential=8, n_commercial=5)
            base_kw = synth_bus_loads_smart_ds(bus_list, nominal_map, idx, profiles, seed=cfg.seed)
            customer_label = "smart_ds"
        except Exception as e:
            print(f"[synthesize] SMART-DS unavailable ({e}); falling back to procedural")

    if base_kw is None:
        base_kw = _procedural_bus_loads(bus_list, nominal_map, idx, hod, hvac, week_factor)
        customer_label = "procedural"

    # ResStock / SMART-DS profiles already encode HVAC for their training year
    # (2018). Apply a gentler Phoenix-specific overlay so heatwaves still push
    # loads above the embedded baseline.
    if customer_label in {"resstock", "smart_ds"}:
        phx_hvac = 1.0 + 0.005 * np.clip(temp - 24.0, 0.0, None) + 0.0003 * np.clip(temp - 24.0, 0.0, None) ** 2
        base_kw = base_kw * phx_hvac[None, :].astype(np.float32)

    # EV evening-peak overlay (per-bus, scaled to bus nominal) and PV offset.
    loads = np.zeros((n_bus, n), dtype=np.float32)
    for i, bus in enumerate(bus_list):
        nominal = nominal_map[bus]
        ev_extra = (cfg.ev_growth_pct / 100.0) * ev_shape * nominal
        pv_offset = pv_nameplate[i] * (ghi / 1000.0)
        loads[i] = np.clip(base_kw[i] + ev_extra - pv_offset, 0.0, None).astype(np.float32)
    print(f"[synthesize] customer_source={customer_label}  buses={n_bus}  "
          f"window={idx[0]} .. {idx[-1]}  hours={n}")

    return {
        "time": idx,
        "temp_c": temp.astype(np.float32),
        "ghi": ghi.astype(np.float32),
        "ev_growth_pct": np.full(n, cfg.ev_growth_pct, dtype=np.float32),
        "bus_list": bus_list,
        "loads_kw": loads,
        "in_heatwave": _heatwave_mask(idx, cfg.heatwave_windows or []),
    }


def _procedural_bus_loads(bus_list, nominal_map, idx, hod, hvac, week_factor):
    """Fallback: prior procedural per-bus shape (phase-shifted residential)."""
    n_bus = len(bus_list)
    n = len(idx)
    phase = RNG.uniform(-0.5, 0.5, size=n_bus)
    bus_noise_amp = RNG.uniform(0.04, 0.10, size=n_bus)
    base = np.zeros((n_bus, n), dtype=np.float32)
    for i, bus in enumerate(bus_list):
        nominal = nominal_map[bus]
        shifted_hod = (hod + phase[i]) % 24
        shape = np.interp(shifted_hod, np.arange(24), HOURLY_SHAPE)
        noise = 1.0 + bus_noise_amp[i] * RNG.standard_normal(n)
        base[i] = (nominal * shape * hvac * week_factor * noise).astype(np.float32)
    return base


def _heatwave_mask(idx: pd.DatetimeIndex, windows) -> np.ndarray:
    m = np.zeros(len(idx), dtype=bool)
    tz = idx.tz
    for hs, he in windows:
        hs_ts = pd.Timestamp(hs).tz_localize(tz) if tz is not None else pd.Timestamp(hs)
        he_ts = (pd.Timestamp(he) + pd.Timedelta(hours=23))
        if tz is not None:
            he_ts = he_ts.tz_localize(tz)
        sel = (idx >= hs_ts) & (idx <= he_ts)
        m |= np.asarray(sel)
    return m


def save_dataset(out_dir: Path, payload: Dict, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / f"{tag}.npz"
    times = payload["time"]
    if hasattr(times, "tz") and times.tz is not None:
        times_naive = times.tz_convert("UTC").tz_localize(None)
    else:
        times_naive = pd.DatetimeIndex(times)
    np.savez_compressed(
        file,
        time=times_naive.to_numpy().astype("datetime64[ns]"),
        temp_c=payload["temp_c"],
        ghi=payload["ghi"],
        ev_growth_pct=payload["ev_growth_pct"],
        loads_kw=payload["loads_kw"],
        bus_list=np.array(payload["bus_list"]),
        in_heatwave=payload["in_heatwave"],
    )
    return file


def load_dataset(file: Path) -> Dict:
    z = np.load(file, allow_pickle=False)
    return {
        "time": pd.to_datetime(z["time"]),
        "temp_c": z["temp_c"],
        "ghi": z["ghi"],
        "ev_growth_pct": z["ev_growth_pct"],
        "loads_kw": z["loads_kw"],
        "bus_list": [str(b) for b in z["bus_list"]],
        "in_heatwave": z["in_heatwave"],
    }


def synth_multi_window(windows: List[Tuple[str, int, str]], cfg_kwargs: dict) -> dict:
    """Stitch together per-summer datasets into a single multi-year payload.

    `windows` = list of (start, days, weather_source) tuples. The returned
    dict mirrors `synth_loads` output but covers all windows concatenated
    chronologically.
    """
    payloads = []
    for start, days, src in windows:
        cfg = SimConfig(start=start, days=days, weather_source=src, **cfg_kwargs)
        payloads.append(synth_loads(cfg))
    bus_list = payloads[0]["bus_list"]
    return {
        "time": pd.DatetimeIndex(np.concatenate([p["time"].asi8 for p in payloads])).tz_localize("UTC").tz_convert("America/Phoenix"),
        "temp_c": np.concatenate([p["temp_c"] for p in payloads]),
        "ghi": np.concatenate([p["ghi"] for p in payloads]),
        "ev_growth_pct": np.concatenate([p["ev_growth_pct"] for p in payloads]),
        "bus_list": bus_list,
        "loads_kw": np.concatenate([p["loads_kw"] for p in payloads], axis=1),
        "in_heatwave": np.concatenate([p["in_heatwave"] for p in payloads]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    p.add_argument("--multi", action="store_true",
                   help="Generate multi-year dataset (2024+2025 real, 2026 projected)")
    p.add_argument("--days", type=int, default=92)
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--source", default="noaa", choices=["noaa", "synthetic"])
    p.add_argument("--customers", default="resstock",
                   choices=["resstock", "smart_ds", "synthetic"])
    args = p.parse_args()

    # Four documented scenarios — none extrapolated, all generated from the same
    # underlying NOAA/NSRDB/ResStock+ComStock pipeline so they're comparable.
    SCENARIOS = [
        ("baseline",            0.0,  0.0),
        ("mild_stress_ev20_pv4",   20.0,  4.0),
        ("stress_ev35_pv8",        35.0,  8.0),
        ("severe_stress_ev50_pv12", 50.0, 12.0),
    ]

    if args.multi:
        # Three summers: 2024 + 2025 (real NOAA + NSRDB) + 2026 (synthetic
        # projection — real data not yet available for the future summer).
        windows = [
            ("2024-06-01", 92, "noaa"),
            ("2025-06-01", 92, "noaa"),
            ("2026-06-01", 92, "synthetic"),
        ]
        outputs = []
        for tag, ev, pv in SCENARIOS:
            print(f"\n=== Generating {tag} (ev_growth_pct={ev}, bm_pv_kw_per_bus={pv}) ===")
            data = synth_multi_window(windows, {
                "ev_growth_pct": ev, "bm_pv_kw_per_bus": pv,
                "customer_source": args.customers,
            })
            outputs.append((tag, data))
        label = "multi"
    else:
        outputs = []
        for tag, ev, pv in SCENARIOS:
            print(f"\n=== Generating {tag} (ev_growth_pct={ev}, bm_pv_kw_per_bus={pv}) ===")
            cfg = SimConfig(start=args.start, days=args.days,
                            ev_growth_pct=ev, bm_pv_kw_per_bus=pv,
                            weather_source=args.source,
                            customer_source=args.customers)
            outputs.append((tag, synth_loads(cfg)))
        label = "single"

    written = []
    for tag, data in outputs:
        f = save_dataset(args.out, data, tag)
        written.append(f)

    base_data = outputs[0][1]
    hw_hours = int(base_data["in_heatwave"].sum())
    print("\nWrote:")
    for f in written:
        print(f"  {f}")
    print(f"mode={label}  customers={args.customers}")
    print(f"baseline window: {base_data['time'][0]} .. {base_data['time'][-1]}  hours={len(base_data['time'])}")
    print(f"baseline temp °C: min={base_data['temp_c'].min():.1f} max={base_data['temp_c'].max():.1f} mean={base_data['temp_c'].mean():.1f}")
    print(f"baseline kW: mean={base_data['loads_kw'].mean():.2f}  max={base_data['loads_kw'].max():.2f}  "
          f"heatwave hours={hw_hours}")


if __name__ == "__main__":
    main()
