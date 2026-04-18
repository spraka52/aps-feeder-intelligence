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
    start: str = "2025-06-01"
    days: int = 92               # Jun-Aug peak summer
    heatwave_windows: List[Tuple[str, str]] = None  # (start, end) inclusive
    ev_growth_pct: float = 0.0   # additional fleet evening-peak load as % of baseline
    bm_pv_kw_per_bus: float = 0.0  # behind-meter PV nameplate per bus (avg)
    seed: int = 7

    def __post_init__(self):
        if self.heatwave_windows is None:
            # Two synthetic but realistic Phoenix-like heatwave events
            self.heatwave_windows = [
                ("2025-07-08", "2025-07-14"),
                ("2025-08-12", "2025-08-18"),
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


def synth_irradiance(idx: pd.DatetimeIndex, lat: float = 33.45) -> np.ndarray:
    """Approximate hourly GHI (W/m^2) using a clear-sky model + cloud noise."""
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60.0

    # Solar declination
    decl = 23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0))
    # Hour angle (15° per hour from solar noon, assume local solar time ~ index hour - 1)
    h_angle = 15.0 * (hour - 12.0)
    # Solar elevation
    sin_alpha = (
        np.sin(np.radians(lat)) * np.sin(np.radians(decl))
        + np.cos(np.radians(lat)) * np.cos(np.radians(decl)) * np.cos(np.radians(h_angle))
    )
    sin_alpha = np.clip(sin_alpha, 0.0, None)
    ghi_clear = 1100.0 * sin_alpha  # peak ~1000 W/m^2

    # Stochastic clouds: AR(1) factor in [0.4, 1.0]
    cloud = np.ones(len(idx))
    rho = 0.85
    for i in range(1, len(cloud)):
        cloud[i] = rho * cloud[i - 1] + (1 - rho) * RNG.uniform(0.5, 1.0)
    ghi = ghi_clear * np.clip(cloud, 0.3, 1.0)
    return ghi


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

    temp = synth_temperature(idx, cfg.heatwave_windows)
    ghi = synth_irradiance(idx)
    hvac = hvac_multiplier(temp)
    hod = idx.hour.values
    dow = idx.dayofweek.values
    weekend = (dow >= 5).astype(float)
    week_factor = 1.0 - 0.06 * weekend

    base_shape = HOURLY_SHAPE[hod]
    ev_shape = EV_SHAPE[hod]

    # Use ALL buses from the topology graph so the GNN sees the full network.
    # Junction / transformer buses without spot loads get a small ambient draw
    # (e.g., line losses, a few unmodeled service drops).
    bus_list = sorted(fg.g.nodes())
    n_bus = len(bus_list)

    # Per-bus phase offset (geographic variation in evening peak timing)
    phase = RNG.uniform(-0.5, 0.5, size=n_bus)  # hours
    bus_noise_amp = RNG.uniform(0.04, 0.10, size=n_bus)

    # Behind-meter PV offset (kW) per bus — proportional to nameplate * GHI/1000
    pv_nameplate = RNG.uniform(0.0, 2.0 * cfg.bm_pv_kw_per_bus, size=n_bus) if cfg.bm_pv_kw_per_bus > 0 else np.zeros(n_bus)

    loads = np.zeros((n_bus, n), dtype=np.float32)
    for i, bus in enumerate(bus_list):
        nominal = SPOT_LOADS_KW.get(bus, 0.5)  # tiny ambient at non-load buses
        # phase-shifted hour shape via interpolation
        shifted_hod = (hod + phase[i]) % 24
        shape = np.interp(shifted_hod, np.arange(24), HOURLY_SHAPE)
        ev_extra = (cfg.ev_growth_pct / 100.0) * ev_shape * nominal
        pv_offset = pv_nameplate[i] * (ghi / 1000.0)
        noise = 1.0 + bus_noise_amp[i] * RNG.standard_normal(n)
        kw = nominal * shape * hvac * week_factor * noise + ev_extra - pv_offset
        loads[i] = np.clip(kw, 0.0, None)

    return {
        "time": idx,
        "temp_c": temp.astype(np.float32),
        "ghi": ghi.astype(np.float32),
        "ev_growth_pct": np.full(n, cfg.ev_growth_pct, dtype=np.float32),
        "bus_list": bus_list,
        "loads_kw": loads,
        "in_heatwave": _heatwave_mask(idx, cfg.heatwave_windows),
    }


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    p.add_argument("--days", type=int, default=92)
    args = p.parse_args()

    base_cfg = SimConfig(days=args.days, ev_growth_pct=0.0, bm_pv_kw_per_bus=0.0)
    base = synth_loads(base_cfg)
    f1 = save_dataset(args.out, base, "baseline")

    stress_cfg = SimConfig(days=args.days, ev_growth_pct=35.0, bm_pv_kw_per_bus=8.0)
    stress = synth_loads(stress_cfg)
    f2 = save_dataset(args.out, stress, "stress_ev35_pv8")

    print(f"Wrote: {f1}\nWrote: {f2}")
    print(
        f"baseline mean kW/bus={base['loads_kw'].mean():.2f}  "
        f"max kW/bus={base['loads_kw'].max():.2f}  heatwave hours={int(base['in_heatwave'].sum())}"
    )


if __name__ == "__main__":
    main()
