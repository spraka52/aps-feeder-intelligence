"""End-to-end pipeline smoke test (no Streamlit).

Reproduces what the dashboard does for one forecast window:
  1. Load model + datasets.
  2. Forecast a 24-hour window for baseline and stress scenarios.
  3. Run OpenDSS on each forecast hour.
  4. Print headline KPIs and the top actions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.synthesize import load_dataset
from data.topology import SPOT_LOADS_KW
from decisions.action_engine import build_actions, headline_kpis, actions_to_df
from models.dataset import FeederWindowDataset, WindowSpec
from models.predict import Forecaster
from physics.opendss_runner import run_forecast_horizon, summarize


REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "models" / "checkpoints" / "graphsage_gru.pt"
BASE = REPO / "data" / "synthetic" / "baseline.npz"
STRESS = REPO / "data" / "synthetic" / "stress_ev35_pv8.npz"


def run_one(npz_path: Path, label: str, t0: int, forecaster: Forecaster):
    ds = FeederWindowDataset(npz_path, WindowSpec(horizon_in=24, horizon_out=24))
    fcst = forecaster.forecast_window(ds, t0)
    res = run_forecast_horizon(fcst, forecaster.bus_order)
    times = ds.times[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
    hw = ds.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
    actions = build_actions(res, times, fcst, forecaster.bus_order, hw)
    kpi = headline_kpis(res, fcst, hw)
    print(f"\n=== {label} (window starts {ds.times[t0]}) ===")
    print(f"  peak_kw={kpi['peak_forecast_kw']:.0f}  avg_kw={kpi['avg_forecast_kw']:.0f}  "
          f"peak_loss={kpi['peak_loss_kw']:.1f}  v_viol={kpi['n_voltage_violations']}  "
          f"thermal={kpi['n_thermal_overloads']}  stress_hrs={kpi['n_stress_hours']}")
    df = actions_to_df(actions)
    if df.empty:
        print("  No violations — feeder is clean for this window.")
    else:
        print("  Top actions:")
        for _, row in df.head(5).iterrows():
            print(f"    [{row['priority']}] {row['kind']:>15s} @ {row['bus_or_line']:>8s} "
                  f"sev={row['severity']:.2f}  hours={row['hours_affected']}  "
                  f"-> {row['recommendation'][:90]}")
    return kpi, actions


def main():
    forecaster = Forecaster.load(CKPT)
    # Pick a window in the middle of the first heatwave
    ds_base = FeederWindowDataset(BASE, WindowSpec(horizon_in=24, horizon_out=24))
    hw_idx = np.where(ds_base.heatwave)[0]
    if hw_idx.size > 0:
        t0 = max(0, int(hw_idx[len(hw_idx) // 2]) - forecaster.horizon_in)
    else:
        t0 = 100
    run_one(BASE, "Baseline", t0, forecaster)
    run_one(STRESS, "Stress (heat + 35% EV evening growth)", t0, forecaster)


if __name__ == "__main__":
    main()
