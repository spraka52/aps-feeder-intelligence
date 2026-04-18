"""APS Spatio-Temporal Feeder Intelligence — Streamlit utility dashboard.

End-to-end demo:
  1. Load the IEEE 34-bus topology + a trained GraphSAGE+GRU checkpoint.
  2. Pick a forecast window (preset scenarios or custom date/hour).
  3. Compare a baseline scenario against a stress scenario (heatwave + EV peak).
  4. Run OpenDSS QSTS on each forecast hour to detect voltage / thermal violations.
  5. Render the spatio-temporal map and an Action Center for operators.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import torch

from data.synthesize import load_dataset
from data.topology import COORDS, SPOT_LOADS_KW, build_graph
from decisions.action_engine import (
    actions_to_df, build_actions, headline_kpis,
)
from models.dataset import FeederWindowDataset, WindowSpec
from models.predict import Forecaster
from physics.opendss_runner import run_forecast_horizon, summarize


REPO = Path(__file__).resolve().parent
CKPT_PATH = REPO / "models" / "checkpoints" / "graphsage_gru.pt"
BASELINE_NPZ = REPO / "data" / "synthetic" / "baseline.npz"
STRESS_NPZ = REPO / "data" / "synthetic" / "stress_ev35_pv8.npz"


# --- Caching wrappers ------------------------------------------------------- #
# Cache keys include the file's mtime + size so a fresh deploy with new .npz
# invalidates the previous app's cached dataset automatically.

@st.cache_resource(show_spinner="Loading model checkpoint…")
def _get_forecaster(ckpt_mtime: float):
    return Forecaster.load(CKPT_PATH)


@st.cache_data(show_spinner="Loading dataset…")
def _get_dataset(npz_path_str: str, mtime: float, size: int):
    return FeederWindowDataset(Path(npz_path_str), WindowSpec(horizon_in=24, horizon_out=24))


@st.cache_data(show_spinner="Solving OpenDSS power flow…")
def _solve(forecast_kw_bytes: bytes, bus_order: tuple, shape: tuple) -> List[dict]:
    from physics.opendss_runner import _hourresult_to_dict
    forecast_kw = np.frombuffer(forecast_kw_bytes, dtype=np.float32).reshape(shape)
    results = run_forecast_horizon(forecast_kw, list(bus_order))
    return [_hourresult_to_dict(r) for r in results]


def _to_hour_results(dicts: List[dict]):
    from physics.opendss_runner import HourResult
    return [HourResult(**d) for d in dicts]


def _file_signature(p: Path) -> Tuple[float, int]:
    s = p.stat()
    return (s.st_mtime, s.st_size)


# --- Visualisation helpers -------------------------------------------------- #

def _v_to_color(v: Optional[float]) -> List[int]:
    """Map a voltage in pu to an RGBA color (red→amber→green→amber→red)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [128, 128, 128, 200]
    dist = abs(v - 1.00) / 0.05
    dist = max(0.0, min(dist, 1.5))
    if dist <= 0.5:
        t = dist / 0.5
        r = int(80 + (235 - 80) * t)
        g = int(195 + (200 - 195) * t)
        b = int(80 + (60 - 80) * t)
    else:
        t = min((dist - 0.5) / 1.0, 1.0)
        r = int(235 + (210 - 235) * t)
        g = int(200 + (50 - 200) * t)
        b = int(60 + (50 - 60) * t)
    return [r, g, b, 230]


def feeder_map(
    bus_voltages_per_hour: List[Dict[str, float]],
    hour_idx: int,
    actions_df: pd.DataFrame,
    title: str,
):
    """Geographic map of the feeder using PyDeck on a Carto dark basemap."""
    fg = build_graph()
    voltages = bus_voltages_per_hour[min(hour_idx, len(bus_voltages_per_hour) - 1)]

    flagged = set()
    if not actions_df.empty:
        flagged = set(actions_df["bus_or_line"].astype(str).tolist())

    nodes, halos, labels = [], [], []
    for b in fg.g.nodes():
        if b not in COORDS:
            continue
        lat, lon = COORDS[b]
        v = voltages.get(b)
        nominal_kw = SPOT_LOADS_KW.get(b, 0.0)
        radius = 35 + 0.55 * float(nominal_kw)
        nodes.append({
            "bus": b, "lat": lat, "lon": lon,
            "v_pu": float(v) if v is not None else None,
            "v_label": f"{v:.3f} pu" if v is not None else "n/a",
            "nominal_kw": float(nominal_kw),
            "color": _v_to_color(v),
            "radius": radius,
            "is_flagged": b in flagged,
        })
        if b in flagged:
            halos.append({
                "bus": b, "lat": lat, "lon": lon,
                "radius": radius * 2.4,
                "color": [255, 196, 0, 200],
            })
        if b in flagged or nominal_kw >= 100:
            labels.append({
                "lat": lat, "lon": lon,
                "text": f"{b}" + ("  ⚠" if b in flagged else ""),
            })

    edges = []
    for u, v, data in fg.g.edges(data=True):
        if u not in COORDS or v not in COORDS:
            continue
        edges.append({
            "from_bus": u, "to_bus": v,
            "kind": data.get("kind", "line"),
            "color": [180, 50, 50, 220] if data.get("kind") == "transformer" else [120, 130, 140, 200],
            "width": 5 if data.get("kind") == "transformer" else 3,
            "path": [[COORDS[u][1], COORDS[u][0]], [COORDS[v][1], COORDS[v][0]]],
        })

    lats = [c[0] for c in COORDS.values()]
    lons = [c[1] for c in COORDS.values()]
    view = pdk.ViewState(
        latitude=(min(lats) + max(lats)) / 2,
        longitude=(min(lons) + max(lons)) / 2,
        zoom=11.6, pitch=35, bearing=0,
    )

    layers = [
        pdk.Layer("PathLayer", data=edges, get_path="path",
                  get_color="color", get_width="width",
                  width_min_pixels=2, width_max_pixels=6, pickable=True),
        pdk.Layer("ScatterplotLayer", data=halos,
                  get_position=["lon", "lat"], get_radius="radius",
                  get_fill_color="color", stroked=False, opacity=0.55),
        pdk.Layer("ScatterplotLayer", data=nodes,
                  get_position=["lon", "lat"], get_radius="radius",
                  get_fill_color="color", stroked=True,
                  get_line_color=[20, 25, 30, 240], line_width_min_pixels=1,
                  pickable=True),
        pdk.Layer("TextLayer", data=labels,
                  get_position=["lon", "lat"],
                  get_text="text", get_size=14,
                  get_color=[235, 235, 240, 255],
                  get_alignment_baseline="'bottom'",
                  get_text_anchor="'middle'",
                  background=True,
                  background_padding=[3, 1],
                  get_background_color=[15, 22, 30, 200]),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="dark",
        tooltip={
            "html": (
                "<b>Bus {bus}</b><br/>"
                "Voltage: <b>{v_label}</b><br/>"
                "Nominal load: {nominal_kw:.0f} kW"
            ),
            "style": {"backgroundColor": "rgb(30,32,38)", "color": "white", "fontSize": "12px"},
        },
    )


def voltage_legend():
    return st.markdown(
        """
        <div style="display:flex; gap:20px; align-items:center; font-size:12px; opacity:0.85; flex-wrap:wrap;">
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#50C350;display:inline-block;"></span>~1.00 pu (healthy)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#EBC83C;display:inline-block;"></span>0.95 / 1.05 pu (limit)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#D23232;display:inline-block;"></span>≤0.93 / ≥1.07 pu (violation)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#FFC400;opacity:0.8;display:inline-block;"></span>action target</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:18px;height:4px;background:#B43232;display:inline-block;"></span>transformer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizon_chart(forecast_kw: np.ndarray, times, label: str, color: str):
    feeder_total = forecast_kw.sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times), y=feeder_total, mode="lines+markers",
        name=label, line=dict(width=2, color=color),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=dict(text=f"{label} — total feeder load (kW)", font=dict(size=14)),
        xaxis_title=None, yaxis_title="kW",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def violations_chart(results, times, label: str):
    n_v = [len(r.voltage_violations) for r in results]
    n_t = [len(r.thermal_overloads) for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(times), y=n_v, name="Voltage", marker_color="#C84B4B"))
    fig.add_trace(go.Bar(x=list(times), y=n_t, name="Thermal", marker_color="#EBC83C"))
    fig.update_layout(
        title=dict(text=f"{label} — violations per hour", font=dict(size=14)),
        barmode="stack", height=240, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def reg_tap_chart(results, times, label: str):
    if not results:
        return go.Figure()
    reg_names = sorted({k for r in results for k in r.regulator_taps.keys()})
    fig = go.Figure()
    palette = ["#5BA3F5", "#F08C3C"]
    for i, name in enumerate(reg_names):
        ys = [r.regulator_taps.get(name, None) for r in results]
        fig.add_trace(go.Scatter(
            x=list(times), y=ys, mode="lines+markers",
            name=name.upper(), line=dict(width=2, color=palette[i % len(palette)]),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=dict(text=f"{label} — regulator tap positions (QSTS)", font=dict(size=14)),
        xaxis_title=None, yaxis_title="tap step (+ boost / – buck)",
        height=240, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# --- Scenario presets ------------------------------------------------------- #

def _summer_scenarios(times: pd.DatetimeIndex, heatwave: np.ndarray) -> Dict[str, datetime]:
    """Build a curated set of one-click scenario starts from the dataset.

    Picks one heatwave-day-evening per available year + one mild morning per year,
    so the dropdown always offers diverse choices without scrolling through dates.
    """
    df = pd.DataFrame({"time": times, "hw": heatwave})
    df["year"] = df["time"].dt.year
    df["date"] = df["time"].dt.date
    presets: Dict[str, datetime] = {}

    for y, sub in df.groupby("year"):
        # Heatwave evening (highest single-hour load proxy = afternoon during a hw day)
        hw_sub = sub[sub["hw"]]
        if not hw_sub.empty:
            # Pick a representative hot afternoon at 18:00 on a heatwave day
            hot_afternoons = hw_sub[hw_sub["time"].dt.hour == 18]
            if not hot_afternoons.empty:
                t = hot_afternoons["time"].iloc[len(hot_afternoons) // 2]
                presets[f"🔥 Heatwave evening · {y}"] = t.to_pydatetime()
        # Mild morning (non-heatwave morning)
        mild = sub[(~sub["hw"]) & (sub["time"].dt.hour == 7)]
        if not mild.empty:
            t = mild["time"].iloc[len(mild) // 2]
            presets[f"☀ Mild morning · {y}"] = t.to_pydatetime()
    return presets


# --- App body --------------------------------------------------------------- #

st.set_page_config(page_title="APS Feeder Intelligence", layout="wide", page_icon="⚡")

# Custom CSS for a slightly tighter look + scenario chips
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.8rem;}
        div[data-testid="stMetricDelta"] {font-size: 0.85rem;}
        .scenario-card {
            background: linear-gradient(135deg, rgba(255,196,0,0.08), rgba(255,196,0,0));
            border-left: 3px solid #FFC400;
            padding: 0.85rem 1rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ APS Spatio-Temporal Feeder Intelligence")
st.caption(
    "GraphSAGE + GRU forecaster on the IEEE 34-bus feeder, validated by OpenDSS QSTS, "
    "with a utility-facing decision layer."
)

if not CKPT_PATH.exists() or not BASELINE_NPZ.exists():
    st.error(
        "Missing artifacts. Run the pipeline first:\n\n"
        "    python -m data.synthesize --multi\n"
        "    python -m models.train --epochs 25"
    )
    st.stop()

# Cache keys derived from file signatures so a refreshed dataset invalidates cleanly.
ckpt_sig = _file_signature(CKPT_PATH)
base_sig = _file_signature(BASELINE_NPZ)
stress_sig = _file_signature(STRESS_NPZ)

forecaster = _get_forecaster(ckpt_sig[0])
ds_base = _get_dataset(str(BASELINE_NPZ), *base_sig)
ds_stress = _get_dataset(str(STRESS_NPZ), *stress_sig)

times = ds_base.times
all_days = sorted({t.date() for t in times})
years_available = sorted({t.year for t in times})
scenarios = _summer_scenarios(times, ds_base.heatwave)

# ---------------- Sidebar (scenario picker + display controls) -------------- #
with st.sidebar:
    st.header("📅 Forecast window")
    st.caption(f"Dataset spans **{min(years_available)} – {max(years_available)}** "
               f"({len(times):,} hourly samples).")

    scenario_keys = ["Custom"] + list(scenarios.keys())
    default_idx = 1 if len(scenario_keys) > 1 else 0  # default to first preset
    pick_scenario = st.radio(
        "Pick a scenario",
        scenario_keys,
        index=default_idx,
        help="Curated scenarios jump straight to a representative window. "
             "Pick *Custom* to choose any date / hour from the dataset.",
    )

    if pick_scenario == "Custom":
        # Custom date + hour
        default_day = scenarios[list(scenarios.keys())[0]].date() if scenarios else all_days[24]
        pick_day = st.date_input(
            "Forecast start date",
            value=default_day,
            min_value=all_days[24],
            max_value=all_days[-2],
        )
        pick_hour = st.slider("Start hour (local)", 0, 23, 6,
                              help="The 24-hour forecast window begins at this hour.")
        scenario_label = f"Custom · {pick_day} {pick_hour:02d}:00"
    else:
        scenario_ts = scenarios[pick_scenario]
        pick_day = scenario_ts.date()
        pick_hour = scenario_ts.hour
        scenario_label = pick_scenario
        st.markdown(
            f"<div class='scenario-card'><b>{pick_scenario}</b><br/>"
            f"<small>start = {pick_day} {pick_hour:02d}:00 local</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("⚙ Display")
    show_action_count = st.slider("Top N actions", 3, 15, 8,
                                  help="How many ranked operator actions to show in the Action Center.")
    show_qsts = st.checkbox("Show regulator-tap chart (QSTS)", value=True)

    st.markdown("---")
    st.caption("**Model**")
    st.code(
        f"buses       : {len(forecaster.bus_order)}\n"
        f"horizon_in  : {forecaster.horizon_in} h\n"
        f"horizon_out : {forecaster.horizon_out} h\n"
        f"params      : {forecaster.model.num_parameters():,}",
        language="text",
    )

    st.caption("[GitHub repo ↗](https://github.com/spraka52/aps-feeder-intelligence)")

# ---------------- Compute the forecast window ------------------------------- #
target_ts = pd.Timestamp(pick_day, tz=times.tz) + pd.Timedelta(hours=pick_hour)
deltas = (times - target_ts).asi8
t0_full = int(np.argmin(np.abs(deltas)))
t0 = max(0, t0_full - forecaster.horizon_in)
if t0 + forecaster.horizon_in + forecaster.horizon_out > len(times):
    t0 = len(times) - forecaster.horizon_in - forecaster.horizon_out
fcst_times = times[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]

hw_in_window = int(ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out].sum())

# ---------------- Header strip with scenario context ------------------------ #
st.markdown(
    f"""
    <div style="background:rgba(70,80,100,0.18); padding:0.85rem 1rem; border-radius:6px; margin-bottom:1rem;">
      <span style="opacity:0.7;">SCENARIO</span> &nbsp; <b>{scenario_label}</b> &nbsp; · &nbsp;
      <span style="opacity:0.7;">FORECAST</span> &nbsp; {fcst_times[0].strftime('%a %b %d, %H:%M')} → {fcst_times[-1].strftime('%a %b %d, %H:%M')} &nbsp; · &nbsp;
      <span style="opacity:0.7;">HEATWAVE HOURS IN WINDOW</span> &nbsp; <b>{hw_in_window}/24</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Generate forecasts + run OpenDSS -------------------------- #
fcst_base = forecaster.forecast_window(ds_base, t0)
fcst_stress = forecaster.forecast_window(ds_stress, t0)
bus_order = tuple(forecaster.bus_order)

res_base_d = _solve(fcst_base.astype(np.float32).tobytes(), bus_order, fcst_base.shape)
res_stress_d = _solve(fcst_stress.astype(np.float32).tobytes(), bus_order, fcst_stress.shape)
res_base = _to_hour_results(res_base_d)
res_stress = _to_hour_results(res_stress_d)

if not any(r.converged for r in res_base) or not any(r.converged for r in res_stress):
    st.warning(
        "OpenDSS solver hit a stability error on this window — showing forecast "
        "panels only. Pick a different start date / hour to retry."
    )

hw_mask = ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]

actions_base = build_actions(res_base, fcst_times, fcst_base, list(bus_order), hw_mask)
actions_stress = build_actions(res_stress, fcst_times, fcst_stress, list(bus_order), hw_mask)

kpi_base = headline_kpis(res_base, fcst_base, hw_mask)
kpi_stress = headline_kpis(res_stress, fcst_stress, hw_mask)

# ---------------- KPI strip ------------------------------------------------- #
st.subheader("Operator KPIs · stress vs baseline")
st.caption("Stress = same forecast window but with a 35 % EV evening-peak overlay and the heatwave temperature signal pushed harder.")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Peak feeder load (kW)",
    f"{kpi_stress['peak_forecast_kw']:.0f}",
    f"{kpi_stress['peak_forecast_kw'] - kpi_base['peak_forecast_kw']:+.0f} vs base",
    help="Highest single-hour total kW across all 34 buses in the 24-hour forecast.",
)
c2.metric(
    "Voltage violations",
    kpi_stress["n_voltage_violations"],
    f"{kpi_stress['n_voltage_violations'] - kpi_base['n_voltage_violations']:+d} vs base",
    delta_color="inverse",
    help="Count of (hour × bus) pairs where OpenDSS reported V outside [0.95, 1.05] pu.",
)
c3.metric(
    "Thermal overloads",
    kpi_stress["n_thermal_overloads"],
    f"{kpi_stress['n_thermal_overloads'] - kpi_base['n_thermal_overloads']:+d} vs base",
    delta_color="inverse",
    help="Count of (hour × line) pairs where line current exceeded NormAmps.",
)
c4.metric(
    "Peak losses (kW)",
    f"{kpi_stress['peak_loss_kw']:.0f}",
    f"{kpi_stress['peak_loss_kw'] - kpi_base['peak_loss_kw']:+.0f} vs base",
    delta_color="inverse",
    help="Worst-hour I²R losses summed over all lines, from the OpenDSS solve.",
)

# ---------------- Spatial maps --------------------------------------------- #
st.subheader("🗺 Spatio-temporal feeder map")

fcst_hour_options = [t.strftime("%a %b %d · %H:%M") for t in fcst_times]
default_hour_idx = 18 if len(fcst_hour_options) > 18 else len(fcst_hour_options) // 2
selected_hour_label = st.select_slider(
    "Hour into the forecast",
    options=fcst_hour_options,
    value=fcst_hour_options[default_hour_idx],
    help="Slide to watch voltages evolve hour-by-hour. Yellow halo = bus flagged for action.",
)
map_hour = fcst_hour_options.index(selected_hour_label)
voltage_legend()

mc1, mc2 = st.columns(2)
voltages_per_hour_base = [r.bus_voltage_pu for r in res_base]
voltages_per_hour_stress = [r.bus_voltage_pu for r in res_stress]
worst_v_base = min(voltages_per_hour_base[map_hour].values()) if voltages_per_hour_base[map_hour] else float("nan")
worst_v_stress = min(voltages_per_hour_stress[map_hour].values()) if voltages_per_hour_stress[map_hour] else float("nan")
with mc1:
    st.markdown(
        f"**Baseline** · worst V at this hour: "
        f"<span style='color:{'#D23232' if worst_v_base < 0.95 else '#50C350'};'><b>{worst_v_base:.3f} pu</b></span>",
        unsafe_allow_html=True,
    )
    st.pydeck_chart(
        feeder_map(voltages_per_hour_base, map_hour, actions_to_df(actions_base),
                   f"Baseline @ {fcst_times[map_hour]}"),
        height=460,
    )
with mc2:
    st.markdown(
        f"**Stress (heat + EV)** · worst V at this hour: "
        f"<span style='color:{'#D23232' if worst_v_stress < 0.95 else '#50C350'};'><b>{worst_v_stress:.3f} pu</b></span>",
        unsafe_allow_html=True,
    )
    st.pydeck_chart(
        feeder_map(voltages_per_hour_stress, map_hour, actions_to_df(actions_stress),
                   f"Stress @ {fcst_times[map_hour]}"),
        height=460,
    )

# ---------------- Time-series comparisons ---------------------------------- #
st.subheader("📈 Forecast horizon")
hc1, hc2 = st.columns(2)
with hc1:
    st.plotly_chart(horizon_chart(fcst_base, fcst_times, "Baseline", "#5BA3F5"), width="stretch")
    st.plotly_chart(violations_chart(res_base, fcst_times, "Baseline"), width="stretch")
    if show_qsts:
        st.plotly_chart(reg_tap_chart(res_base, fcst_times, "Baseline"), width="stretch")
with hc2:
    st.plotly_chart(horizon_chart(fcst_stress, fcst_times, "Stress (heat+EV)", "#F08C3C"), width="stretch")
    st.plotly_chart(violations_chart(res_stress, fcst_times, "Stress (heat+EV)"), width="stretch")
    if show_qsts:
        st.plotly_chart(reg_tap_chart(res_stress, fcst_times, "Stress (heat+EV)"), width="stretch")

# ---------------- Action Center -------------------------------------------- #
st.subheader("⚙ Action Center · prioritized utility interventions")
st.caption("Each violation is grouped by location and time, scored by how far out of bounds and how persistent, "
           "and converted into a sized recommendation a control-room operator could act on.")

tab_stress, tab_base = st.tabs(["🔥 Stress scenario", "🌤 Baseline scenario"])
for tab, actions_list, label in [
    (tab_stress, actions_stress, "stress"),
    (tab_base, actions_base, "baseline"),
]:
    with tab:
        df = actions_to_df(actions_list).head(show_action_count)
        if df.empty:
            st.success(f"No violations detected in the {label} forecast — feeder is operating within limits.")
        else:
            df_display = df.rename(columns={
                "priority": "Pri.", "kind": "Kind", "bus_or_line": "Where",
                "when": "When (worst hour)", "hours_affected": "Hrs affected",
                "severity": "Severity", "target_kw": "Sized kW",
                "detail": "What happened", "recommendation": "Recommended action",
            })
            st.dataframe(
                df_display[["Pri.", "Kind", "Where", "When (worst hour)", "Hrs affected",
                            "Severity", "Sized kW", "What happened", "Recommended action"]],
                width="stretch", hide_index=True,
            )

# ---------------- Model performance + Why this matters --------------------- #
with st.expander("📊 Model performance (held-out validation)"):
    report_path = REPO / "models" / "checkpoints" / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        m = report["final_metrics"]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Overall RMSE (kW)", f"{m['overall']['rmse']:.2f}")
        rc1.metric("Overall MAE (kW)", f"{m['overall']['mae']:.2f}")
        rc1.metric("Overall MAPE (%)", f"{m['overall']['mape']:.2f}")
        rc2.metric("Heatwave RMSE (kW)", f"{m['heatwave']['rmse']:.2f}")
        rc2.metric("Heatwave MAE (kW)", f"{m['heatwave']['mae']:.2f}")
        rc2.metric("Heatwave MAPE (%)", f"{m['heatwave']['mape']:.2f}")
        rc3.metric("Normal RMSE (kW)", f"{m['normal']['rmse']:.2f}")
        rc3.metric("Normal MAE (kW)", f"{m['normal']['mae']:.2f}")
        rc3.metric("Normal MAPE (%)", f"{m['normal']['mape']:.2f}")
        st.caption(
            f"Trained for {report['epochs']} epochs on {report.get('trainable_params', 'n/a'):,} "
            f"parameters; checkpoint at `{Path(report['checkpoint']).name}`."
        )
    else:
        st.info("Training report not found — run `python -m models.train` to generate one.")

with st.expander("ℹ Why this matters"):
    st.markdown(
        """
        - **Spatio-temporal:** GraphSAGE layers mix information across the 34-bus
          feeder; the GRU captures the diurnal evolution. Temperature, irradiance,
          and EV-growth all enter as drivers — not just labels.
        - **Real data:** Phoenix Sky Harbor temperature is from NOAA NCEI, satellite
          irradiance from NREL NSRDB, per-customer load shapes from NREL SMART-DS.
        - **Physics-validated:** every forecast hour is solved with OpenDSS in
          quasi-static time-series mode so regulator taps and capacitor switching
          accumulate state across the 24-hour horizon. The dashboard never claims
          a constraint that doesn't actually exist in the power-flow solution.
        - **Decision-ready:** the Action Center groups violations by location, scores
          severity by how far out of bounds and how persistent, and proposes a sized
          intervention (battery dispatch, Volt-VAR, deferrable-load shed) — the kind
          of punch list a control-room operator could act on.
        """
    )
