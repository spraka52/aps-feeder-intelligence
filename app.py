"""APS Spatio-Temporal Feeder Intelligence — Streamlit utility dashboard."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from data.synthesize import load_dataset
from data.topology import COORDS, SPOT_LOADS_KW, build_graph
from decisions.action_engine import (
    actions_to_df, build_actions, headline_kpis,
)
from models.dataset import FeederWindowDataset, WindowSpec
from models.predict import Forecaster
from physics.opendss_runner import run_forecast_horizon


REPO = Path(__file__).resolve().parent
CKPT_PATH = REPO / "models" / "checkpoints" / "graphsage_gru.pt"
BASELINE_NPZ = REPO / "data" / "synthetic" / "baseline.npz"
STRESS_NPZ = REPO / "data" / "synthetic" / "stress_ev35_pv8.npz"


# --- Caching wrappers ------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading model…")
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


# --- Map helpers ----------------------------------------------------------- #

def _v_to_color(v: Optional[float]) -> List[int]:
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


def feeder_map_deck(
    bus_voltages_per_hour: List[Dict[str, float]],
    hour_idx: int,
    actions_df: pd.DataFrame,
):
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
            "v_label": f"{v:.3f} pu" if v is not None else "n/a",
            "nominal_kw": float(nominal_kw),
            "color": _v_to_color(v),
            "radius": radius,
        })
        if b in flagged:
            halos.append({
                "lat": lat, "lon": lon,
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
            "html": ("<b>Bus {bus}</b><br/>Voltage: <b>{v_label}</b><br/>Nominal load: {nominal_kw:.0f} kW"),
            "style": {"backgroundColor": "rgb(30,32,38)", "color": "white", "fontSize": "12px"},
        },
    )


def voltage_legend_chip():
    st.markdown(
        """
        <div style="display:flex; gap:24px; align-items:center; font-size:12px; opacity:0.85; flex-wrap:wrap; margin: 8px 0 4px;">
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#50C350;display:inline-block;"></span>~1.00 pu (healthy)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#EBC83C;display:inline-block;"></span>0.95 / 1.05 pu (limit)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#D23232;display:inline-block;"></span>≤0.93 / ≥1.07 pu (violation)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#FFC400;opacity:0.8;display:inline-block;"></span>action target halo</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:18px;height:4px;background:#B43232;display:inline-block;"></span>transformer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Plotly chart helpers -------------------------------------------------- #

def horizon_chart(forecast_kw_b: np.ndarray, forecast_kw_s: np.ndarray, times) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(times), y=forecast_kw_b.sum(axis=1),
                             mode="lines+markers", name="Baseline",
                             line=dict(width=2, color="#5BA3F5"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=list(times), y=forecast_kw_s.sum(axis=1),
                             mode="lines+markers", name="Stress (heat + EV)",
                             line=dict(width=2, color="#F08C3C"), marker=dict(size=5)))
    fig.update_layout(
        title=dict(text="Total feeder load · 24-hour forecast (kW)", font=dict(size=15)),
        xaxis_title=None, yaxis_title="kW",
        height=320, margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    return fig


def violations_chart(res_b, res_s, times) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(times),
                         y=[len(r.voltage_violations) for r in res_b],
                         name="Baseline · V violations", marker_color="#5BA3F5"))
    fig.add_trace(go.Bar(x=list(times),
                         y=[len(r.voltage_violations) for r in res_s],
                         name="Stress · V violations", marker_color="#F08C3C"))
    fig.update_layout(
        title=dict(text="Voltage violations per hour", font=dict(size=15)),
        barmode="group", height=300, margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def reg_tap_chart(res_b, res_s, times) -> go.Figure:
    if not res_b:
        return go.Figure()
    reg_names = sorted({k for r in res_b for k in r.regulator_taps.keys()})
    fig = go.Figure()
    palettes = {"baseline": "#5BA3F5", "stress": "#F08C3C"}
    line_dash = {"reg1": "solid", "reg2": "dash"}
    for kind, results, dash_offset in [("baseline", res_b, 1.0), ("stress", res_s, 0.7)]:
        for name in reg_names:
            ys = [r.regulator_taps.get(name, None) for r in results]
            fig.add_trace(go.Scatter(
                x=list(times), y=ys, mode="lines+markers",
                name=f"{kind} · {name.upper()}",
                line=dict(width=2, color=palettes[kind], dash=line_dash.get(name, "solid")),
                marker=dict(size=4),
                opacity=dash_offset,
            ))
    fig.update_layout(
        title=dict(text="Regulator tap positions · QSTS evolution across the 24-hour horizon", font=dict(size=15)),
        xaxis_title=None, yaxis_title="tap step (+ boost / – buck)",
        height=320, margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# --- Scenario presets ------------------------------------------------------ #

def _summer_scenarios(times: pd.DatetimeIndex, heatwave: np.ndarray) -> Dict[str, datetime]:
    df = pd.DataFrame({"time": times, "hw": heatwave})
    df["year"] = df["time"].dt.year
    df["date"] = df["time"].dt.date
    presets: Dict[str, datetime] = {}
    for y, sub in df.groupby("year"):
        hw_sub = sub[sub["hw"]]
        if not hw_sub.empty:
            hot_afternoons = hw_sub[hw_sub["time"].dt.hour == 18]
            if not hot_afternoons.empty:
                t = hot_afternoons["time"].iloc[len(hot_afternoons) // 2]
                presets[f"🔥 Heatwave evening · {y}"] = t.to_pydatetime()
        mild = sub[(~sub["hw"]) & (sub["time"].dt.hour == 7)]
        if not mild.empty:
            t = mild["time"].iloc[len(mild) // 2]
            presets[f"☀ Mild morning · {y}"] = t.to_pydatetime()
    return presets


# --- App body --------------------------------------------------------------- #

st.set_page_config(
    page_title="APS Feeder Intelligence",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# Generous custom CSS for breathing room + cleaner typography
st.markdown(
    """
    <style>
        /* Wider center column with more padding */
        .block-container {padding-top: 2.5rem !important; padding-bottom: 3rem !important; max-width: 1400px;}
        /* Bigger metric numbers, generous spacing */
        div[data-testid="stMetricValue"] {font-size: 2.0rem; font-weight: 600;}
        div[data-testid="stMetricLabel"] {font-size: 0.85rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.05em;}
        div[data-testid="stMetricDelta"] {font-size: 0.95rem;}
        div[data-testid="stMetric"] {
            background: rgba(60, 70, 90, 0.10);
            padding: 1rem 1.2rem;
            border-radius: 8px;
            border-left: 3px solid rgba(91, 163, 245, 0.5);
        }
        /* Tab labels bigger */
        button[data-baseweb="tab"] {font-size: 1rem !important; padding: 0.75rem 1.5rem !important;}
        /* Scenario chip card */
        .scenario-banner {
            background: linear-gradient(135deg, rgba(255,196,0,0.10), rgba(255,196,0,0.02));
            border-left: 4px solid #FFC400;
            padding: 1.0rem 1.2rem;
            border-radius: 6px;
            margin: 0.5rem 0 1.5rem;
            font-size: 0.95rem;
        }
        .scenario-banner b {color: #FFC400;}
        /* Section spacing */
        h2, h3 {margin-top: 1.5rem !important; margin-bottom: 0.6rem !important;}
        /* Keep table rows compact */
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col_logo, col_title, col_links = st.columns([1, 6, 2])
with col_title:
    st.markdown("## ⚡ APS Spatio-Temporal Feeder Intelligence")
    st.caption("GraphSAGE + GRU forecaster · OpenDSS QSTS validation · operator-ready decision layer")
with col_links:
    st.markdown(
        "<div style='text-align:right; padding-top:1.4rem;'>"
        "<a href='https://github.com/spraka52/aps-feeder-intelligence' style='text-decoration:none; padding:6px 12px; "
        "background:rgba(91,163,245,0.15); border-radius:5px; font-size:0.85rem;'>📦 GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )

if not CKPT_PATH.exists() or not BASELINE_NPZ.exists():
    st.error("Missing artifacts. Run `python -m data.synthesize --multi` then `python -m models.train --epochs 25`.")
    st.stop()

# Load data
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

# ================== TOP: scenario picker (horizontal bar) ================== #
st.markdown("##### 📅 Pick a scenario")
scenario_keys = list(scenarios.keys()) + ["✏ Custom date / hour"]
default_idx = 0
pick_scenario = st.radio(
    "Pick a scenario",
    scenario_keys,
    index=default_idx,
    horizontal=True,
    label_visibility="collapsed",
)

if pick_scenario == "✏ Custom date / hour":
    cc1, cc2, _ = st.columns([2, 1, 4])
    with cc1:
        default_day = scenarios[list(scenarios.keys())[0]].date() if scenarios else all_days[24]
        pick_day = st.date_input("Date", value=default_day,
                                 min_value=all_days[24], max_value=all_days[-2])
    with cc2:
        pick_hour = st.slider("Start hour", 0, 23, 6)
    scenario_label = f"Custom · {pick_day} {pick_hour:02d}:00"
else:
    scenario_ts = scenarios[pick_scenario]
    pick_day = scenario_ts.date()
    pick_hour = scenario_ts.hour
    scenario_label = pick_scenario

# Compute window
target_ts = pd.Timestamp(pick_day, tz=times.tz) + pd.Timedelta(hours=pick_hour)
deltas = (times - target_ts).asi8
t0_full = int(np.argmin(np.abs(deltas)))
t0 = max(0, t0_full - forecaster.horizon_in)
if t0 + forecaster.horizon_in + forecaster.horizon_out > len(times):
    t0 = len(times) - forecaster.horizon_in - forecaster.horizon_out
fcst_times = times[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
hw_in_window = int(ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out].sum())

# Scenario banner
st.markdown(
    f"""
    <div class='scenario-banner'>
      <b>{scenario_label}</b> &nbsp;·&nbsp;
      forecast horizon: {fcst_times[0].strftime('%a %b %d, %H:%M')} → {fcst_times[-1].strftime('%a %b %d, %H:%M')}
      &nbsp;·&nbsp; <b>{hw_in_window}/24</b> hours fall inside a heatwave window
    </div>
    """,
    unsafe_allow_html=True,
)

# Run forecasts + OpenDSS
fcst_base = forecaster.forecast_window(ds_base, t0)
fcst_stress = forecaster.forecast_window(ds_stress, t0)
bus_order = tuple(forecaster.bus_order)

res_base_d = _solve(fcst_base.astype(np.float32).tobytes(), bus_order, fcst_base.shape)
res_stress_d = _solve(fcst_stress.astype(np.float32).tobytes(), bus_order, fcst_stress.shape)
res_base = _to_hour_results(res_base_d)
res_stress = _to_hour_results(res_stress_d)

if not any(r.converged for r in res_base) or not any(r.converged for r in res_stress):
    st.warning("OpenDSS solver hit a stability error on this window — showing forecast panels only. Pick a different start.")

hw_mask = ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
actions_base = build_actions(res_base, fcst_times, fcst_base, list(bus_order), hw_mask)
actions_stress = build_actions(res_stress, fcst_times, fcst_stress, list(bus_order), hw_mask)
kpi_base = headline_kpis(res_base, fcst_base, hw_mask)
kpi_stress = headline_kpis(res_stress, fcst_stress, hw_mask)

# ================== KPI strip ============================================== #
st.markdown("### Operator KPIs · stress vs baseline")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak feeder load", f"{kpi_stress['peak_forecast_kw']:.0f} kW",
          f"{kpi_stress['peak_forecast_kw'] - kpi_base['peak_forecast_kw']:+.0f} vs base",
          help="Highest single-hour total kW across all 34 buses in the 24-hour forecast.")
c2.metric("Voltage violations", kpi_stress["n_voltage_violations"],
          f"{kpi_stress['n_voltage_violations'] - kpi_base['n_voltage_violations']:+d} vs base",
          delta_color="inverse",
          help="Count of (hour × bus) pairs where OpenDSS reported V outside [0.95, 1.05] pu.")
c3.metric("Thermal overloads", kpi_stress["n_thermal_overloads"],
          f"{kpi_stress['n_thermal_overloads'] - kpi_base['n_thermal_overloads']:+d} vs base",
          delta_color="inverse",
          help="Count of (hour × line) pairs where line current exceeded NormAmps.")
c4.metric("Peak losses", f"{kpi_stress['peak_loss_kw']:.0f} kW",
          f"{kpi_stress['peak_loss_kw'] - kpi_base['peak_loss_kw']:+.0f} vs base",
          delta_color="inverse",
          help="Worst-hour I²R losses summed over all lines, from the OpenDSS solve.")

st.divider()

# ================== Tabs: Map | Forecast | Actions | About =============== #
tab_map, tab_forecast, tab_actions, tab_about = st.tabs([
    "🗺  Operations Map",
    "📈  Forecast & Physics",
    "⚙  Action Center",
    "ℹ  Model & About",
])

# ---------------- Map tab --------------------------------------------------- #
with tab_map:
    fcst_hour_options = [t.strftime("%a %b %d · %H:%M") for t in fcst_times]
    default_hour_idx = 18 if len(fcst_hour_options) > 18 else len(fcst_hour_options) // 2

    cs1, cs2 = st.columns([3, 1])
    with cs1:
        selected_hour_label = st.select_slider(
            "Hour into the forecast",
            options=fcst_hour_options,
            value=fcst_hour_options[default_hour_idx],
            help="Slide to watch voltages evolve hour-by-hour. Yellow halo = bus flagged for action.",
        )
    with cs2:
        compare_view = st.toggle(
            "Show baseline beside stress",
            value=False,
            help="When off, you see one map (the stress scenario). Toggle on for a side-by-side comparison.",
        )
    map_hour = fcst_hour_options.index(selected_hour_label)
    voltage_legend_chip()

    voltages_per_hour_base = [r.bus_voltage_pu for r in res_base]
    voltages_per_hour_stress = [r.bus_voltage_pu for r in res_stress]
    worst_v_base = min(voltages_per_hour_base[map_hour].values()) if voltages_per_hour_base[map_hour] else float("nan")
    worst_v_stress = min(voltages_per_hour_stress[map_hour].values()) if voltages_per_hour_stress[map_hour] else float("nan")

    if compare_view:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(
                f"**Baseline** · worst V at this hour: "
                f"<span style='color:{'#D23232' if worst_v_base < 0.95 else '#50C350'};'><b>{worst_v_base:.3f} pu</b></span>",
                unsafe_allow_html=True,
            )
            st.pydeck_chart(
                feeder_map_deck(voltages_per_hour_base, map_hour, actions_to_df(actions_base)),
                height=520,
            )
        with mc2:
            st.markdown(
                f"**Stress (heat + EV)** · worst V at this hour: "
                f"<span style='color:{'#D23232' if worst_v_stress < 0.95 else '#50C350'};'><b>{worst_v_stress:.3f} pu</b></span>",
                unsafe_allow_html=True,
            )
            st.pydeck_chart(
                feeder_map_deck(voltages_per_hour_stress, map_hour, actions_to_df(actions_stress)),
                height=520,
            )
    else:
        st.markdown(
            f"**Stress scenario (heat + 35% EV evening growth)** · worst V at this hour: "
            f"<span style='color:{'#D23232' if worst_v_stress < 0.95 else '#50C350'};'><b>{worst_v_stress:.3f} pu</b></span>"
            f" &nbsp;·&nbsp; <span style='opacity:0.7;'>baseline worst: {worst_v_base:.3f} pu</span>",
            unsafe_allow_html=True,
        )
        st.pydeck_chart(
            feeder_map_deck(voltages_per_hour_stress, map_hour, actions_to_df(actions_stress)),
            height=620,
        )

# ---------------- Forecast tab --------------------------------------------- #
with tab_forecast:
    st.plotly_chart(horizon_chart(fcst_base, fcst_stress, fcst_times), width="stretch")
    st.divider()
    cf1, cf2 = st.columns(2)
    with cf1:
        st.plotly_chart(violations_chart(res_base, res_stress, fcst_times), width="stretch")
    with cf2:
        st.plotly_chart(reg_tap_chart(res_base, res_stress, fcst_times), width="stretch")

# ---------------- Action Center tab ---------------------------------------- #
with tab_actions:
    st.caption("Each violation is grouped by location and time, scored by how far out of bounds and how persistent, "
               "and converted into a sized recommendation a control-room operator could act on.")
    show_action_count = st.slider("Top N actions", 3, 15, 8)

    sub_stress, sub_base = st.tabs(["🔥 Stress scenario", "🌤 Baseline scenario"])

    def _render_actions(actions_list, label):
        df = actions_to_df(actions_list).head(show_action_count)
        if df.empty:
            st.success(f"No violations detected in the {label} forecast — feeder is operating within limits.")
            return
        df_display = df.rename(columns={
            "priority": "Pri.", "kind": "Kind", "bus_or_line": "Where",
            "when": "When (worst hour)", "hours_affected": "Hrs",
            "severity": "Severity", "target_kw": "Sized kW",
            "detail": "What happened", "recommendation": "Recommended action",
        })
        st.dataframe(
            df_display[["Pri.", "Kind", "Where", "When (worst hour)", "Hrs",
                        "Severity", "Sized kW", "What happened", "Recommended action"]],
            width="stretch", hide_index=True, height=380,
        )

    with sub_stress:
        _render_actions(actions_stress, "stress")
    with sub_base:
        _render_actions(actions_base, "baseline")

# ---------------- About tab ------------------------------------------------ #
with tab_about:
    cab1, cab2 = st.columns([1, 1])
    with cab1:
        st.markdown("### Model")
        st.markdown(
            f"""
            - **Architecture**: GraphSAGE (2 layers, 32 hidden) → GRU (64 hidden) → linear head
            - **Buses**: {len(forecaster.bus_order)} (IEEE 34-bus radial feeder)
            - **Horizon**: in {forecaster.horizon_in} h → out {forecaster.horizon_out} h
            - **Trainable parameters**: {forecaster.model.num_parameters():,}
            """
        )
        report_path = REPO / "models" / "checkpoints" / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            m = report["final_metrics"]
            st.markdown("**Held-out validation metrics**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Overall RMSE (kW)", f"{m['overall']['rmse']:.2f}")
            mc1.metric("Overall MAPE (%)", f"{m['overall']['mape']:.1f}")
            mc2.metric("Heatwave RMSE (kW)", f"{m['heatwave']['rmse']:.2f}")
            mc2.metric("Heatwave MAPE (%)", f"{m['heatwave']['mape']:.1f}")
            mc3.metric("Normal RMSE (kW)", f"{m['normal']['rmse']:.2f}")
            mc3.metric("Normal MAPE (%)", f"{m['normal']['mape']:.1f}")
            st.caption(f"Trained for {report['epochs']} epochs on {report.get('trainable_params', 'n/a'):,} params.")
    with cab2:
        st.markdown("### Data")
        st.markdown(
            f"""
            - **Topology**: IEEE 34-bus radial feeder (NetworkX + OpenDSS deck), Phoenix-area coordinates
            - **Temperature**: real NOAA NCEI ISD-Lite hourly · KPHX
            - **Irradiance**: real NREL NSRDB GOES Aggregated v4 · Phoenix
            - **Per-customer load shapes**: real NREL SMART-DS Austin P1R 2018
            - **EV stress**: NREL EVI-Pro inspired residential evening curve
            - **Coverage**: {len(times):,} hourly samples spanning {min(years_available)} – {max(years_available)}
            """
        )
        st.markdown("### How it works")
        st.markdown(
            """
            1. The GNN forecasts 24 h of per-bus load given the past 24 h.
            2. OpenDSS QSTS solves power flow each hour with regulator/cap state carried across.
            3. Violations get grouped, severity-ranked, and turned into sized operator actions.
            4. Subprocess isolation keeps the UI alive even if the OpenDSS native engine SIGILLs.
            """
        )

st.divider()
st.caption("Built for the APS / ASU AI for Energy hackathon. "
           "[Source on GitHub](https://github.com/spraka52/aps-feeder-intelligence) · "
           "real NOAA + NREL NSRDB + NREL SMART-DS data · "
           "OpenDSS QSTS physics validation.")
