"""APS Spatio-Temporal Feeder Intelligence — operator + planner dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
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
    actions_to_df as ops_actions_to_df,
    build_actions, headline_kpis,
)
from decisions.planner_actions import (
    aggregate_weekly_violations, build_planner_actions,
    actions_to_df as planner_actions_to_df, _bus_day_hours_matrix,
)
from models.dataset import FeederWindowDataset, WindowSpec
from models.predict import Forecaster
from physics.opendss_runner import run_forecast_horizon


REPO = Path(__file__).resolve().parent
CKPT_PATH = REPO / "models" / "checkpoints" / "graphsage_gru.pt"
BASELINE_NPZ = REPO / "data" / "synthetic" / "baseline.npz"
STRESS_NPZ = REPO / "data" / "synthetic" / "stress_ev35_pv8.npz"


# --- Caching --------------------------------------------------------------- #

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


@st.cache_data(show_spinner="Computing week-aggregate physics for the planner…")
def _solve_week_truth(week_kw_bytes: bytes, bus_order: tuple, day_shape: tuple,
                      week_label: str) -> List[List[dict]]:
    """Run OpenDSS day-by-day on the synthesized "ground truth" loads
    (rather than the 24-hour-ahead forecasts) to produce a clean weekly view."""
    from physics.opendss_runner import _hourresult_to_dict
    week_kw = np.frombuffer(week_kw_bytes, dtype=np.float32).reshape(day_shape)
    out: List[List[dict]] = []
    for d in range(week_kw.shape[0]):
        day_loads = week_kw[d]  # [24, N]
        day_results = run_forecast_horizon(day_loads, list(bus_order))
        out.append([_hourresult_to_dict(r) for r in day_results])
    return out


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
        # planner_actions uses 'bus', operator uses 'bus_or_line'
        col = "bus_or_line" if "bus_or_line" in actions_df.columns else "bus"
        flagged = set(actions_df[col].astype(str).tolist())

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
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#50C350;display:inline-block;"></span>~1.00 pu</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#EBC83C;display:inline-block;"></span>limit</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#D23232;display:inline-block;"></span>violation</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#FFC400;opacity:0.8;display:inline-block;"></span>action target</span>
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
                         name="Baseline", marker_color="#5BA3F5"))
    fig.add_trace(go.Bar(x=list(times),
                         y=[len(r.voltage_violations) for r in res_s],
                         name="Stress (heat+EV)", marker_color="#F08C3C"))
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
    for kind, results, op in [("baseline", res_b, 1.0), ("stress", res_s, 0.7)]:
        for name in reg_names:
            ys = [r.regulator_taps.get(name, None) for r in results]
            fig.add_trace(go.Scatter(
                x=list(times), y=ys, mode="lines+markers",
                name=f"{kind} · {name.upper()}",
                line=dict(width=2, color=palettes[kind], dash=line_dash.get(name, "solid")),
                marker=dict(size=4), opacity=op,
            ))
    fig.update_layout(
        title=dict(text="Regulator tap positions · QSTS", font=dict(size=15)),
        xaxis_title=None, yaxis_title="tap step",
        height=320, margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def violation_heatmap(mat_df: pd.DataFrame, dates) -> go.Figure:
    """Bus × Day heatmap, color = violation hours."""
    # Sort buses by total violation hours descending; show top 20 for readability
    totals = mat_df.sum(axis=1)
    top_buses = totals[totals > 0].sort_values(ascending=False).head(20).index.tolist()
    if not top_buses:
        # show all buses anyway
        top_buses = mat_df.index.tolist()[:20]
    sub = mat_df.loc[top_buses]
    fig = go.Figure(data=go.Heatmap(
        z=sub.values,
        x=[d.strftime("%a %b %d") for d in dates],
        y=[f"Bus {b}" for b in sub.index],
        colorscale=[[0, "#1F2630"], [0.01, "#2C3340"], [0.3, "#EBC83C"], [0.6, "#F08C3C"], [1, "#D23232"]],
        zmin=0, zmax=max(1, int(sub.values.max())),
        hovertemplate="Bus %{y}<br>%{x}<br><b>%{z}</b> violation hours<extra></extra>",
        colorbar=dict(title="hr/day", thickness=12, len=0.8),
    ))
    fig.update_layout(
        title=dict(text="Bus × Day · voltage-violation hours", font=dict(size=15)),
        height=520, margin=dict(l=10, r=10, t=50, b=20),
        xaxis=dict(side="top"),
    )
    return fig


def top_buses_bar(weekly_df: pd.DataFrame, n: int = 10) -> go.Figure:
    sub = weekly_df.head(n).copy()
    sub = sub[sub["violation_hours_week"] > 0]
    fig = go.Figure(go.Bar(
        x=sub["violation_hours_week"],
        y=[f"Bus {b}" for b in sub["bus"]],
        orientation="h",
        marker=dict(color=sub["violation_hours_week"], colorscale="Reds", showscale=False),
        text=[f"{int(v)} hr · worst {w:.3f} pu" for v, w in
              zip(sub["violation_hours_week"], sub["worst_v_pu"])],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"Top {n} stressed buses · weekly violation-hours", font=dict(size=15)),
        height=420, margin=dict(l=10, r=10, t=50, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def weekly_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    if trend_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["week_start"], y=trend_df["total_violation_hours"],
        mode="lines+markers", line=dict(width=2, color="#F08C3C"),
        name="Total violation-hours", marker=dict(size=8),
    ))
    fig.update_layout(
        title=dict(text="Multi-week trend · stress evolution across the season", font=dict(size=15)),
        xaxis_title=None, yaxis_title="violation hours / week",
        height=300, margin=dict(l=10, r=10, t=50, b=20),
    )
    return fig


# --- Scenario presets (operator) ------------------------------------------ #

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


# --- Page setup ----------------------------------------------------------- #

st.set_page_config(
    page_title="APS Feeder Intelligence",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.5rem !important; padding-bottom: 3rem !important; max-width: 1480px;}
        div[data-testid="stMetricValue"] {font-size: 2.0rem; font-weight: 600;}
        div[data-testid="stMetricLabel"] {font-size: 0.85rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.05em;}
        div[data-testid="stMetricDelta"] {font-size: 0.95rem;}
        div[data-testid="stMetric"] {
            background: rgba(60, 70, 90, 0.10);
            padding: 1rem 1.2rem;
            border-radius: 8px;
            border-left: 3px solid rgba(91, 163, 245, 0.5);
        }
        button[data-baseweb="tab"] {font-size: 1rem !important; padding: 0.75rem 1.5rem !important;}
        .role-banner {
            background: linear-gradient(135deg, rgba(91,163,245,0.12), rgba(91,163,245,0.02));
            border-left: 4px solid #5BA3F5;
            padding: 1.0rem 1.2rem;
            border-radius: 6px;
            margin: 0.5rem 0 1.2rem;
        }
        .role-banner-planner {
            background: linear-gradient(135deg, rgba(167,139,255,0.12), rgba(167,139,255,0.02));
            border-left: 4px solid #A78BFF;
            padding: 1.0rem 1.2rem;
            border-radius: 6px;
            margin: 0.5rem 0 1.2rem;
        }
        .scenario-banner {
            background: linear-gradient(135deg, rgba(255,196,0,0.10), rgba(255,196,0,0.02));
            border-left: 4px solid #FFC400;
            padding: 0.85rem 1.1rem;
            border-radius: 6px;
            margin: 0.4rem 0 1.0rem;
            font-size: 0.92rem;
        }
        .now-callout {
            background: linear-gradient(135deg, rgba(240,140,60,0.18), rgba(240,140,60,0.04));
            border-left: 4px solid #F08C3C;
            padding: 1.2rem 1.4rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .now-callout b {color: #F08C3C; font-size: 1.05rem;}
        h2, h3 {margin-top: 1.5rem !important; margin-bottom: 0.6rem !important;}
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header --------------------------------------------------------------- #

col_title, col_links = st.columns([8, 2])
with col_title:
    st.markdown("## ⚡ APS Spatio-Temporal Feeder Intelligence")
    st.caption("GraphSAGE + GRU forecaster · OpenDSS QSTS validation · operator + planner decision layers")
with col_links:
    st.markdown(
        "<div style='text-align:right; padding-top:1.4rem;'>"
        "<a href='https://github.com/spraka52/aps-feeder-intelligence' style='text-decoration:none; padding:6px 12px; "
        "background:rgba(91,163,245,0.15); border-radius:5px; font-size:0.85rem;'>📦 GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )

if not CKPT_PATH.exists() or not BASELINE_NPZ.exists():
    st.error("Missing artifacts. Run `python -m data.synthesize --multi --customers resstock` then `python -m models.train --epochs 25`.")
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

# ================== TOP: ROLE SWITCH ====================================== #
role = st.radio(
    "Role",
    ["👷 Operator · hour-by-hour", "📐 Planner · week-by-week"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")


# =============================================================================
#                              OPERATOR VIEW
# =============================================================================

def render_operator_view():
    st.markdown(
        "<div class='role-banner'>"
        "<b style='color:#5BA3F5;'>👷 OPERATOR VIEW</b> &nbsp;·&nbsp; "
        "Pick a forecast window. The model predicts the next 24 hours per bus, "
        "OpenDSS QSTS validates the physics each hour, and the Action Center ranks "
        "what to do — sized in kW, ordered by priority."
        "</div>",
        unsafe_allow_html=True,
    )

    scenarios = _summer_scenarios(times, ds_base.heatwave)
    st.markdown("##### 📅 Pick a scenario")
    scenario_keys = list(scenarios.keys()) + ["✏ Custom date / hour"]
    pick_scenario = st.radio(
        "Pick a scenario",
        scenario_keys, index=0, horizontal=True, label_visibility="collapsed",
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

    target_ts = pd.Timestamp(pick_day, tz=times.tz) + pd.Timedelta(hours=pick_hour)
    deltas = (times - target_ts).asi8
    t0_full = int(np.argmin(np.abs(deltas)))
    t0 = max(0, t0_full - forecaster.horizon_in)
    if t0 + forecaster.horizon_in + forecaster.horizon_out > len(times):
        t0 = len(times) - forecaster.horizon_in - forecaster.horizon_out
    fcst_times = times[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
    hw_in_window = int(ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out].sum())

    st.markdown(
        f"<div class='scenario-banner'><b>{scenario_label}</b> "
        f"&nbsp;·&nbsp; horizon: {fcst_times[0].strftime('%a %b %d, %H:%M')} → "
        f"{fcst_times[-1].strftime('%a %b %d, %H:%M')} &nbsp;·&nbsp; "
        f"<b>{hw_in_window}/24</b> heatwave hours in window</div>",
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
        st.warning("OpenDSS solver hit a stability error on this window — pick a different start.")

    hw_mask = ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
    actions_base = build_actions(res_base, fcst_times, fcst_base, list(bus_order), hw_mask)
    actions_stress = build_actions(res_stress, fcst_times, fcst_stress, list(bus_order), hw_mask)
    kpi_base = headline_kpis(res_base, fcst_base, hw_mask)
    kpi_stress = headline_kpis(res_stress, fcst_stress, hw_mask)

    # ----- "Right now" callout — first-hour or worst-hour action ----------- #
    df_stress_actions = ops_actions_to_df(actions_stress)
    if not df_stress_actions.empty:
        top = df_stress_actions.iloc[0]
        first_hour = pd.Timestamp(top["when"])
        hours_until = max(0, int((first_hour - fcst_times[0]).total_seconds() // 3600))
        urgency = "Take action now" if hours_until <= 1 else f"Pre-position in {hours_until} h"
        st.markdown(
            f"""
            <div class='now-callout'>
              <b>⚡ {urgency}</b> &nbsp;·&nbsp; Priority {int(top['priority'])} · {top['kind']}
              <br/>
              <span style='font-size:1.05rem;'>{top['recommendation']}</span>
              <br/>
              <span style='opacity:0.75; font-size:0.9rem;'>worst at {first_hour.strftime('%a %b %d %H:%M')} ·
              {int(top['hours_affected'])} hours affected · severity {float(top['severity']):.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='now-callout'><b>✅ No actions required in the next 24 hours.</b> "
            "Feeder is operating within voltage and thermal limits.</div>",
            unsafe_allow_html=True,
        )

    # ----- KPI strip ------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak feeder load", f"{kpi_stress['peak_forecast_kw']:.0f} kW",
              f"{kpi_stress['peak_forecast_kw'] - kpi_base['peak_forecast_kw']:+.0f} vs base",
              help="Highest single-hour total kW.")
    c2.metric("Voltage violations", kpi_stress["n_voltage_violations"],
              f"{kpi_stress['n_voltage_violations'] - kpi_base['n_voltage_violations']:+d} vs base",
              delta_color="inverse", help="(hour × bus) pairs outside [0.95, 1.05] pu.")
    c3.metric("Thermal overloads", kpi_stress["n_thermal_overloads"],
              f"{kpi_stress['n_thermal_overloads'] - kpi_base['n_thermal_overloads']:+d} vs base",
              delta_color="inverse")
    c4.metric("Peak losses", f"{kpi_stress['peak_loss_kw']:.0f} kW",
              f"{kpi_stress['peak_loss_kw'] - kpi_base['peak_loss_kw']:+.0f} vs base",
              delta_color="inverse")

    st.divider()

    # ----- Tabs: Map | Forecast | Action timeline | Action Center ---------- #
    tab_map, tab_forecast, tab_timeline, tab_actions = st.tabs([
        "🗺  Operations Map", "📈  Forecast & Physics",
        "⏱  Hourly Action Timeline", "⚙  Action Center",
    ])

    with tab_map:
        fcst_hour_options = [t.strftime("%a %b %d · %H:%M") for t in fcst_times]
        default_hour_idx = 18 if len(fcst_hour_options) > 18 else len(fcst_hour_options) // 2

        cs1, cs2 = st.columns([3, 1])
        with cs1:
            selected_hour_label = st.select_slider(
                "Hour into the forecast",
                options=fcst_hour_options,
                value=fcst_hour_options[default_hour_idx],
            )
        with cs2:
            compare_view = st.toggle("Show baseline beside stress", value=False)
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
                    feeder_map_deck(voltages_per_hour_base, map_hour, ops_actions_to_df(actions_base)),
                    height=520,
                )
            with mc2:
                st.markdown(
                    f"**Stress (heat + EV)** · worst V at this hour: "
                    f"<span style='color:{'#D23232' if worst_v_stress < 0.95 else '#50C350'};'><b>{worst_v_stress:.3f} pu</b></span>",
                    unsafe_allow_html=True,
                )
                st.pydeck_chart(
                    feeder_map_deck(voltages_per_hour_stress, map_hour, ops_actions_to_df(actions_stress)),
                    height=520,
                )
        else:
            st.markdown(
                f"**Stress scenario (heat + 35% EV)** · worst V at this hour: "
                f"<span style='color:{'#D23232' if worst_v_stress < 0.95 else '#50C350'};'><b>{worst_v_stress:.3f} pu</b></span>"
                f" &nbsp;·&nbsp; <span style='opacity:0.7;'>baseline worst: {worst_v_base:.3f} pu</span>",
                unsafe_allow_html=True,
            )
            st.pydeck_chart(
                feeder_map_deck(voltages_per_hour_stress, map_hour, ops_actions_to_df(actions_stress)),
                height=620,
            )

    with tab_forecast:
        st.plotly_chart(horizon_chart(fcst_base, fcst_stress, fcst_times), width="stretch")
        st.divider()
        cf1, cf2 = st.columns(2)
        with cf1:
            st.plotly_chart(violations_chart(res_base, res_stress, fcst_times), width="stretch")
        with cf2:
            st.plotly_chart(reg_tap_chart(res_base, res_stress, fcst_times), width="stretch")

    with tab_timeline:
        st.caption("One row per forecast hour. Each row shows whether OpenDSS flagged a problem at that hour and what to do.")
        # Build hour-by-hour timeline DataFrame
        rows = []
        actions_df = ops_actions_to_df(actions_stress)
        for h, t in enumerate(fcst_times):
            r = res_stress[h]
            n_v = len(r.voltage_violations)
            n_t = len(r.thermal_overloads)
            worst_pu = min(r.bus_voltage_pu.values()) if r.bus_voltage_pu else float("nan")
            # Find any action whose 'when' falls in this hour or earlier (but not yet executed)
            hour_action = "—"
            if not actions_df.empty:
                same_hour = actions_df[pd.to_datetime(actions_df["when"]) == pd.Timestamp(t)]
                if not same_hour.empty:
                    hour_action = same_hour.iloc[0]["recommendation"]
            status = "🟢 clean" if (n_v == 0 and n_t == 0) else (
                f"🔴 {n_v} V · {n_t} thermal" if n_v + n_t > 3 else f"🟡 {n_v} V · {n_t} thermal")
            rows.append({
                "Hour": t.strftime("%a %b %d · %H:%M"),
                "Status": status,
                "Worst V (pu)": f"{worst_pu:.3f}",
                "Total kW": f"{fcst_stress[h].sum():.0f}",
                "Recommended action": hour_action,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=720)

    with tab_actions:
        show_n = st.slider("Top N actions", 3, 15, 8)
        sub_stress, sub_base = st.tabs(["🔥 Stress scenario", "🌤 Baseline scenario"])

        def _render(actions_list, label):
            df = ops_actions_to_df(actions_list).head(show_n)
            if df.empty:
                st.success(f"No violations in the {label} forecast — feeder is within limits.")
                return
            df_disp = df.rename(columns={
                "priority": "Pri.", "kind": "Kind", "bus_or_line": "Where",
                "when": "When (worst hour)", "hours_affected": "Hrs",
                "severity": "Severity", "target_kw": "Sized kW",
                "detail": "What happened", "recommendation": "Recommended action",
            })
            st.dataframe(
                df_disp[["Pri.", "Kind", "Where", "When (worst hour)", "Hrs",
                         "Severity", "Sized kW", "What happened", "Recommended action"]],
                width="stretch", hide_index=True, height=380,
            )

        with sub_stress:
            _render(actions_stress, "stress")
        with sub_base:
            _render(actions_base, "baseline")


# =============================================================================
#                              PLANNER VIEW
# =============================================================================

def _build_summer_weeks(times: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return a list of (week_start, week_end) Timestamps for full weeks within the dataset."""
    weeks = []
    start = times.min().normalize()
    while start + pd.Timedelta(days=7) <= times.max():
        weeks.append((start, start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)))
        start = start + pd.Timedelta(days=7)
    return weeks


def render_planner_view():
    st.markdown(
        "<div class='role-banner-planner'>"
        "<b style='color:#A78BFF;'>📐 PLANNER VIEW</b> &nbsp;·&nbsp; "
        "Pick a week (or compare two). We aggregate the OpenDSS QSTS solve across the whole "
        "week, identify chronic stress patterns, and propose ranked capital projects "
        "— sized in kW with rough cost + payback estimates."
        "</div>",
        unsafe_allow_html=True,
    )

    weeks = _build_summer_weeks(times)
    if not weeks:
        st.error("Dataset doesn't contain any full weeks.")
        return

    week_labels = [f"Week of {ws.strftime('%a %b %d, %Y')}" for ws, _ in weeks]
    cw1, cw2, cw3 = st.columns([3, 2, 2])
    with cw1:
        pick_week_label = st.selectbox("Week to analyse", week_labels, index=min(2, len(week_labels) - 1))
    with cw2:
        scenario_for_planner = st.radio(
            "Scenario", ["🌤 Baseline", "🔥 Stress (heat + EV)"],
            index=1, horizontal=True,
        )
    with cw3:
        view_mode = st.radio("Aggregate by", ["Per-day", "Per-hour-of-week"],
                             index=0, horizontal=True)

    week_idx = week_labels.index(pick_week_label)
    ws, we = weeks[week_idx]

    # Pick the dataset for this scenario
    ds_for = ds_stress if "Stress" in scenario_for_planner else ds_base

    # Slice the synthesized loads for this week (use the synthesized "ground
    # truth" loads_kw rather than the model forecast — for a planner view, we
    # want a clean unbiased view of how the feeder would perform under the
    # week's actual demand profile).
    week_mask = (ds_for.times >= ws) & (ds_for.times <= we)
    week_times = ds_for.times[week_mask]
    week_loads = ds_for.loads[:, week_mask]  # [N, T]

    # Reshape to per-day [days, 24, N]
    n_days = len(week_times) // 24
    if n_days < 1:
        st.error("Selected week has fewer than 24 hours of data.")
        return
    day_kw = week_loads[:, : n_days * 24].T.reshape(n_days, 24, -1).astype(np.float32)
    bus_order = tuple(ds_for.bus_order)

    # Solve per-day OpenDSS QSTS (cached)
    week_label_key = f"{ws.isoformat()}_{scenario_for_planner}"
    per_day_dicts = _solve_week_truth(
        day_kw.astype(np.float32).tobytes(),
        bus_order,
        day_kw.shape,
        week_label_key,
    )
    per_day_results = [_to_hour_results(d) for d in per_day_dicts]

    # Aggregations
    weekly_df = aggregate_weekly_violations(per_day_results, list(bus_order))
    hours_matrix = _bus_day_hours_matrix(per_day_results, list(bus_order))
    nominal_map = {b: SPOT_LOADS_KW.get(b, 0.0) for b in bus_order}
    plan_actions = build_planner_actions(weekly_df, nominal_map)

    # Multi-week trend (compute lightweight stats for ALL weeks of the chosen scenario)
    trend_rows = []
    for w_start, w_end in weeks:
        w_mask = (ds_for.times >= w_start) & (ds_for.times <= w_end)
        if w_mask.sum() < 24:
            continue
        # Approximate weekly stress: count "heatwave hours" + sum of synthetic feeder kW peaks
        wk_temps = ds_for.temp[w_mask]
        wk_loads = ds_for.loads[:, w_mask].sum(axis=0)  # feeder total per hour
        trend_rows.append({
            "week_start": w_start,
            "total_violation_hours": int((wk_temps >= 41).sum()),  # proxy = heat-stress hours
            "peak_kw": float(wk_loads.max()) if wk_loads.size else 0.0,
        })
    trend_df = pd.DataFrame(trend_rows)

    # Replace the proxy "total_violation_hours" for the *currently selected week* with the
    # actual computed value, so the trend chart's selected point reflects real physics.
    actual_total = int(weekly_df["violation_hours_week"].sum())
    if not trend_df.empty:
        sel_row = trend_df[trend_df["week_start"] == ws]
        if not sel_row.empty:
            trend_df.loc[sel_row.index, "total_violation_hours"] = actual_total

    # ----- Weekly KPIs ----------------------------------------------------- #
    total_v = int(weekly_df["violation_hours_week"].sum())
    worst_bus = weekly_df.iloc[0] if not weekly_df.empty else None
    n_buses_hit = int((weekly_df["violation_hours_week"] > 0).sum())
    peak_week_kw = float(week_loads.sum(axis=0).max()) if week_loads.size else 0.0

    st.markdown(f"### Week of {ws.strftime('%a %b %d, %Y')} · {scenario_for_planner}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total voltage violations", f"{total_v}",
              help="Sum of (hour × bus) violations across all 7 days.")
    if worst_bus is not None and worst_bus["violation_hours_week"] > 0:
        k2.metric("Worst bus", f"Bus {worst_bus['bus']}",
                  f"{int(worst_bus['violation_hours_week'])} hr · worst {worst_bus['worst_v_pu']:.3f} pu")
    else:
        k2.metric("Worst bus", "—", "no violations")
    k3.metric("Buses with violations", f"{n_buses_hit} / {len(bus_order)}")
    k4.metric("Peak weekly load", f"{peak_week_kw:.0f} kW",
              help="Highest single-hour feeder-total kW across the week.")

    st.divider()

    # ----- Tabs: Heatmap | Top buses | Trend | Capital actions ----------- #
    tab_hm, tab_top, tab_trend, tab_capex = st.tabs([
        "🔲  Bus × Day Heatmap", "📊  Top Stressed Buses",
        "📈  Multi-Week Trend", "🏗  Capital Action Plan",
    ])

    with tab_hm:
        if view_mode == "Per-day":
            day_dates = pd.date_range(ws, periods=n_days, freq="D")
            st.plotly_chart(violation_heatmap(hours_matrix, day_dates), width="stretch")
        else:
            # Per-hour-of-week: 168 columns (7 × 24)
            big = np.zeros((len(bus_order), n_days * 24), dtype=int)
            bus_idx = {b: i for i, b in enumerate(bus_order)}
            for d, day_results in enumerate(per_day_results):
                for h, hr in enumerate(day_results):
                    if not hr.converged:
                        continue
                    for b, v in hr.bus_voltage_pu.items():
                        if v is None:
                            continue
                        if v < 0.95 or v > 1.05:
                            big[bus_idx.get(b, 0), d * 24 + h] += 1
            big_df = pd.DataFrame(big, index=list(bus_order),
                                  columns=[f"D{d+1}H{h:02d}" for d in range(n_days) for h in range(24)])
            big_dates = pd.date_range(ws, periods=n_days * 24, freq="h")
            st.plotly_chart(violation_heatmap(big_df, big_dates), width="stretch")

    with tab_top:
        st.plotly_chart(top_buses_bar(weekly_df, n=10), width="stretch")
        with st.expander("Per-bus weekly stats (full table)"):
            st.dataframe(
                weekly_df.rename(columns={
                    "bus": "Bus", "violation_hours_week": "Violation hr/wk",
                    "worst_v_pu": "Worst V (pu)", "days_with_violation": "Days affected",
                }),
                width="stretch", hide_index=True, height=420,
            )

    with tab_trend:
        st.plotly_chart(weekly_trend_chart(trend_df), width="stretch")
        st.caption(
            "Bars show weekly stress (heat-stress hours per week as a proxy for "
            "the full season; the selected week shows the actual computed violation "
            "count from the OpenDSS solve). The seasonal arc tells you whether the "
            "feeder is degrading earlier each year, or whether DER additions are "
            "starting to pay off."
        )

    with tab_capex:
        st.caption(
            "Capital projects ranked by annualised violation hours × severity. "
            "Cost and payback are order-of-magnitude estimates: $1,500/kW for "
            "battery; avoided customer-minute valued at $0.15/min × 20 customers/bus."
        )
        if not plan_actions:
            st.success("No capital projects warranted this week — feeder is operating within limits.")
        else:
            df_actions = planner_actions_to_df(plan_actions)
            df_disp = df_actions.rename(columns={
                "priority": "Pri.",
                "bus": "Bus",
                "kind": "Project type",
                "suggested_kw": "Size (kW)",
                "est_cost_usd": "Cost (USD)",
                "violation_hours_per_week": "Hr/wk now",
                "violation_hours_per_year": "Hr/yr (annualized)",
                "worst_v_pu": "Worst V",
                "n_days_with_violation": "Days affected",
                "rationale": "Why this project",
                "payback_years": "Payback (yr)",
            })
            # Format cost / payback nicely
            df_disp["Cost (USD)"] = df_disp["Cost (USD)"].map(lambda x: f"${x:,.0f}")
            df_disp["Hr/yr (annualized)"] = df_disp["Hr/yr (annualized)"].map(lambda x: f"{x:.0f}")
            df_disp["Size (kW)"] = df_disp["Size (kW)"].map(lambda x: f"{x:.0f}")
            df_disp["Worst V"] = df_disp["Worst V"].map(lambda x: f"{x:.3f} pu")
            df_disp["Payback (yr)"] = df_disp["Payback (yr)"].map(
                lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
            st.dataframe(
                df_disp[["Pri.", "Bus", "Project type", "Size (kW)", "Cost (USD)",
                         "Hr/wk now", "Hr/yr (annualized)", "Worst V",
                         "Days affected", "Payback (yr)", "Why this project"]],
                width="stretch", hide_index=True, height=420,
            )


# ================== Render selected role ================================== #

if role.startswith("👷"):
    render_operator_view()
else:
    render_planner_view()


# ---------------- Footer --------------------------------------------------- #

st.divider()
st.caption("Built for the APS / ASU AI for Energy hackathon. "
           "[Source on GitHub](https://github.com/spraka52/aps-feeder-intelligence) · "
           "real NOAA + NREL NSRDB + NREL ResStock/ComStock data · "
           "OpenDSS QSTS physics validation · GraphSAGE+GRU forecaster.")
