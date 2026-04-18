"""APS Spatio-Temporal Feeder Intelligence — Streamlit utility dashboard.

End-to-end demo:
  1. Load the IEEE 34-bus topology + a trained GraphSAGE+GRU checkpoint.
  2. Pick a forecast window from the synthesized time-series.
  3. Compare a baseline scenario against a stress scenario (heatwave + EV peak).
  4. Run OpenDSS on each forecast hour to detect voltage / thermal violations.
  5. Render the spatio-temporal map and an Action Center for operators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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

@st.cache_resource(show_spinner="Loading model checkpoint…")
def _get_forecaster():
    return Forecaster.load(CKPT_PATH)


@st.cache_data(show_spinner="Loading baseline dataset…")
def _get_dataset(npz_path_str: str):
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


# --- UI helpers ------------------------------------------------------------- #

def _v_to_color(v: float) -> List[int]:
    """Map a voltage in pu to an RGBA color (red→amber→green→amber→red).

    0.93 → deep red, 0.95 → amber (lower limit), 1.00 → green, 1.05 → amber,
    1.07 → deep red. Anything outside [0.93, 1.07] is clamped.
    """
    if v is None or np.isnan(v):
        return [128, 128, 128, 200]
    # Distance from nominal scaled so 0.05 pu = full saturation
    dist = abs(v - 1.00) / 0.05
    dist = max(0.0, min(dist, 1.5))
    if dist <= 0.5:
        # Healthy band: green → amber
        t = dist / 0.5
        r = int(80 + (235 - 80) * t)
        g = int(195 + (200 - 195) * t)
        b = int(80 + (60 - 80) * t)
    else:
        # Out of band: amber → red
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
    """Geographic map of the feeder using PyDeck + a Carto basemap.

    Each bus is a circle whose colour encodes its voltage at the selected
    hour and whose radius scales with its nominal load. Lines are drawn as
    grey paths between connected buses. Buses targeted by the decision
    engine get a yellow outer halo + a label, so the operator's eye lands
    on them immediately.
    """
    fg = build_graph()
    voltages = bus_voltages_per_hour[min(hour_idx, len(bus_voltages_per_hour) - 1)]

    flagged = set()
    if not actions_df.empty:
        flagged = set(actions_df["bus_or_line"].astype(str).tolist())

    nodes = []
    halos = []
    labels = []
    for b in fg.g.nodes():
        if b not in COORDS:
            continue
        lat, lon = COORDS[b]
        v = voltages.get(b)
        nominal_kw = SPOT_LOADS_KW.get(b, 0.0)
        radius = 35 + 0.55 * float(nominal_kw)  # meters
        nodes.append({
            "bus": b,
            "lat": lat,
            "lon": lon,
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
                "color": [255, 196, 0, 200],  # APS gold
            })
        if b in flagged or nominal_kw >= 100:
            labels.append({
                "lat": lat,
                "lon": lon,
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

    # Center on the centroid; fit zoom to span
    lats = [c[0] for c in COORDS.values()]
    lons = [c[1] for c in COORDS.values()]
    view = pdk.ViewState(
        latitude=(min(lats) + max(lats)) / 2,
        longitude=(min(lons) + max(lons)) / 2,
        zoom=11.6, pitch=35, bearing=0,
    )

    layers = [
        pdk.Layer(
            "PathLayer", data=edges, get_path="path",
            get_color="color", get_width="width",
            width_min_pixels=2, width_max_pixels=6,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=halos,
            get_position=["lon", "lat"], get_radius="radius",
            get_fill_color="color", stroked=False, opacity=0.55,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=nodes,
            get_position=["lon", "lat"], get_radius="radius",
            get_fill_color="color", stroked=True,
            get_line_color=[20, 25, 30, 240], line_width_min_pixels=1,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer", data=labels,
            get_position=["lon", "lat"],
            get_text="text", get_size=14,
            get_color=[235, 235, 240, 255],
            get_alignment_baseline="'bottom'",
            get_text_anchor="'middle'",
            background=True,
            background_padding=[3, 1],
            get_background_color=[15, 22, 30, 200],
        ),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="dark",  # uses Carto dark basemap (no Mapbox token)
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
    """Compact text legend explaining the colour ramp."""
    return st.markdown(
        """
        <div style="display:flex; gap:20px; align-items:center; font-size:12px; opacity:0.85;">
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#50C350;display:inline-block;"></span>~1.00 pu (healthy)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#EBC83C;display:inline-block;"></span>0.95 / 1.05 pu (limit)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#D23232;display:inline-block;"></span>≤0.93 / ≥1.07 pu (violation)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:14px;height:14px;border-radius:50%;background:#FFC400;opacity:0.8;display:inline-block;"></span>action target (halo)</span>
          <span style="display:flex;align-items:center;gap:6px;"><span style="width:18px;height:4px;background:#B43232;display:inline-block;"></span>transformer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def horizon_chart(forecast_kw: np.ndarray, times, label: str):
    feeder_total = forecast_kw.sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times), y=feeder_total, mode="lines+markers",
        name=label, line=dict(width=2),
    ))
    fig.update_layout(
        title=f"{label}: feeder-total kW (24-hour forecast)",
        xaxis_title="time", yaxis_title="kW",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def violations_chart(results, times, label: str):
    n_v = [len(r.voltage_violations) for r in results]
    n_t = [len(r.thermal_overloads) for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(times), y=n_v, name="V violations"))
    fig.add_trace(go.Bar(x=list(times), y=n_t, name="Thermal overloads"))
    fig.update_layout(
        title=f"{label}: per-hour OpenDSS violations",
        barmode="stack", height=260, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def reg_tap_chart(results, times, label: str):
    """Show regulator tap positions across the QSTS horizon."""
    if not results:
        return go.Figure()
    reg_names = sorted({k for r in results for k in r.regulator_taps.keys()})
    fig = go.Figure()
    for name in reg_names:
        ys = [r.regulator_taps.get(name, None) for r in results]
        fig.add_trace(go.Scatter(
            x=list(times), y=ys, mode="lines+markers",
            name=name.upper(), line=dict(width=2),
        ))
    fig.update_layout(
        title=f"{label}: regulator tap positions (QSTS)",
        xaxis_title="time", yaxis_title="tap step (+ boost / – buck)",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    return fig


# --- App body --------------------------------------------------------------- #

st.set_page_config(page_title="APS Feeder Intelligence", layout="wide", page_icon="⚡")
st.title("⚡ APS Spatio-Temporal Feeder Intelligence")
st.caption(
    "GraphSAGE + GRU forecaster on the IEEE 34-bus feeder, validated by OpenDSS, "
    "with a utility-facing decision layer."
)

if not CKPT_PATH.exists() or not BASELINE_NPZ.exists():
    st.error(
        "Missing artifacts. Run the pipeline first:\n\n"
        "    python -m data.synthesize --days 92\n"
        "    python -m models.train --epochs 12"
    )
    st.stop()

forecaster = _get_forecaster()
ds_base = _get_dataset(str(BASELINE_NPZ))
ds_stress = _get_dataset(str(STRESS_NPZ))

with st.sidebar:
    st.header("Forecast window")
    # Pick a date — default to a heatwave day for demo flair
    times = ds_base.times
    hw_days = sorted({t.date() for t, hw in zip(times, ds_base.heatwave) if hw})
    all_days = sorted({t.date() for t in times})
    default_day = hw_days[len(hw_days) // 2] if hw_days else all_days[0]
    pick_day = st.date_input("Forecast start date", value=default_day, min_value=all_days[24], max_value=all_days[-2])
    pick_hour = st.slider("Forecast start hour", 0, 23, 6)
    show_action_count = st.slider("Top N actions to display", 3, 20, 8)
    st.markdown("---")
    st.markdown("**Model**")
    st.code(
        f"buses: {len(forecaster.bus_order)}\n"
        f"horizon_in: {forecaster.horizon_in}h\n"
        f"horizon_out: {forecaster.horizon_out}h\n"
        f"params: {forecaster.model.num_parameters():,}"
    )

# Find dataset index for the chosen window start
target_ts = pd.Timestamp(pick_day, tz=times.tz) + pd.Timedelta(hours=pick_hour)
deltas = (times - target_ts).asi8
t0_full = int(np.argmin(np.abs(deltas)))
t0 = max(0, t0_full - forecaster.horizon_in)
if t0 + forecaster.horizon_in + forecaster.horizon_out > len(times):
    t0 = len(times) - forecaster.horizon_in - forecaster.horizon_out
fcst_times = times[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]

st.markdown(
    f"**Forecast horizon:** {fcst_times[0]} → {fcst_times[-1]}  •  "
    f"in-window heatwave hours: {int(ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out].sum())}"
)

# Generate forecasts
fcst_base = forecaster.forecast_window(ds_base, t0)        # [T, N]
fcst_stress = forecaster.forecast_window(ds_stress, t0)
bus_order = tuple(forecaster.bus_order)

# Run OpenDSS (cached on the forecast bytes). The runner is subprocess-isolated
# so a SIGILL inside libdss_capi never takes the Streamlit server with it.
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

# Headline KPIs
st.subheader("Operator KPIs — baseline vs. stressed (heatwave + 35% EV evening growth)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak feeder load (kW)", f"{kpi_stress['peak_forecast_kw']:.0f}",
          f"{kpi_stress['peak_forecast_kw'] - kpi_base['peak_forecast_kw']:+.0f} vs base")
c2.metric("Voltage violations (hr-buses)", kpi_stress["n_voltage_violations"],
          f"{kpi_stress['n_voltage_violations'] - kpi_base['n_voltage_violations']:+d} vs base")
c3.metric("Thermal overloads", kpi_stress["n_thermal_overloads"],
          f"{kpi_stress['n_thermal_overloads'] - kpi_base['n_thermal_overloads']:+d} vs base")
c4.metric("Peak losses (kW)", f"{kpi_stress['peak_loss_kw']:.0f}",
          f"{kpi_stress['peak_loss_kw'] - kpi_base['peak_loss_kw']:+.0f} vs base")

# Spatial maps
st.subheader("Spatio-temporal feeder map")
map_hour = st.slider(
    "Hour into the 24-hour forecast", 0, len(fcst_times) - 1, 18,
    help="Slide to watch voltages evolve hour by hour. Yellow halo = decision-engine action target.",
)
voltage_legend()
mc1, mc2 = st.columns(2)
voltages_per_hour_base = [r.bus_voltage_pu for r in res_base]
voltages_per_hour_stress = [r.bus_voltage_pu for r in res_stress]
with mc1:
    st.markdown(f"**Baseline · {fcst_times[map_hour]}**")
    st.pydeck_chart(
        feeder_map(voltages_per_hour_base, map_hour, actions_to_df(actions_base),
                   f"Baseline @ {fcst_times[map_hour]}"),
        height=480,
    )
with mc2:
    st.markdown(f"**Stress (heat + EV) · {fcst_times[map_hour]}**")
    st.pydeck_chart(
        feeder_map(voltages_per_hour_stress, map_hour, actions_to_df(actions_stress),
                   f"Stress @ {fcst_times[map_hour]}"),
        height=480,
    )

# Time-series comparisons
st.subheader("Forecast horizon — feeder load and violation count")
hc1, hc2 = st.columns(2)
with hc1:
    st.plotly_chart(horizon_chart(fcst_base, fcst_times, "Baseline"), width="stretch")
    st.plotly_chart(violations_chart(res_base, fcst_times, "Baseline"), width="stretch")
    st.plotly_chart(reg_tap_chart(res_base, fcst_times, "Baseline"), width="stretch")
with hc2:
    st.plotly_chart(horizon_chart(fcst_stress, fcst_times, "Stress (heat+EV)"), width="stretch")
    st.plotly_chart(violations_chart(res_stress, fcst_times, "Stress (heat+EV)"), width="stretch")
    st.plotly_chart(reg_tap_chart(res_stress, fcst_times, "Stress (heat+EV)"), width="stretch")

# Action Center
st.subheader("⚙ Action Center — prioritized utility interventions")
tab_stress, tab_base = st.tabs(["Stress scenario", "Baseline scenario"])
with tab_stress:
    df = actions_to_df(actions_stress).head(show_action_count)
    if df.empty:
        st.success("No violations detected in the forecast horizon — feeder is operating within limits.")
    else:
        st.dataframe(
            df[["priority", "kind", "bus_or_line", "when", "hours_affected",
                "severity", "target_kw", "detail", "recommendation"]],
            width="stretch", hide_index=True,
        )
with tab_base:
    df = actions_to_df(actions_base).head(show_action_count)
    if df.empty:
        st.success("No violations detected in the baseline forecast — feeder is operating within limits.")
    else:
        st.dataframe(
            df[["priority", "kind", "bus_or_line", "when", "hours_affected",
                "severity", "target_kw", "detail", "recommendation"]],
            width="stretch", hide_index=True,
        )

# Model diagnostics
with st.expander("Model performance (held-out validation)"):
    import json
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
            f"parameters; checkpoint at {report['checkpoint']}."
        )
    else:
        st.info("Training report not found — run `python -m models.train` to generate one.")

with st.expander("Why this matters"):
    st.markdown(
        """
        - **Spatio-temporal:** the GraphSAGE layers mix information across the 34-bus
          feeder while the GRU captures the diurnal evolution. The model isn't a label
          on temperature — temperature, irradiance, and EV-growth all enter as drivers.
        - **Physics-validated:** every forecast hour is solved with OpenDSS so the
          dashboard never claims a constraint that doesn't actually exist in the
          power-flow solution.
        - **Decision-ready:** the Action Center groups violations by location, scores
          severity by how far out of bounds and how persistent, and proposes a sized
          intervention (battery dispatch, Volt-VAR, deferrable-load shed) — the kind of
          punch list a control-room operator could act on.
        """
    )
