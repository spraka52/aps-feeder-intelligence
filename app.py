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
import streamlit.components.v1 as components

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


# -----------------------------------------------------------------------------
# Light, utility-control-center palette.
# Inspired by SCADA / EMS dashboards: clean white background, slate text,
# narrow accent palette, data takes precedence over chrome.
# -----------------------------------------------------------------------------
COLOR = {
    "accent":     "#C77F00",   # APS gold (deeper for light-bg contrast)
    "accent_dim": "#A26800",
    "baseline":   "#2F66A3",   # deeper blue for light bg
    "stress":     "#B85525",   # deeper orange
    "ok":         "#1F8252",   # deeper green
    "warn":       "#B27A1A",   # amber
    "alert":      "#9C2828",   # deeper red
    "neutral":    "#5A6270",
    "text":       "#1A1F2B",   # near-black
    "text_dim":   "#5A6270",   # medium slate
    "bg_card":    "#F4F6F8",   # very light slate
    "bg_panel":   "#FFFFFF",
    "border":     "rgba(20, 30, 50, 0.10)",
    "grid":       "rgba(20, 30, 50, 0.07)",
}


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


@st.cache_data(show_spinner="Computing weekly aggregate…")
def _solve_week_truth(week_kw_bytes: bytes, bus_order: tuple, day_shape: tuple,
                      week_label: str) -> List[List[dict]]:
    from physics.opendss_runner import _hourresult_to_dict
    week_kw = np.frombuffer(week_kw_bytes, dtype=np.float32).reshape(day_shape)
    out: List[List[dict]] = []
    for d in range(week_kw.shape[0]):
        day_loads = week_kw[d]
        day_results = run_forecast_horizon(day_loads, list(bus_order))
        out.append([_hourresult_to_dict(r) for r in day_results])
    return out


# --- Map ----------------------------------------------------------------- #

def _v_to_color(v: Optional[float]) -> List[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [136, 144, 158, 200]
    dist = abs(v - 1.00) / 0.05
    dist = max(0.0, min(dist, 1.5))
    if dist <= 0.5:
        t = dist / 0.5
        r = int(63 + (212 - 63) * t)
        g = int(166 + (160 - 166) * t)
        b = int(110 + (58 - 110) * t)
    else:
        t = min((dist - 0.5) / 1.0, 1.0)
        r = int(212 + (184 - 212) * t)
        g = int(160 + (56 - 160) * t)
        b = int(58 + (56 - 58) * t)
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
                "color": [232, 163, 23, 200],   # APS gold
            })
        if b in flagged or nominal_kw >= 100:
            labels.append({
                "lat": lat, "lon": lon,
                "text": f"BUS {b}" + ("  •" if b in flagged else ""),
            })

    edges = []
    for u, v, data in fg.g.edges(data=True):
        if u not in COORDS or v not in COORDS:
            continue
        edges.append({
            "kind": data.get("kind", "line"),
            "color": [184, 56, 56, 220] if data.get("kind") == "transformer" else [136, 144, 158, 200],
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
                  get_line_color=[40, 50, 65, 240], line_width_min_pixels=1,
                  pickable=True),
        pdk.Layer("TextLayer", data=labels,
                  get_position=["lon", "lat"],
                  get_text="text", get_size=12,
                  get_color=[26, 31, 43, 255],
                  get_alignment_baseline="'bottom'",
                  get_text_anchor="'middle'",
                  background=True,
                  background_padding=[4, 2],
                  get_background_color=[255, 255, 255, 230]),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="light",
        tooltip={
            "html": "<b>Bus {bus}</b><br/>Voltage: <b>{v_label}</b><br/>Nominal load: {nominal_kw:.0f} kW",
            "style": {"backgroundColor": "#FFFFFF", "color": "#1A1F2B",
                      "fontSize": "12px", "border": "1px solid #D8DCE2",
                      "borderRadius": "3px", "padding": "6px 10px",
                      "boxShadow": "0 2px 8px rgba(0,0,0,0.10)"},
        },
    )


def voltage_legend_chip():
    st.markdown(
        f"""
        <div style="display:flex; gap:24px; align-items:center; font-size:11px;
                    color:{COLOR['text_dim']}; flex-wrap:wrap; margin: 6px 0 12px;
                    letter-spacing:0.04em; text-transform:uppercase;">
          <span style="display:flex;align-items:center;gap:6px;">
            <span style="width:11px;height:11px;border-radius:50%;background:{COLOR['ok']};display:inline-block;"></span>
            Within band (~1.00 pu)</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span style="width:11px;height:11px;border-radius:50%;background:{COLOR['warn']};display:inline-block;"></span>
            At limit (0.95 / 1.05 pu)</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span style="width:11px;height:11px;border-radius:50%;background:{COLOR['alert']};display:inline-block;"></span>
            Out of band (≤0.93 / ≥1.07)</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span style="width:11px;height:11px;border-radius:50%;background:{COLOR['accent']};opacity:0.85;display:inline-block;"></span>
            Action target</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span style="width:18px;height:3px;background:{COLOR['alert']};display:inline-block;"></span>
            Transformer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, kind: str = "ok") -> str:
    """Inline HTML pill — replaces emoji status indicators."""
    bg = {"ok": COLOR["ok"], "warn": COLOR["warn"], "alert": COLOR["alert"],
          "neutral": COLOR["neutral"]}.get(kind, COLOR["neutral"])
    return (
        f"<span style='display:inline-block; padding:2px 9px; border-radius:10px; "
        f"background:{bg}; color:#FFFFFF; font-size:11px; font-weight:600; "
        f"letter-spacing:0.04em; text-transform:uppercase;'>{label}</span>"
    )


# Streamlit's st.dataframe doesn't expose a reliable horizontal-scroll knob —
# wide tables get squeezed and long descriptions get truncated mid-cell. Render
# as a real HTML <table> wrapped in an overflow:auto div, sandboxed inside a
# components.v1.html iframe so the markdown sanitiser doesn't strip styling.
def scrollable_table(df: pd.DataFrame, max_height: int = 460):
    if df.empty:
        return
    df_safe = df.copy()
    for col in df_safe.columns:
        df_safe[col] = df_safe[col].astype(str)
    table_html = df_safe.to_html(index=False, classes="aps-table",
                                 border=0, escape=True)
    table_html = table_html.replace("<thead>", "<thead class='aps-thead'>")
    full_html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          html, body {{
            margin: 0; padding: 0;
            background: #FFFFFF;
            color: {COLOR['text']};
            font-family: -apple-system, system-ui, "Inter", Helvetica, Arial, sans-serif;
          }}
          .aps-scroll {{
            overflow-x: auto; overflow-y: auto;
            max-height: {max_height}px;
            border: 1px solid {COLOR['border']};
            border-radius: 4px;
            background: #FFFFFF;
          }}
          .aps-scroll::-webkit-scrollbar {{ height: 10px; width: 10px; }}
          .aps-scroll::-webkit-scrollbar-thumb {{
            background: rgba(20, 30, 50, 0.20); border-radius: 5px;
          }}
          .aps-scroll::-webkit-scrollbar-track {{ background: transparent; }}
          .aps-table {{
            width: max-content; min-width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            font-feature-settings: "tnum" 1;
            color: {COLOR['text']};
          }}
          .aps-table th, .aps-table td {{
            padding: 9px 14px;
            border-bottom: 1px solid rgba(20, 30, 50, 0.08);
            text-align: left;
            white-space: nowrap;
            vertical-align: top;
          }}
          .aps-thead th {{
            background: #F4F6F8;
            color: {COLOR['text_dim']};
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 11px;
            position: sticky; top: 0; z-index: 2;
            border-bottom: 1px solid {COLOR['border']};
          }}
          .aps-table tbody tr:hover td {{ background: #FAFBFC; }}
          .aps-table tbody tr:nth-child(even) td {{ background: #FCFCFD; }}
        </style>
      </head>
      <body>
        <div class="aps-scroll">{table_html}</div>
      </body>
    </html>
    """
    components.html(full_html, height=max_height + 12, scrolling=False)


def column_picker(
    df: pd.DataFrame,
    key: str,
    default_cols: Optional[List[str]] = None,
    essential_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter-icon popover with a stacked list of checkboxes — one per column.

    - Trigger button is a single Material `filter_list` icon, no text noise.
    - Each optional column is its own checkbox on its own row.
    - `essential_cols` are always present (and shown above the checkboxes as
      a static line) even if the user un-ticks everything.
    - A "Reset" button restores the default selection.
    """
    all_cols = list(df.columns)
    if default_cols is None:
        default_cols = all_cols
    defaults = [c for c in default_cols if c in all_cols]
    essential = [c for c in (essential_cols or []) if c in all_cols]
    optional = [c for c in all_cols if c not in essential]

    # Lazy-init each per-checkbox state from defaults the first time we render.
    for col in optional:
        ck = f"colchk_{key}__{col}"
        if ck not in st.session_state:
            st.session_state[ck] = col in defaults

    chosen_now = [c for c in optional if st.session_state.get(f"colchk_{key}__{c}", False)]
    visible_count = len(chosen_now) + len(essential)

    # Material symbol — the trigger label.
    trigger_label = f":material/filter_list: Columns · {visible_count}/{len(all_cols)}"

    with st.popover(trigger_label, use_container_width=False):
        rcol1, rcol2 = st.columns([3, 2])
        with rcol1:
            st.markdown(
                f"<div style='font-size:0.78rem;color:{COLOR['text_dim']};"
                f"text-transform:uppercase;letter-spacing:0.06em;font-weight:600;"
                f"margin-bottom:6px;'>Show columns</div>",
                unsafe_allow_html=True,
            )
        with rcol2:
            if st.button("Reset", key=f"colrst_{key}",
                         use_container_width=True, type="secondary"):
                for col in optional:
                    st.session_state[f"colchk_{key}__{col}"] = col in defaults
                st.rerun()

        if essential:
            st.markdown(
                f"<div style='font-size:0.74rem;color:{COLOR['text_dim']};"
                f"margin: 2px 0 10px; padding: 4px 8px; "
                f"background:#F4F6F8; border-radius:3px;'>"
                f"<b>Always shown:</b> {', '.join(essential)}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # One checkbox per optional column, stacked vertically.
        for col in optional:
            st.checkbox(col, key=f"colchk_{key}__{col}")

    chosen = [c for c in optional if st.session_state.get(f"colchk_{key}__{c}", False)]
    final = [c for c in all_cols if c in chosen or c in essential]
    if not final:
        final = defaults
    return df[final]


# --- Plotly defaults ------------------------------------------------------ #

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(244, 246, 248, 0.55)",
    font=dict(family="-apple-system, system-ui, Helvetica, Arial, sans-serif",
              size=12, color=COLOR["text"]),
    margin=dict(l=12, r=12, t=44, b=44),
    xaxis=dict(gridcolor=COLOR["grid"], zeroline=False, ticks="outside",
               linecolor=COLOR["border"], tickcolor=COLOR["border"]),
    yaxis=dict(gridcolor=COLOR["grid"], zeroline=False, ticks="outside",
               linecolor=COLOR["border"], tickcolor=COLOR["border"]),
    legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11), itemsizing="constant"),
)


def _layout(title: str, **overrides) -> dict:
    base = json.loads(json.dumps(PLOTLY_LAYOUT_BASE, default=str))
    # restore non-serialisable keys
    base["xaxis"] = PLOTLY_LAYOUT_BASE["xaxis"]
    base["yaxis"] = PLOTLY_LAYOUT_BASE["yaxis"]
    base["font"] = PLOTLY_LAYOUT_BASE["font"]
    base["legend"] = PLOTLY_LAYOUT_BASE["legend"]
    base["margin"] = PLOTLY_LAYOUT_BASE["margin"]
    base["title"] = dict(text=title.upper(),
                         font=dict(size=12, color=COLOR["text_dim"]),
                         x=0.0, xanchor="left", y=0.97)
    base.update(overrides)
    return base


def horizon_chart(forecast_kw_b: np.ndarray, forecast_kw_s: np.ndarray, times) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(times), y=forecast_kw_b.sum(axis=1),
                             mode="lines+markers", name="Baseline",
                             line=dict(width=2, color=COLOR["baseline"]), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=list(times), y=forecast_kw_s.sum(axis=1),
                             mode="lines+markers", name="Stress · heat + EV",
                             line=dict(width=2, color=COLOR["stress"]), marker=dict(size=4)))
    fig.update_layout(**_layout("TOTAL FEEDER LOAD · 24-HOUR FORECAST (KW)",
                                height=300, hovermode="x unified"))
    fig.update_yaxes(title_text="kW", title_font=dict(size=11, color=COLOR["text_dim"]))
    return fig


def violations_chart(res_b, res_s, times) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(times),
                         y=[len(r.voltage_violations) for r in res_b],
                         name="Baseline", marker_color=COLOR["baseline"]))
    fig.add_trace(go.Bar(x=list(times),
                         y=[len(r.voltage_violations) for r in res_s],
                         name="Stress", marker_color=COLOR["stress"]))
    fig.update_layout(**_layout("VOLTAGE VIOLATIONS PER HOUR",
                                barmode="group", height=290, bargap=0.25))
    fig.update_yaxes(title_text="count", title_font=dict(size=11, color=COLOR["text_dim"]))
    return fig


def reg_tap_chart(res_b, res_s, times) -> go.Figure:
    if not res_b:
        return go.Figure()
    reg_names = sorted({k for r in res_b for k in r.regulator_taps.keys()})
    fig = go.Figure()
    line_dash = {"reg1": "solid", "reg2": "dash"}
    for kind, results, op in [("Baseline", res_b, 1.0), ("Stress", res_s, 0.85)]:
        col = COLOR["baseline"] if kind == "Baseline" else COLOR["stress"]
        for name in reg_names:
            ys = [r.regulator_taps.get(name, None) for r in results]
            fig.add_trace(go.Scatter(
                x=list(times), y=ys, mode="lines+markers",
                name=f"{kind} · {name.upper()}",
                line=dict(width=2, color=col, dash=line_dash.get(name, "solid")),
                marker=dict(size=4), opacity=op,
            ))
    layout = _layout("REGULATOR TAP POSITIONS · QSTS", height=340)
    # 4 series wrap onto two rows in the legend — push the legend below the
    # x-axis tick labels so they don't collide.
    layout["margin"] = dict(l=12, r=12, t=44, b=110)
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.50,
                            xanchor="center", x=0.5,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10), itemsizing="constant",
                            itemwidth=40)
    layout["xaxis"] = dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                           ticks="outside", automargin=True)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="tap step (+ boost / − buck)",
                     title_font=dict(size=11, color=COLOR["text_dim"]))
    return fig


def violation_heatmap(mat_df: pd.DataFrame, dates) -> go.Figure:
    totals = mat_df.sum(axis=1)
    top_buses = totals[totals > 0].sort_values(ascending=False).head(20).index.tolist()
    if not top_buses:
        top_buses = mat_df.index.tolist()[:20]
    sub = mat_df.loc[top_buses]
    # Compact "Mon Jun 03" labels — short enough to lay flat without overlap.
    x_labels = [d.strftime("%a %-m/%-d") for d in dates]
    fig = go.Figure(data=go.Heatmap(
        z=sub.values,
        x=x_labels,
        y=[f"Bus {b}" for b in sub.index],
        colorscale=[[0, "#F4F6F8"], [0.01, "#EAE3D2"], [0.3, "#E2B062"], [0.6, "#C97244"], [1, "#9C2828"]],
        zmin=0, zmax=max(1, int(sub.values.max())),
        hovertemplate="Bus %{y}<br>%{x}<br><b>%{z}</b> violation hours<extra></extra>",
        colorbar=dict(title="hr", thickness=10, len=0.78, x=1.04,
                      tickfont=dict(size=10, color=COLOR["text_dim"])),
    ))
    layout = _layout("BUS × DAY · VOLTAGE-VIOLATION HOURS", height=540)
    # Title sits above the chart row; tall top margin so the date labels at the
    # top of the heatmap don't bleed into the title.
    layout["margin"] = dict(l=110, r=110, t=110, b=30)
    layout["title"] = dict(text="BUS × DAY · VOLTAGE-VIOLATION HOURS",
                           font=dict(size=12, color=COLOR["text_dim"]),
                           x=0.0, xanchor="left", y=0.985, yanchor="top")
    layout["xaxis"] = dict(side="top", tickfont=dict(size=10),
                           gridcolor="rgba(0,0,0,0)", automargin=True,
                           tickangle=0)
    layout["yaxis"] = dict(tickfont=dict(size=10),
                           gridcolor="rgba(0,0,0,0)", automargin=True)
    layout["legend"] = dict()
    fig.update_layout(**layout)
    return fig


def top_buses_bar(weekly_df: pd.DataFrame, n: int = 10) -> go.Figure:
    sub = weekly_df.head(n).copy()
    sub = sub[sub["violation_hours_week"] > 0]
    fig = go.Figure(go.Bar(
        x=sub["violation_hours_week"],
        y=[f"Bus {b}" for b in sub["bus"]],
        orientation="h",
        marker=dict(color=sub["violation_hours_week"],
                    colorscale=[[0, "#E2B062"], [0.5, "#C97244"], [1, "#9C2828"]],
                    showscale=False),
        text=[f"{int(v)} hr · worst {w:.3f} pu" for v, w in
              zip(sub["violation_hours_week"], sub["worst_v_pu"])],
        textposition="outside", textfont=dict(size=11, color=COLOR["text"]),
    ))
    layout = _layout(f"TOP {n} STRESSED BUSES · WEEKLY VIOLATION-HOURS", height=420)
    layout["margin"] = dict(l=20, r=140, t=44, b=20)
    layout["yaxis"] = dict(autorange="reversed", tickfont=dict(size=11), gridcolor="rgba(0,0,0,0)")
    layout["legend"] = dict()
    fig.update_layout(**layout)
    return fig


def weekly_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    if trend_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["week_start"], y=trend_df["total_violation_hours"],
        mode="lines+markers", line=dict(width=2, color=COLOR["stress"]),
        name="Total violation-hours", marker=dict(size=7),
    ))
    fig.update_layout(**_layout("MULTI-WEEK TREND · STRESS EVOLUTION", height=290))
    fig.update_yaxes(title_text="violation hours / week",
                     title_font=dict(size=11, color=COLOR["text_dim"]))
    return fig


# --- Scenario presets ----------------------------------------------------- #

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
                presets[f"Heatwave · {y}"] = t.to_pydatetime()
        mild = sub[(~sub["hw"]) & (sub["time"].dt.hour == 7)]
        if not mild.empty:
            t = mild["time"].iloc[len(mild) // 2]
            presets[f"Mild day · {y}"] = t.to_pydatetime()
    return presets


# --- Action-kind formatting ----------------------------------------------- #

KIND_LABEL = {
    "undervoltage": "Undervoltage",
    "overvoltage": "Overvoltage",
    "thermal_overload": "Thermal overload",
    "battery_install": "Battery installation",
    "volt_var_program": "Volt-VAR program",
    "cap_bank_install": "Capacitor bank installation",
    "reconductor": "Line reconductoring",
    "monitor": "Monitor / watch list",
}


def fmt_kind(k: str) -> str:
    return KIND_LABEL.get(k, k.replace("_", " ").title())


# --- Page setup ---------------------------------------------------------- #

st.set_page_config(
    page_title="APS Feeder Intelligence",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="collapsed",
)

# Restrained, utility-grade CSS. Neutral typography, sober spacing,
# narrow accent palette.
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: -apple-system, "Inter", system-ui, "Helvetica Neue", Arial, sans-serif;
        font-feature-settings: "tnum" 1, "ss01" 1;
        background: #FFFFFF;
    }}
    .block-container {{
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 1480px;
    }}
    h1, h2, h3, h4 {{
        font-weight: 600 !important;
        letter-spacing: -0.005em;
        color: {COLOR['text']};
    }}
    h2 {{ font-size: 1.55rem !important; margin-top: 1.4rem !important; margin-bottom: 0.6rem !important; }}
    h3 {{ font-size: 1.2rem !important; margin-top: 1.2rem !important; margin-bottom: 0.5rem !important; color: {COLOR['text']}; }}
    .top-rule {{
        height: 3px; background: {COLOR['accent']}; opacity: 0.95;
        margin: 0 0 1.4rem 0; border-radius: 1px;
    }}
    .nav-bar {{
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 0 0 0.4rem 0;
    }}
    .brand {{
        font-size: 1.45rem; font-weight: 700; color: {COLOR['text']};
        letter-spacing: -0.01em;
    }}
    .brand-tag {{
        margin-left: 12px; padding: 3px 9px; border: 1px solid {COLOR['accent']};
        color: {COLOR['accent']}; border-radius: 3px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .subtitle {{ color: {COLOR['text_dim']}; font-size: 0.85rem; margin-bottom: 1.6rem; }}
    div[data-testid="stMetric"] {{
        background: {COLOR['bg_card']};
        padding: 0.95rem 1.1rem;
        border-radius: 4px;
        border: 1px solid {COLOR['border']};
        border-left: 3px solid {COLOR['accent']};
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem !important; color: {COLOR['text_dim']};
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.85rem !important; font-weight: 600; color: {COLOR['text']};
        font-feature-settings: "tnum" 1;
    }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.85rem !important; }}
    button[data-baseweb="tab"] {{
        font-size: 0.92rem !important; padding: 0.6rem 1.3rem !important;
        font-weight: 500 !important; color: {COLOR['text_dim']};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLOR['text']} !important;
    }}
    .role-banner {{
        background: #EEF4FB;
        border-left: 3px solid {COLOR['baseline']};
        padding: 0.7rem 1rem;
        border-radius: 2px;
        margin: 0.4rem 0 1.0rem;
        font-size: 0.85rem; color: {COLOR['text']};
    }}
    .role-banner-planner {{
        background: #F1EEFB;
        border-left: 3px solid #6E5BB0;
        padding: 0.7rem 1rem;
        border-radius: 2px;
        margin: 0.4rem 0 1.0rem;
        font-size: 0.85rem; color: {COLOR['text']};
    }}
    .scenario-banner {{
        background: #FAF1DD;
        border-left: 3px solid {COLOR['accent']};
        padding: 0.65rem 1rem;
        border-radius: 2px;
        margin: 0.4rem 0 1.0rem;
        font-size: 0.85rem; color: {COLOR['text']};
    }}
    .priority-card {{
        background: #FCF3EC;
        border: 1px solid {COLOR['border']};
        border-left: 3px solid {COLOR['stress']};
        padding: 1.1rem 1.3rem;
        border-radius: 3px;
        margin-bottom: 1rem;
    }}
    .priority-card-ok {{
        background: #ECF7F1;
        border: 1px solid {COLOR['border']};
        border-left: 3px solid {COLOR['ok']};
        padding: 1.1rem 1.3rem;
        border-radius: 3px;
        margin-bottom: 1rem;
    }}
    .priority-card .label {{
        font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: {COLOR['stress']}; font-weight: 600;
    }}
    .priority-card-ok .label {{
        font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
        color: {COLOR['ok']}; font-weight: 600;
    }}
    .priority-card .body, .priority-card-ok .body {{
        font-size: 1.0rem; line-height: 1.45; margin-top: 4px; color: {COLOR['text']};
    }}
    .priority-card .meta, .priority-card-ok .meta {{
        font-size: 0.78rem; color: {COLOR['text_dim']}; margin-top: 6px;
    }}
    div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {{
        font-size: 0.86rem;
        font-feature-settings: "tnum" 1;
    }}
    a {{ color: {COLOR['accent']}; }}
    .stRadio > div {{ gap: 0.4rem; }}
    /* Filter popover button — make the trigger compact and icon-style. */
    button[data-testid="stPopoverButton"] {{
        background: #FFFFFF !important;
        border: 1px solid {COLOR['border']} !important;
        color: {COLOR['text']} !important;
        font-size: 0.82rem !important;
        padding: 4px 12px !important;
        border-radius: 3px !important;
    }}
    button[data-testid="stPopoverButton"]:hover {{
        border-color: {COLOR['accent']} !important;
        color: {COLOR['accent']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Top header ---------------------------------------------------------- #

st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
nav_l, nav_r = st.columns([8, 2])
with nav_l:
    st.markdown(
        f"""
        <div class='nav-bar'>
          <div>
            <span class='brand'>APS Feeder Intelligence</span>
            <span class='brand-tag'>Distribution operations</span>
          </div>
        </div>
        <div class='subtitle'>
          Spatio-temporal forecasting · OpenDSS QSTS validation · advisory decision layer
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_r:
    st.markdown(
        f"""
        <div style='text-align:right; padding-top:0.4rem;'>
          <a href='https://github.com/spraka52/aps-feeder-intelligence'
             style='text-decoration:none; padding:5px 12px;
                    border:1px solid {COLOR['border']}; border-radius:3px;
                    font-size:0.78rem; color:{COLOR['text_dim']};
                    letter-spacing:0.06em; text-transform:uppercase;'>
            View source
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not CKPT_PATH.exists() or not BASELINE_NPZ.exists():
    st.error("Missing artifacts. Run `python -m data.synthesize --multi --customers resstock` then `python -m models.train --epochs 25`.")
    st.stop()

# Load
ckpt_sig = _file_signature(CKPT_PATH)
base_sig = _file_signature(BASELINE_NPZ)
stress_sig = _file_signature(STRESS_NPZ)

forecaster = _get_forecaster(ckpt_sig[0])
ds_base = _get_dataset(str(BASELINE_NPZ), *base_sig)
ds_stress = _get_dataset(str(STRESS_NPZ), *stress_sig)

times = ds_base.times
all_days = sorted({t.date() for t in times})
years_available = sorted({t.year for t in times})

# --- Role switch -------------------------------------------------------- #

role = st.radio(
    "Role",
    ["Operator · hour-by-hour", "Planner · week-by-week"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("")


# =============================================================================
#                              OPERATOR VIEW
# =============================================================================

def render_operator_view():
    st.markdown(
        f"<div class='role-banner'>"
        f"<b style='color:{COLOR['baseline']};'>OPERATOR VIEW</b> &nbsp; "
        f"Pick a forecast window. The model predicts the next 24 hours per bus, "
        f"OpenDSS QSTS validates the physics each hour, and the Action Center ranks "
        f"what to do — sized in kW, ordered by priority."
        f"</div>",
        unsafe_allow_html=True,
    )

    scenarios = _summer_scenarios(times, ds_base.heatwave)
    st.markdown("##### Forecast window")
    scenario_keys = list(scenarios.keys()) + ["Custom date / hour"]
    pick_scenario = st.radio(
        "Forecast window", scenario_keys, index=0,
        horizontal=True, label_visibility="collapsed",
    )

    if pick_scenario == "Custom date / hour":
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
        f"<div class='scenario-banner'>"
        f"<b>{scenario_label}</b> &nbsp;·&nbsp; "
        f"horizon: {fcst_times[0].strftime('%a %b %d, %H:%M')} → "
        f"{fcst_times[-1].strftime('%a %b %d, %H:%M')} &nbsp;·&nbsp; "
        f"<b>{hw_in_window}/24</b> heatwave hours in window"
        f"</div>",
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
        st.warning("OpenDSS solver did not converge for this window — pick a different start.")

    hw_mask = ds_base.heatwave[t0 + forecaster.horizon_in : t0 + forecaster.horizon_in + forecaster.horizon_out]
    actions_base = build_actions(res_base, fcst_times, fcst_base, list(bus_order), hw_mask)
    actions_stress = build_actions(res_stress, fcst_times, fcst_stress, list(bus_order), hw_mask)
    kpi_base = headline_kpis(res_base, fcst_base, hw_mask)
    kpi_stress = headline_kpis(res_stress, fcst_stress, hw_mask)

    # Top action callout
    df_stress_actions = ops_actions_to_df(actions_stress)
    if not df_stress_actions.empty:
        top = df_stress_actions.iloc[0]
        first_hour = pd.Timestamp(top["when"])
        hours_until = max(0, int((first_hour - fcst_times[0]).total_seconds() // 3600))
        urgency = "Take action now" if hours_until <= 1 else f"Pre-position in {hours_until} hours"
        st.markdown(
            f"""
            <div class='priority-card'>
              <div class='label'>Priority {int(top['priority'])} · {urgency}</div>
              <div class='body'>{top['recommendation']}</div>
              <div class='meta'>
                {fmt_kind(top['kind'])} at Bus {top['bus_or_line']} ·
                worst at {first_hour.strftime('%a %b %d %H:%M')} ·
                {int(top['hours_affected'])} hours affected ·
                severity {float(top['severity']):.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='priority-card-ok'><div class='label'>No action required</div>"
            f"<div class='body'>Feeder is operating within voltage and thermal limits "
            f"across the next 24 hours.</div></div>",
            unsafe_allow_html=True,
        )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak feeder load", f"{kpi_stress['peak_forecast_kw']:.0f} kW",
              f"{kpi_stress['peak_forecast_kw'] - kpi_base['peak_forecast_kw']:+.0f} vs base",
              help="Highest single-hour total kW.")
    c2.metric("Voltage violations", kpi_stress["n_voltage_violations"],
              f"{kpi_stress['n_voltage_violations'] - kpi_base['n_voltage_violations']:+d} vs base",
              delta_color="inverse")
    c3.metric("Thermal overloads", kpi_stress["n_thermal_overloads"],
              f"{kpi_stress['n_thermal_overloads'] - kpi_base['n_thermal_overloads']:+d} vs base",
              delta_color="inverse")
    c4.metric("Peak losses", f"{kpi_stress['peak_loss_kw']:.0f} kW",
              f"{kpi_stress['peak_loss_kw'] - kpi_base['peak_loss_kw']:+.0f} vs base",
              delta_color="inverse")

    st.markdown("")

    # Tabs
    tab_map, tab_forecast, tab_timeline, tab_actions = st.tabs([
        "Operations Map", "Forecast & Physics", "Hourly Action Timeline", "Action Center",
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
            compare_view = st.toggle("Side-by-side baseline", value=False)
        map_hour = fcst_hour_options.index(selected_hour_label)
        voltage_legend_chip()

        voltages_per_hour_base = [r.bus_voltage_pu for r in res_base]
        voltages_per_hour_stress = [r.bus_voltage_pu for r in res_stress]
        worst_v_base = min(voltages_per_hour_base[map_hour].values()) if voltages_per_hour_base[map_hour] else float("nan")
        worst_v_stress = min(voltages_per_hour_stress[map_hour].values()) if voltages_per_hour_stress[map_hour] else float("nan")

        def _v_label(v):
            color = COLOR['alert'] if (v < 0.95 or v > 1.05) else COLOR['ok']
            return f"<span style='color:{color};'><b>{v:.3f} pu</b></span>"

        if compare_view:
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**Baseline** &nbsp;·&nbsp; worst V: {_v_label(worst_v_base)}", unsafe_allow_html=True)
                st.pydeck_chart(
                    feeder_map_deck(voltages_per_hour_base, map_hour, ops_actions_to_df(actions_base)),
                    height=520,
                )
            with mc2:
                st.markdown(f"**Stress · heat + EV** &nbsp;·&nbsp; worst V: {_v_label(worst_v_stress)}", unsafe_allow_html=True)
                st.pydeck_chart(
                    feeder_map_deck(voltages_per_hour_stress, map_hour, ops_actions_to_df(actions_stress)),
                    height=520,
                )
        else:
            st.markdown(
                f"**Stress scenario · heat + 35% EV** &nbsp;·&nbsp; worst V at this hour: {_v_label(worst_v_stress)}"
                f" &nbsp;·&nbsp; <span style='color:{COLOR['text_dim']};'>baseline worst: {worst_v_base:.3f} pu</span>",
                unsafe_allow_html=True,
            )
            st.pydeck_chart(
                feeder_map_deck(voltages_per_hour_stress, map_hour, ops_actions_to_df(actions_stress)),
                height=620,
            )

    with tab_forecast:
        st.plotly_chart(horizon_chart(fcst_base, fcst_stress, fcst_times), width="stretch")
        cf1, cf2 = st.columns(2)
        with cf1:
            st.plotly_chart(violations_chart(res_base, res_stress, fcst_times), width="stretch")
        with cf2:
            st.plotly_chart(reg_tap_chart(res_base, res_stress, fcst_times), width="stretch")

    with tab_timeline:
        st.caption("One row per forecast hour. Each row shows whether OpenDSS flagged a problem at that hour and what to do.")
        rows = []
        actions_df = ops_actions_to_df(actions_stress)
        for h, t in enumerate(fcst_times):
            r = res_stress[h]
            n_v = len(r.voltage_violations)
            n_t = len(r.thermal_overloads)
            worst_pu = min(r.bus_voltage_pu.values()) if r.bus_voltage_pu else float("nan")
            hour_action = "—"
            if not actions_df.empty:
                same_hour = actions_df[pd.to_datetime(actions_df["when"]) == pd.Timestamp(t)]
                if not same_hour.empty:
                    hour_action = same_hour.iloc[0]["recommendation"]
            if n_v == 0 and n_t == 0:
                status = "Clean"
            elif n_v + n_t > 3:
                status = "Stressed"
            else:
                status = "Watch"
            rows.append({
                "Hour": t.strftime("%a %b %d · %H:%M"),
                "Status": status,
                "V violations": n_v,
                "Thermal overloads": n_t,
                "Worst V (pu)": f"{worst_pu:.3f}",
                "Total kW": f"{fcst_stress[h].sum():.0f}",
                "Recommended action": hour_action,
            })
        timeline_df = pd.DataFrame(rows)
        timeline_df = column_picker(
            timeline_df, key="op_timeline",
            default_cols=list(timeline_df.columns),
            essential_cols=["Hour", "Status"],
        )
        scrollable_table(timeline_df, max_height=620)

    with tab_actions:
        show_n = st.slider("Top N actions", 3, 15, 8)
        sub_stress, sub_base = st.tabs(["Stress scenario", "Baseline scenario"])

        def _render(actions_list, label, picker_key: str):
            df = ops_actions_to_df(actions_list).head(show_n)
            if df.empty:
                st.success(f"No violations in the {label} forecast.")
                return
            df["kind"] = df["kind"].apply(fmt_kind)
            # Pre-format numeric columns so they render cleanly in the HTML table.
            df["severity"] = df["severity"].map(lambda x: f"{float(x):.2f}")
            df["target_kw"] = df["target_kw"].map(
                lambda x: f"{float(x):.0f}" if pd.notna(x) and x else "—")
            df["when"] = pd.to_datetime(df["when"]).dt.strftime("%a %b %d · %H:%M")
            df_disp = df.rename(columns={
                "priority": "Pri.", "kind": "Kind", "bus_or_line": "Bus / line",
                "when": "Worst hour", "hours_affected": "Hrs",
                "severity": "Severity", "target_kw": "Sized kW",
                "detail": "Description", "recommendation": "Recommended action",
            })
            ordered = ["Pri.", "Kind", "Bus / line", "Worst hour", "Hrs",
                       "Severity", "Sized kW", "Description", "Recommended action"]
            df_disp = df_disp[ordered]
            df_disp = column_picker(
                df_disp, key=picker_key,
                default_cols=ordered,
                essential_cols=["Pri.", "Bus / line", "Recommended action"],
            )
            scrollable_table(df_disp, max_height=520)

        with sub_stress:
            _render(actions_stress, "stress", picker_key="op_actions_stress")
        with sub_base:
            _render(actions_base, "baseline", picker_key="op_actions_base")


# =============================================================================
#                              PLANNER VIEW
# =============================================================================

def _build_summer_weeks(times: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return only weeks where the dataset has at least 6 days × 24 hours of data.

    The dataset only covers Jun-Aug for each year — naively iterating week-by-week
    from min to max would generate Sep-May weeks that have zero hours and crash
    the downstream solver.
    """
    if len(times) == 0:
        return []
    times_sorted = pd.DatetimeIndex(sorted(set(times)))
    weeks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    # Walk year by year, only emit weeks that lie inside the per-year coverage.
    for year in sorted({t.year for t in times_sorted}):
        year_times = times_sorted[times_sorted.year == year]
        if len(year_times) == 0:
            continue
        start = year_times.min().normalize()
        end = year_times.max()
        while start + pd.Timedelta(days=7) <= end + pd.Timedelta(hours=1):
            ws = start
            we = ws + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)
            n_hours = int(((times_sorted >= ws) & (times_sorted <= we)).sum())
            if n_hours >= 24 * 6:
                weeks.append((ws, we))
            start = start + pd.Timedelta(days=7)
    return weeks


def render_planner_view():
    st.markdown(
        f"<div class='role-banner-planner'>"
        f"<b style='color:#A78BFF;'>PLANNER VIEW</b> &nbsp; "
        f"Pick a week. We aggregate the OpenDSS QSTS solve across the whole "
        f"week, identify chronic stress patterns, and propose ranked capital projects "
        f"— sized in kW with rough cost + payback estimates."
        f"</div>",
        unsafe_allow_html=True,
    )

    weeks = _build_summer_weeks(times)
    if not weeks:
        st.error("Dataset doesn't contain any full weeks.")
        return

    week_labels = [f"Week of {ws.strftime('%a %b %d, %Y')}" for ws, _ in weeks]
    cw1, cw2 = st.columns([3, 2])
    with cw1:
        pick_week_label = st.selectbox("Week to analyse", week_labels, index=min(2, len(week_labels) - 1))
    with cw2:
        scenario_for_planner = st.radio(
            "Scenario", ["Baseline", "Stress · heat + EV"],
            index=1, horizontal=True,
        )

    week_idx = week_labels.index(pick_week_label)
    ws, we = weeks[week_idx]

    ds_for = ds_stress if "Stress" in scenario_for_planner else ds_base
    week_mask = (ds_for.times >= ws) & (ds_for.times <= we)
    week_times = ds_for.times[week_mask]
    week_loads = ds_for.loads[:, week_mask]

    n_days = len(week_times) // 24
    if n_days < 1:
        st.error("Selected week has fewer than 24 hours of data.")
        return
    day_kw = week_loads[:, : n_days * 24].T.reshape(n_days, 24, -1).astype(np.float32)
    bus_order = tuple(ds_for.bus_order)

    week_label_key = f"{ws.isoformat()}_{scenario_for_planner}"
    per_day_dicts = _solve_week_truth(
        day_kw.astype(np.float32).tobytes(),
        bus_order, day_kw.shape, week_label_key,
    )
    per_day_results = [_to_hour_results(d) for d in per_day_dicts]

    weekly_df = aggregate_weekly_violations(per_day_results, list(bus_order))
    hours_matrix = _bus_day_hours_matrix(per_day_results, list(bus_order))
    nominal_map = {b: SPOT_LOADS_KW.get(b, 0.0) for b in bus_order}
    plan_actions = build_planner_actions(weekly_df, nominal_map)

    trend_rows = []
    for w_start, w_end in weeks:
        w_mask = (ds_for.times >= w_start) & (ds_for.times <= w_end)
        if w_mask.sum() < 24:
            continue
        wk_temps = ds_for.temp[w_mask]
        wk_loads = ds_for.loads[:, w_mask].sum(axis=0)
        trend_rows.append({
            "week_start": w_start,
            "total_violation_hours": int((wk_temps >= 41).sum()),
            "peak_kw": float(wk_loads.max()) if wk_loads.size else 0.0,
        })
    trend_df = pd.DataFrame(trend_rows)
    actual_total = int(weekly_df["violation_hours_week"].sum())
    if not trend_df.empty:
        sel_row = trend_df[trend_df["week_start"] == ws]
        if not sel_row.empty:
            trend_df.loc[sel_row.index, "total_violation_hours"] = actual_total

    total_v = int(weekly_df["violation_hours_week"].sum())
    worst_bus = weekly_df.iloc[0] if not weekly_df.empty else None
    n_buses_hit = int((weekly_df["violation_hours_week"] > 0).sum())
    peak_week_kw = float(week_loads.sum(axis=0).max()) if week_loads.size else 0.0

    st.markdown(f"### Week of {ws.strftime('%a %b %d, %Y')} · {scenario_for_planner}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total voltage violations", f"{total_v}")
    if worst_bus is not None and worst_bus["violation_hours_week"] > 0:
        k2.metric("Worst bus", f"Bus {worst_bus['bus']}",
                  f"{int(worst_bus['violation_hours_week'])} hr · worst {worst_bus['worst_v_pu']:.3f} pu")
    else:
        k2.metric("Worst bus", "—", "no violations")
    k3.metric("Buses with violations", f"{n_buses_hit} / {len(bus_order)}")
    k4.metric("Peak weekly load", f"{peak_week_kw:.0f} kW")

    st.markdown("")

    tab_hm, tab_top, tab_trend, tab_capex = st.tabs([
        "Bus × Day Heatmap", "Top Stressed Buses",
        "Multi-Week Trend", "Capital Action Plan",
    ])

    with tab_hm:
        day_dates = pd.date_range(ws, periods=n_days, freq="D")
        st.plotly_chart(violation_heatmap(hours_matrix, day_dates), width="stretch")
        st.caption(
            "Daily voltage-violation hours per bus across the selected week. "
            "Top 20 most-stressed buses shown — empty cells mean the bus was inside the "
            "[0.95, 1.05] pu band that day."
        )

    with tab_top:
        st.plotly_chart(top_buses_bar(weekly_df, n=10), width="stretch")
        with st.expander("Per-bus weekly stats (full table)"):
            stats_df = weekly_df.copy()
            stats_df["worst_v_pu"] = stats_df["worst_v_pu"].map(lambda x: f"{x:.3f} pu")
            stats_df = stats_df.rename(columns={
                "bus": "Bus", "violation_hours_week": "Violation hr/wk",
                "worst_v_pu": "Worst V (pu)", "days_with_violation": "Days affected",
            })
            stats_df = column_picker(
                stats_df, key="planner_perbus",
                default_cols=list(stats_df.columns),
                essential_cols=["Bus"],
            )
            scrollable_table(stats_df, max_height=420)

    with tab_trend:
        st.plotly_chart(weekly_trend_chart(trend_df), width="stretch")
        st.caption(
            "Selected week shows the actual computed violation count from the OpenDSS solve; "
            "other weeks show heat-stress hours per week as a proxy for the full season."
        )

    with tab_capex:
        st.caption(
            "Capital projects ranked by annualised violation hours × severity. "
            "Cost and payback are order-of-magnitude planning estimates — useful for "
            "deciding *where* to invest, not for procurement."
        )
        with st.expander("How are cost and payback calculated?"):
            st.markdown(
                """
                **Sizing**
                - **Battery (undervoltage)** kW = `clip(20 × (0.95 − worst_pu)/0.01 × nominal/100 × 5, 25, 2000)`
                  — closes the voltage-sag gap with real-power injection.
                - **Volt-VAR program (overvoltage)** kVAr = `50 + 3 × bus_nominal_kw`
                  — reactive support to absorb PV backfeed.

                **Cost (capex, order-of-magnitude 2024 USD)**
                - Battery: **$1,500 / kW** installed (inverter + civil + commissioning).
                - Volt-VAR: **$100 / kVAr** of reactive support.

                **Annualised violation hours**
                `hours_year = hours_this_week × 13 weeks × 3 seasons / 3`
                — projects this week onto a 13-summer-week season, repeated for ~3 stress seasons/yr.

                **Payback (years)** = `cost / annual_avoided_value`
                - **Battery**: avoided value = `hours_year × 60 min × 20 customers/bus × $0.15/min`
                  (CAIDI proxy: $0.15 per customer-minute of avoided outage).
                - **Volt-VAR**: avoided value = `hours_year × 50 kW curtailed × $0.08/kWh`
                  (recovered PV revenue at wholesale).
                - **Monitor / watch-list** entries have no investment, so payback is *n/a* by design.
                - Payback above 100 years is shown as *n/a* — beyond planning horizon.
                """
            )
        if not plan_actions:
            st.success("No capital projects warranted this week — feeder is operating within limits.")
        else:
            df_actions = planner_actions_to_df(plan_actions)
            df_actions["kind"] = df_actions["kind"].apply(fmt_kind)
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
                "rationale": "Rationale",
                "payback_years": "Payback (yr)",
            })
            df_disp["Cost (USD)"] = df_disp["Cost (USD)"].map(
                lambda x: f"${x:,.0f}" if x and x > 0 else "—")
            df_disp["Hr/yr (annualized)"] = df_disp["Hr/yr (annualized)"].map(lambda x: f"{x:.0f}")
            df_disp["Size (kW)"] = df_disp["Size (kW)"].map(
                lambda x: f"{x:.0f}" if x and x > 0 else "—")
            df_disp["Worst V"] = df_disp["Worst V"].map(lambda x: f"{x:.3f} pu")
            df_disp["Payback (yr)"] = df_disp["Payback (yr)"].map(
                lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
            ordered = ["Pri.", "Bus", "Project type", "Size (kW)", "Cost (USD)",
                       "Hr/wk now", "Hr/yr (annualized)", "Worst V",
                       "Days affected", "Payback (yr)", "Rationale"]
            df_capex = df_disp[ordered]
            df_capex = column_picker(
                df_capex, key="planner_capex",
                default_cols=ordered,
                essential_cols=["Pri.", "Bus", "Project type"],
            )
            scrollable_table(df_capex, max_height=460)


# Render
if role.startswith("Operator"):
    render_operator_view()
else:
    render_planner_view()


# Footer
st.markdown("")
st.markdown(
    f"<div style='border-top: 1px solid {COLOR['border']}; padding-top: 1rem; "
    f"margin-top: 1.5rem; font-size: 0.78rem; color: {COLOR['text_dim']};'>"
    f"APS / ASU AI for Energy hackathon. NOAA NCEI weather · NREL NSRDB irradiance · "
    f"NREL ResStock + ComStock load profiles · OpenDSS QSTS physics validation · "
    f"GraphSAGE + GRU forecaster (27 k parameters)."
    f"</div>",
    unsafe_allow_html=True,
)
