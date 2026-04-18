"""Convert forecasts + OpenDSS results into prioritized utility actions.

For each hour of the forecast horizon we ask:
  * Is any bus voltage outside [0.95, 1.05] pu?
  * Is any line carrying more than 100% of NormAmps?

Then we group those by location and time, score them by severity (how far
out of bounds, and how persistent), and emit a recommended intervention:

  - Undervoltage at a load pocket -> "Dispatch battery / capacitor" at the
    nearest tappable bus, with a target kW that closes the gap.
  - Overvoltage from PV backfeed   -> "Curtail DER / engage Volt-VAR" upstream.
  - Thermal overload               -> "Shed deferrable load (e.g., EV)" or
    reconfigure the affected line segment.

The output is a list of `Action` records that the dashboard renders directly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from physics.opendss_runner import HourResult


VMIN_PU = 0.95
VMAX_PU = 1.05


@dataclass
class Action:
    priority: int                 # 1 (highest) .. N
    severity: float               # composite score (higher = worse)
    kind: str                     # "undervoltage" | "overvoltage" | "thermal"
    bus_or_line: str
    when: str                     # ISO timestamp
    hours_affected: int
    detail: str
    recommendation: str
    target_kw: Optional[float] = None    # rough sizing for the intervention


def _bus_geo_lookup() -> Dict[str, tuple[float, float]]:
    from data.topology import COORDS
    return COORDS


def _nearest_bus_for_line(line_name: str) -> str:
    """Line name format is 'l_<from>_<to>' — return the downstream bus."""
    parts = line_name.lower().split("_")
    if len(parts) >= 3:
        return parts[2]
    return line_name


def build_actions(
    results: List[HourResult],
    times: pd.DatetimeIndex,
    forecast_kw: np.ndarray,
    bus_order: List[str],
    in_heatwave: Optional[np.ndarray] = None,
) -> List[Action]:
    """Walk the OpenDSS results and produce a ranked action list.

    forecast_kw shape: [T_out, N]
    """
    # Aggregate violations by (kind, location) across the horizon
    agg: Dict[tuple[str, str], dict] = {}

    for r in results:
        ts = times[r.hour_index] if r.hour_index < len(times) else None
        for bus, v in r.voltage_violations:
            kind = "undervoltage" if v < VMIN_PU else "overvoltage"
            key = (kind, bus)
            entry = agg.setdefault(key, {"hours": 0, "worst": v, "worst_when": ts, "values": []})
            entry["hours"] += 1
            entry["values"].append(v)
            # "worst" = furthest from nominal in the violating direction
            if (kind == "undervoltage" and v < entry["worst"]) or (kind == "overvoltage" and v > entry["worst"]):
                entry["worst"] = v
                entry["worst_when"] = ts
        for line, pct in r.thermal_overloads:
            key = ("thermal", line)
            entry = agg.setdefault(key, {"hours": 0, "worst": 0.0, "worst_when": ts, "values": []})
            entry["hours"] += 1
            entry["values"].append(pct)
            if pct > entry["worst"]:
                entry["worst"] = pct
                entry["worst_when"] = ts

    # Score & convert to Actions
    actions: List[Action] = []
    bus_to_idx = {b: i for i, b in enumerate(bus_order)}

    for (kind, where), entry in agg.items():
        hours = entry["hours"]
        worst = entry["worst"]
        when = entry["worst_when"].isoformat() if entry["worst_when"] is not None else "n/a"

        if kind == "undervoltage":
            gap_pu = max(0.0, VMIN_PU - worst)         # how far below 0.95
            severity = 100.0 * gap_pu * (1.0 + 0.05 * hours)
            target_bus = where
            # rough sizing: closing a 1 pu sag at peak load ~ 60% of bus kW
            i = bus_to_idx.get(target_bus)
            peak_kw = float(forecast_kw[:, i].max()) if i is not None else 0.0
            target = round(0.6 * peak_kw * (gap_pu / 0.05), 1) if peak_kw > 0 else None
            detail = f"Bus {target_bus} dipped to {worst:.3f} pu (limit 0.95) for {hours} hour(s)."
            rec = (
                f"Dispatch battery / cap bank at Bus {target_bus} "
                f"≈ {target} kW to close the voltage gap"
                if target else f"Dispatch reactive support near Bus {target_bus}"
            )
            actions.append(Action(
                priority=0, severity=severity, kind=kind,
                bus_or_line=target_bus, when=when, hours_affected=hours,
                detail=detail, recommendation=rec, target_kw=target,
            ))

        elif kind == "overvoltage":
            gap_pu = max(0.0, worst - VMAX_PU)
            severity = 100.0 * gap_pu * (1.0 + 0.05 * hours)
            i = bus_to_idx.get(where)
            peak_kw = float(forecast_kw[:, i].max()) if i is not None else 0.0
            target = round(0.4 * peak_kw * (gap_pu / 0.05), 1) if peak_kw > 0 else None
            detail = f"Bus {where} rose to {worst:.3f} pu (limit 1.05) for {hours} hour(s)."
            rec = (
                f"Curtail DER export and engage Volt-VAR control near Bus {where}"
                + (f" (~{target} kW reduction)" if target else "")
            )
            actions.append(Action(
                priority=0, severity=severity, kind=kind,
                bus_or_line=where, when=when, hours_affected=hours,
                detail=detail, recommendation=rec, target_kw=target,
            ))

        else:  # thermal
            severity = (worst - 100.0) * (1.0 + 0.05 * hours)
            target_bus = _nearest_bus_for_line(where)
            i = bus_to_idx.get(target_bus)
            peak_kw = float(forecast_kw[:, i].max()) if i is not None else 0.0
            shed_kw = round(peak_kw * (worst - 100.0) / 100.0, 1) if peak_kw > 0 else None
            detail = f"Line {where} loaded to {worst:.1f}% of NormAmps for {hours} hour(s)."
            rec = (
                f"Shed deferrable load (EV / HVAC pre-cool) at Bus {target_bus} "
                f"≈ {shed_kw} kW or open a tie switch to offload"
                if shed_kw else f"Shed deferrable load downstream of {where}"
            )
            actions.append(Action(
                priority=0, severity=severity, kind="thermal_overload",
                bus_or_line=where, when=when, hours_affected=hours,
                detail=detail, recommendation=rec, target_kw=shed_kw,
            ))

    actions.sort(key=lambda a: (-a.severity, a.kind, a.bus_or_line))
    for i, a in enumerate(actions, start=1):
        a.priority = i
    return actions


def actions_to_df(actions: List[Action]) -> pd.DataFrame:
    if not actions:
        return pd.DataFrame(columns=["priority", "severity", "kind", "bus_or_line", "when", "hours_affected", "detail", "recommendation", "target_kw"])
    return pd.DataFrame([asdict(a) for a in actions])


# ---- Headline operator KPIs --------------------------------------------------

def headline_kpis(
    results: List[HourResult], forecast_kw: np.ndarray,
    in_heatwave: Optional[np.ndarray] = None,
) -> dict:
    n_v = sum(len(r.voltage_violations) for r in results)
    n_t = sum(len(r.thermal_overloads) for r in results)
    peak_total = float(np.array([r.total_load_kw for r in results]).max() if results else 0.0)
    peak_loss = float(np.array([r.total_losses_kw for r in results]).max() if results else 0.0)
    feeder_load = forecast_kw.sum(axis=1)
    return {
        "n_voltage_violations": n_v,
        "n_thermal_overloads": n_t,
        "peak_feeder_kw": peak_total,
        "peak_loss_kw": peak_loss,
        "peak_forecast_kw": float(feeder_load.max()) if feeder_load.size else 0.0,
        "avg_forecast_kw": float(feeder_load.mean()) if feeder_load.size else 0.0,
        "n_stress_hours": int(in_heatwave.sum()) if in_heatwave is not None else 0,
    }
