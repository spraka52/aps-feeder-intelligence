"""Long-horizon (weekly / multi-week) planner actions.

Operator actions (in `action_engine.py`) answer *what do I do in the next
24 hours?* — tactical dispatch recommendations.

Planner actions answer *where should we invest in new capacity this year?* —
strategic, capital-level recommendations based on how a feeder behaves over
weeks, not hours. A bus that violates voltage for 3 hours once a month is
nuisance-worthy; a bus that violates 15 hours every single week is a
capital-project candidate.

We deliberately keep the cost / payback numbers rough — the point is to
rank which buses deserve a project, not to produce a bankable business case.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from physics.opendss_runner import HourResult


VMIN_PU = 0.95
VMAX_PU = 1.05


@dataclass
class PlannerAction:
    priority: int
    bus: str
    kind: str                   # "battery_install" | "cap_bank_install" | "reconductor" | "ders_curtailment"
    suggested_kw: float         # rough sizing (kW)
    est_cost_usd: float         # rough capital cost
    violation_hours_per_week: float
    violation_hours_per_year: float
    worst_v_pu: float
    n_days_with_violation: int
    rationale: str
    payback_years: Optional[float] = None  # rough — uses $0.15/kWh avoided-customer-minute-equivalent


def _aggregate_bus_week(bus: str, per_day_results: List[List[HourResult]]) -> dict:
    """Aggregate voltage-violation stats for one bus across a week of hours."""
    total_hours = 0
    worst_v = 1.0
    days_with_violation = 0
    for day_results in per_day_results:
        had_violation_today = False
        for hr in day_results:
            if not hr.converged:
                continue
            v = hr.bus_voltage_pu.get(bus)
            if v is None:
                continue
            if v < VMIN_PU:
                total_hours += 1
                had_violation_today = True
                if v < worst_v:
                    worst_v = v
            elif v > VMAX_PU:
                total_hours += 1
                had_violation_today = True
                if v > worst_v:
                    worst_v = v
        if had_violation_today:
            days_with_violation += 1
    return {
        "bus": bus,
        "violation_hours_week": total_hours,
        "worst_v_pu": worst_v,
        "days_with_violation": days_with_violation,
    }


def aggregate_weekly_violations(
    per_day_results: List[List[HourResult]],
    bus_order: List[str],
) -> pd.DataFrame:
    """Per-bus weekly aggregate — one row per bus, sorted worst-first."""
    rows = [_aggregate_bus_week(b, per_day_results) for b in bus_order]
    df = pd.DataFrame(rows).sort_values("violation_hours_week", ascending=False)
    return df.reset_index(drop=True)


def _bus_day_hours_matrix(
    per_day_results: List[List[HourResult]],
    bus_order: List[str],
) -> pd.DataFrame:
    """Matrix [bus × day] where each cell = number of violation hours that day."""
    mat = np.zeros((len(bus_order), len(per_day_results)), dtype=int)
    bus_idx = {b: i for i, b in enumerate(bus_order)}
    for d, day_results in enumerate(per_day_results):
        for hr in day_results:
            if not hr.converged:
                continue
            for bus, v in hr.bus_voltage_pu.items():
                if v is None:
                    continue
                if v < VMIN_PU or v > VMAX_PU:
                    i = bus_idx.get(bus)
                    if i is not None:
                        mat[i, d] += 1
    return pd.DataFrame(mat, index=bus_order, columns=[f"D{d+1}" for d in range(len(per_day_results))])


# ---- Capital-action heuristics ------------------------------------------- #

# Rough unit costs (2024 USD), order-of-magnitude:
# - Grid-scale battery: ~$1,500 / kW installed (includes inverter + civil)
# - Capacitor bank: ~$50 / kVAr
# - Reconductor (line upgrade): ~$300,000 / mile
# - Avoided-outage value: ~$0.15 / customer-minute (CAIDI proxy)


def _size_battery_for_sag(worst_v_pu: float, bus_nominal_kw: float) -> float:
    """Rough kW sizing: close the voltage gap via real-power support."""
    gap_pu = max(0.0, VMIN_PU - worst_v_pu)
    if gap_pu == 0:
        return 0.0
    # A rule-of-thumb: 1% voltage restoration needs ~20% of bus nominal kW
    # as real-power injection at typical X/R ratios. Clamp to a sensible band.
    kw = 20.0 * (gap_pu / 0.01) * (bus_nominal_kw / 100.0) * 5.0
    return float(np.clip(kw, 25.0, 2000.0))


def build_planner_actions(
    weekly_df: pd.DataFrame,
    bus_nominal_kw: dict[str, float],
    seasons_per_year: int = 3,  # we model summer only; extrapolate conservatively
    min_hours_for_capex: float = 1.0,
    min_hours_for_monitor: float = 0.0,
) -> List[PlannerAction]:
    """Turn the weekly aggregate into capital-project recommendations.

    Ranking is by annualized violation hours × severity, so a bus that
    violates 14 h this week (every day) ranks higher than one that violates
    10 h in a single-day spike.

    Buses with at least one violation hour become *capital* candidates;
    buses with no violation but high nominal load get a *monitor* entry so
    a planner always sees the broader watch list, not only the worst case.
    """
    actions: List[PlannerAction] = []
    seen_buses: set[str] = set()
    for _, row in weekly_df.iterrows():
        hrs_week = float(row["violation_hours_week"])
        if hrs_week < min_hours_for_capex:
            continue

        bus = row["bus"]
        worst = float(row["worst_v_pu"])
        nominal = float(bus_nominal_kw.get(bus, 50.0))
        # Annualize: 13 summer weeks × N simulated seasons/year + a shoulder factor
        hours_year = hrs_week * 13 * seasons_per_year / 3.0
        seen_buses.add(bus)

        if worst < VMIN_PU:
            # Undervoltage: battery install (real-power) is the canonical answer
            size_kw = _size_battery_for_sag(worst, nominal)
            cost = size_kw * 1500.0
            avoided_cust_mins = hours_year * 60 * 20  # rough: 20 customers downstream affected
            avoided_usd = avoided_cust_mins * 0.15
            payback = cost / max(avoided_usd, 1.0)
            actions.append(PlannerAction(
                priority=0,
                bus=bus,
                kind="battery_install",
                suggested_kw=size_kw,
                est_cost_usd=cost,
                violation_hours_per_week=hrs_week,
                violation_hours_per_year=hours_year,
                worst_v_pu=worst,
                n_days_with_violation=int(row["days_with_violation"]),
                rationale=(
                    f"Undervoltage at Bus {bus} "
                    f"({hrs_week:.0f} hr/week, worst {worst:.3f} pu). "
                    f"Battery / cap-bank dispatch closes the gap; "
                    f"battery also enables EV-load deferral and DR enrollment."
                ),
                payback_years=payback if payback < 100 else None,
            ))
        elif worst > VMAX_PU:
            # Overvoltage: PV-backfeed symptom → Volt-VAR + smart inverter coordination
            size_kvar = 50.0 + 3.0 * nominal
            cost = size_kvar * 100.0  # reactive-support cost
            # Overvoltage payback: avoided PV curtailment. When inverters trip on
            # overvoltage, ~50 kW of generation per cluster is curtailed. Recover
            # that at ~$0.08/kWh wholesale-equivalent.
            avoided_kwh = hours_year * 50.0
            avoided_usd = avoided_kwh * 0.08
            payback = cost / max(avoided_usd, 1.0)
            actions.append(PlannerAction(
                priority=0,
                bus=bus,
                kind="volt_var_program",
                suggested_kw=size_kvar,
                est_cost_usd=cost,
                violation_hours_per_week=hrs_week,
                violation_hours_per_year=hours_year,
                worst_v_pu=worst,
                n_days_with_violation=int(row["days_with_violation"]),
                rationale=(
                    f"Overvoltage at Bus {bus} ({hrs_week:.0f} hr/week, "
                    f"worst {worst:.3f} pu) — consistent with midday PV backfeed. "
                    f"Enrol inverters in Volt-VAR curtailment; consider fixed-cap bank switching schedule."
                ),
                payback_years=payback if payback < 100 else None,
            ))

    # If the violations list is small, add the largest non-violating buses as
    # *monitor* entries so a planner always sees a broader watch-list.
    if len(actions) < 5:
        # Highest-load buses we haven't already flagged.
        watch_pool = sorted(
            [(b, kw) for b, kw in bus_nominal_kw.items() if b not in seen_buses],
            key=lambda x: x[1], reverse=True,
        )
        for bus, nominal_kw in watch_pool[: max(0, 5 - len(actions))]:
            if nominal_kw < 10:
                continue
            actions.append(PlannerAction(
                priority=0,
                bus=bus,
                kind="monitor",
                suggested_kw=0.0,
                est_cost_usd=0.0,
                violation_hours_per_week=0.0,
                violation_hours_per_year=0.0,
                worst_v_pu=1.0,
                n_days_with_violation=0,
                rationale=(
                    f"Bus {bus} ({nominal_kw:.0f} kW nominal) is currently inside the voltage band "
                    f"but is one of the largest loads on the feeder — track on the AMI dashboard "
                    f"and re-evaluate as EV adoption climbs."
                ),
                payback_years=None,
            ))

    # Rank: real violations first (by annualised hours × severity), then monitors.
    def _score(a: PlannerAction) -> float:
        if a.kind == "monitor":
            return -1.0  # always last
        severity = abs(a.worst_v_pu - 1.0) / 0.05
        return a.violation_hours_per_year * (1.0 + severity)
    actions.sort(key=_score, reverse=True)
    for i, a in enumerate(actions, start=1):
        a.priority = i
    return actions


def actions_to_df(actions: List[PlannerAction]) -> pd.DataFrame:
    if not actions:
        return pd.DataFrame(columns=["priority", "bus", "kind", "suggested_kw", "est_cost_usd",
                                     "violation_hours_per_week", "violation_hours_per_year",
                                     "worst_v_pu", "n_days_with_violation", "rationale", "payback_years"])
    return pd.DataFrame([asdict(a) for a in actions])


# ---- Multi-week trend ---------------------------------------------------- #

def weekly_trend(
    per_week_stats: List[dict],
) -> pd.DataFrame:
    """Multi-week trend DataFrame for the trend chart.

    Each input dict: {week_start, total_violation_hours, peak_kw, worst_v_pu, n_buses_hit}
    """
    df = pd.DataFrame(per_week_stats)
    if df.empty:
        return df
    df = df.sort_values("week_start").reset_index(drop=True)
    return df
