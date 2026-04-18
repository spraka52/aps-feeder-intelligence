"""OpenDSS power-flow validator.

Bridges the AI forecast (per-bus kW) to physics:
  1. Generate an OpenDSS deck for the IEEE 34-bus feeder.
  2. For each forecast hour, scale per-bus loads using the predicted multipliers.
  3. Run a snapshot power flow.
  4. Detect voltage violations (V outside [0.95, 1.05] p.u.) and thermal
     overloads (line current > NormalAmps).

Uses opendssdirect.py (the maintained EPRI Python interface). The problem
statement references py_dss_interface; both are thin wrappers over the same
OpenDSS engine and produce equivalent results — opendssdirect installs
cleanly on macOS without extra system packages.

Subprocess isolation
--------------------
The native OpenDSS engine (libdss_capi.dylib) has known stability issues on
macOS + Python 3.13 when a circuit is re-loaded multiple times in the same
process — the solution object hits an EXC_BAD_INSTRUCTION inside
`Set_Frequency`. To keep the Streamlit server alive across user interactions,
we therefore run the entire horizon solve in a separate `python -m
physics._solver_worker` subprocess (pure subprocess.Popen, not
multiprocessing.spawn — the latter tries to re-execute the parent script
which doesn't work under `streamlit run`). A native crash in the child
returns a non-zero exit code; the parent serves an empty result and the
UI stays responsive with a friendly warning banner.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.topology import SPOT_LOADS_KW, build_graph, write_opendss_deck

REPO = Path(__file__).resolve().parent.parent

REPO = Path(__file__).resolve().parent.parent
DECK_DIR = REPO / "data" / "opendss"


VMIN_PU = 0.95
VMAX_PU = 1.05


@dataclass
class HourResult:
    hour_index: int
    bus_voltage_pu: Dict[str, float]
    line_loading_pct: Dict[str, float]
    voltage_violations: List[Tuple[str, float]] = field(default_factory=list)
    thermal_overloads: List[Tuple[str, float]] = field(default_factory=list)
    converged: bool = True
    total_load_kw: float = 0.0
    total_losses_kw: float = 0.0
    # QSTS-only diagnostics (dict of element-name -> position/state)
    regulator_taps: Dict[str, int] = field(default_factory=dict)
    capacitor_states: Dict[str, int] = field(default_factory=dict)


def _ensure_deck() -> Path:
    fg = build_graph()
    deck = write_opendss_deck(DECK_DIR, fg)
    return deck


def _load_deck(deck: Path):
    import opendssdirect as dss
    if not dss.Basic.Start(0):
        raise RuntimeError("Failed to start OpenDSS engine")
    dss.Text.Command("ClearAll")
    dss.Text.Command(f'Redirect "{deck.as_posix()}"')


def _set_load_multipliers(load_mults: Dict[str, float]):
    """Scale each load by the per-bus multiplier."""
    import opendssdirect as dss
    for bus, mult in load_mults.items():
        name = f"LD_{bus}"
        # Re-edit the load element kW (and proportional kvar at pf 0.9)
        nominal_kw = SPOT_LOADS_KW.get(bus, 0.0)
        if nominal_kw <= 0:
            continue
        kw_eff = max(0.001, nominal_kw * mult)
        kvar_eff = kw_eff * 0.4843  # tan(acos(0.9))
        dss.Text.Command(f"Edit Load.{name} kW={kw_eff:.4f} kvar={kvar_eff:.4f}")


def _solve_snapshot() -> bool:
    import opendssdirect as dss
    dss.Solution.Solve()
    return dss.Solution.Converged()


def _collect_results(hour_index: int) -> HourResult:
    import opendssdirect as dss
    bus_v = {}
    bus_names = dss.Circuit.AllBusNames()
    for b in bus_names:
        dss.Circuit.SetActiveBus(b)
        v_mag_pu = dss.Bus.puVmagAngle()[::2]  # magnitudes only
        if v_mag_pu:
            v = float(np.mean([x for x in v_mag_pu if x > 0]))
            if not np.isnan(v) and v > 0:
                bus_v[b] = v

    line_loading = {}
    dss.Lines.First()
    for _ in range(dss.Lines.Count()):
        name = dss.Lines.Name()
        norm_amps = dss.Lines.NormAmps() or 1.0
        # Get per-phase currents
        dss.Circuit.SetActiveElement(f"Line.{name}")
        currents = dss.CktElement.CurrentsMagAng()[::2]
        i_max = float(max(currents[:len(currents)//2] or [0.0]))
        loading_pct = 100.0 * i_max / norm_amps if norm_amps > 0 else 0.0
        line_loading[name] = loading_pct
        dss.Lines.Next()

    # Exclude infrastructure buses that are not feeder customer buses:
    #  - sourcebus (the upstream slack)
    #  - <bus>r endpoints of regulators (internal nodes)
    EXCLUDE = {"sourcebus"}
    v_viol = [
        (b, v) for b, v in bus_v.items()
        if (v < VMIN_PU or v > VMAX_PU)
        and b not in EXCLUDE and not b.endswith("r")
    ]
    thermal = [(n, p) for n, p in line_loading.items() if p > 100.0]

    total_kw = abs(dss.Circuit.TotalPower()[0])
    total_losses_kw = dss.Circuit.Losses()[0] / 1000.0

    # Regulator tap positions (integer steps from neutral; +ve = boost)
    reg_taps: Dict[str, int] = {}
    if dss.RegControls.First() > 0:
        while True:
            name = dss.RegControls.Name()
            try:
                reg_taps[name] = int(dss.RegControls.TapNumber())
            except Exception:
                pass
            if dss.RegControls.Next() == 0:
                break

    # Capacitor on/off state (1 = at least one step in, 0 = all out)
    cap_states: Dict[str, int] = {}
    if dss.Capacitors.First() > 0:
        while True:
            name = dss.Capacitors.Name()
            try:
                steps = dss.Capacitors.States()
                cap_states[name] = int(any(s > 0 for s in steps))
            except Exception:
                pass
            if dss.Capacitors.Next() == 0:
                break

    return HourResult(
        hour_index=hour_index,
        bus_voltage_pu=bus_v,
        line_loading_pct=line_loading,
        voltage_violations=v_viol,
        thermal_overloads=thermal,
        converged=True,
        total_load_kw=float(total_kw),
        total_losses_kw=float(total_losses_kw),
        regulator_taps=reg_taps,
        capacitor_states=cap_states,
    )


def _run_horizon_in_process(per_hour_load_kw: np.ndarray, bus_order: List[str]) -> List[HourResult]:
    """In-process snapshot solve — each hour is independent. Used as a fallback
    when QSTS is disabled."""
    deck = _ensure_deck()
    _load_deck(deck)
    results: List[HourResult] = []

    nominal = np.array([SPOT_LOADS_KW.get(b, 0.0) for b in bus_order], dtype=float)
    safe_nom = np.where(nominal > 0, nominal, 1.0)

    for t in range(per_hour_load_kw.shape[0]):
        kw = per_hour_load_kw[t]
        mults = {bus_order[i]: float(kw[i] / safe_nom[i]) for i in range(len(bus_order)) if nominal[i] > 0}
        _set_load_multipliers(mults)
        ok = _solve_snapshot()
        if ok:
            res = _collect_results(t)
        else:
            res = HourResult(
                hour_index=t, bus_voltage_pu={}, line_loading_pct={},
                converged=False,
            )
        results.append(res)
    return results


def _run_horizon_qsts(per_hour_load_kw: np.ndarray, bus_order: List[str]) -> List[HourResult]:
    """Quasi-static time-series (QSTS) solve.

    Defines a Loadshape per load with the forecast multipliers, then advances
    OpenDSS one hour at a time in `daily` mode. Regulator tap and capacitor
    control actions accumulate across timesteps instead of resetting, so the
    voltage/current trajectory reflects how the network actually responds to
    a load curve rather than treating each hour as an independent puzzle.
    """
    import opendssdirect as dss
    deck = _ensure_deck()
    _load_deck(deck)

    nominal = np.array([SPOT_LOADS_KW.get(b, 0.0) for b in bus_order], dtype=float)
    safe_nom = np.where(nominal > 0, nominal, 1.0)
    T = per_hour_load_kw.shape[0]

    # 1) Define a Loadshape per load and bind it to the load's `daily` shape.
    for i, bus in enumerate(bus_order):
        if nominal[i] <= 0:
            continue
        mults = per_hour_load_kw[:, i] / safe_nom[i]
        mult_str = " ".join(f"{float(m):.5f}" for m in mults)
        dss.Text.Command(f"New Loadshape.LS_{bus} npts={T} interval=1.0 mult=({mult_str})")
        dss.Text.Command(f"Edit Load.LD_{bus} daily=LS_{bus}")

    # 2) Switch to daily mode, 1-hour steps, fresh state.
    dss.Text.Command("Set Mode=daily Number=1 Stepsize=1h Hour=0")
    dss.Text.Command("Set ControlMode=Static MaxControlIter=20")

    # 3) March one hour at a time, capturing results after each Solve.
    results: List[HourResult] = []
    for t in range(T):
        dss.Solution.Solve()
        if dss.Solution.Converged():
            res = _collect_results(t)
        else:
            res = HourResult(hour_index=t, bus_voltage_pu={}, line_loading_pct={}, converged=False)
        results.append(res)
    return results


def _hourresult_to_dict(r: HourResult) -> dict:
    return {
        "hour_index": r.hour_index,
        "bus_voltage_pu": r.bus_voltage_pu,
        "line_loading_pct": r.line_loading_pct,
        "voltage_violations": r.voltage_violations,
        "thermal_overloads": r.thermal_overloads,
        "converged": r.converged,
        "total_load_kw": r.total_load_kw,
        "total_losses_kw": r.total_losses_kw,
        "regulator_taps": r.regulator_taps,
        "capacitor_states": r.capacitor_states,
    }


def _dict_to_hourresult(d: dict) -> HourResult:
    return HourResult(**d)


def run_forecast_horizon(
    per_hour_load_kw: np.ndarray,
    bus_order: List[str],
    use_subprocess: bool = True,
    timeout_s: float = 60.0,
    mode: str = "qsts",
) -> List[HourResult]:
    """Run OpenDSS for each forecast hour.

    per_hour_load_kw: shape [T_out, N] — predicted kW per bus.
    bus_order:        list of length N matching the predictor.
    mode:             "qsts" (default, recommended) — daily-mode time-series
                       so regulator/cap controls evolve across hours.
                      "snapshot" — independent snapshot solve per hour
                       (kept for comparison / debugging).

    By default each call runs in a fresh `python -m physics._solver_worker`
    subprocess so a SIGILL inside the OpenDSS native engine never crashes
    the Streamlit server. On crash, timeout, or unpickling error we return
    one empty (non-converged) HourResult per hour so the dashboard keeps
    rendering with a friendly warning.
    """
    if not use_subprocess:
        if mode == "qsts":
            return _run_horizon_qsts(per_hour_load_kw, bus_order)
        return _run_horizon_in_process(per_hour_load_kw, bus_order)

    payload = pickle.dumps({
        "forecast_kw": np.ascontiguousarray(per_hour_load_kw, dtype=np.float64),
        "bus_order": list(bus_order),
        "mode": mode,
    })

    cmd = [sys.executable, "-m", "physics._solver_worker"]
    try:
        completed = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(REPO),
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("[opendss_runner] subprocess timed out", file=sys.stderr)
        return _empty_results(per_hour_load_kw.shape[0])

    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="replace")[-500:]
        print(
            f"[opendss_runner] subprocess exit={completed.returncode}; stderr tail:\n{err}",
            file=sys.stderr,
        )
        return _empty_results(per_hour_load_kw.shape[0])

    try:
        out = pickle.loads(completed.stdout)
    except Exception as e:
        print(f"[opendss_runner] failed to unpickle worker output: {e}", file=sys.stderr)
        return _empty_results(per_hour_load_kw.shape[0])

    return [_dict_to_hourresult(d) for d in out]


def _empty_results(n: int) -> List[HourResult]:
    return [
        HourResult(
            hour_index=t, bus_voltage_pu={}, line_loading_pct={},
            converged=False,
        )
        for t in range(n)
    ]


def summarize(results: List[HourResult]) -> dict:
    n_v = sum(len(r.voltage_violations) for r in results)
    n_t = sum(len(r.thermal_overloads) for r in results)
    worst_v = min(
        ((b, v, r.hour_index) for r in results for b, v in r.bus_voltage_pu.items()),
        key=lambda x: x[1],
        default=(None, None, None),
    )
    worst_load = max(
        ((n, p, r.hour_index) for r in results for n, p in r.line_loading_pct.items()),
        key=lambda x: x[1],
        default=(None, None, None),
    )
    return {
        "n_voltage_violations": n_v,
        "n_thermal_overloads": n_t,
        "worst_voltage": {"bus": worst_v[0], "pu": worst_v[1], "hour": worst_v[2]},
        "worst_loading": {"line": worst_load[0], "pct": worst_load[1], "hour": worst_load[2]},
        "n_hours": len(results),
        "avg_total_load_kw": float(np.mean([r.total_load_kw for r in results])) if results else 0.0,
        "avg_losses_kw": float(np.mean([r.total_losses_kw for r in results])) if results else 0.0,
    }


if __name__ == "__main__":
    # Quick smoke test: run baseline at unit multipliers
    fg = build_graph()
    bus_order = sorted(SPOT_LOADS_KW.keys())
    nominal = np.array([SPOT_LOADS_KW[b] for b in bus_order], dtype=float)
    horizon = np.tile(nominal, (3, 1))
    res = run_forecast_horizon(horizon, bus_order)
    s = summarize(res)
    print("OpenDSS smoke test:", s)
