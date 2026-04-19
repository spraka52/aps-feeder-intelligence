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
    # Per-phase voltages: bus -> [V_A, V_B, V_C] in pu (NaN for phases not present)
    # Lets the dashboard surface single-phase vs three-phase imbalance — a real
    # planner needs to know whether overvoltage at Bus 820 is on Phase A only
    # (re-balance phases) vs all three phases (Volt-VAR / cap bank).
    bus_voltage_per_phase: Dict[str, List[float]] = field(default_factory=dict)


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
    bus_v_phase: Dict[str, List[float]] = {}
    bus_names = dss.Circuit.AllBusNames()
    for b in bus_names:
        dss.Circuit.SetActiveBus(b)
        v_mag_pu = dss.Bus.puVmagAngle()[::2]  # magnitudes only
        if v_mag_pu:
            valid = [x for x in v_mag_pu if x > 0]
            v = float(np.mean(valid)) if valid else float("nan")
            if not np.isnan(v) and v > 0:
                bus_v[b] = v
                # Pad to 3 phases with NaN for missing phases (single-phase laterals).
                phase_v = [float(x) if x > 0 else float("nan") for x in v_mag_pu[:3]]
                while len(phase_v) < 3:
                    phase_v.append(float("nan"))
                bus_v_phase[b] = phase_v

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
        bus_voltage_per_phase=bus_v_phase,
    )


def _add_injections(inject_kw: Dict[str, float]):
    """Inject real-power generators at specified buses (counterfactual mitigation).

    Each entry maps bus -> kW of real-power injection (e.g., a battery
    discharging to support voltage). We model this as an OpenDSS Generator
    with constant kW output. Negative kW = absorption (rare).
    """
    if not inject_kw:
        return
    import opendssdirect as dss
    for bus, kw in inject_kw.items():
        if abs(kw) < 0.01:
            continue
        kv = 4.16 if bus in {"888", "890"} else 24.9
        # Generator at unity PF — pure real-power support.
        dss.Text.Command(
            f"New Generator.WHATIF_{bus} bus1={bus} phases=3 conn=wye "
            f"kv={kv} kw={kw:.3f} kvar=0 model=1 status=fixed"
        )


def _disable_elements(elements: List[str]):
    """Open / disable the named elements (N-1 contingency)."""
    if not elements:
        return
    import opendssdirect as dss
    for el in elements:
        try:
            dss.Text.Command(f"Open {el}")
        except Exception:
            pass


def _run_horizon_in_process(
    per_hour_load_kw: np.ndarray,
    bus_order: List[str],
    inject_kw: Dict[str, float] | None = None,
    disabled_elements: List[str] | None = None,
) -> List[HourResult]:
    """In-process snapshot solve — each hour is independent. Used as a fallback
    when QSTS is disabled."""
    deck = _ensure_deck()
    _load_deck(deck)
    _add_injections(inject_kw or {})
    _disable_elements(disabled_elements or [])
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


def _run_horizon_qsts(
    per_hour_load_kw: np.ndarray,
    bus_order: List[str],
    inject_kw: Dict[str, float] | None = None,
    disabled_elements: List[str] | None = None,
) -> List[HourResult]:
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
    _add_injections(inject_kw or {})
    _disable_elements(disabled_elements or [])

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
        "bus_voltage_per_phase": r.bus_voltage_per_phase,
    }


def _dict_to_hourresult(d: dict) -> HourResult:
    return HourResult(**d)


def run_forecast_horizon(
    per_hour_load_kw: np.ndarray,
    bus_order: List[str],
    use_subprocess: bool = True,
    timeout_s: float = 60.0,
    mode: str = "qsts",
    inject_kw: Dict[str, float] | None = None,
    disabled_elements: List[str] | None = None,
) -> List[HourResult]:
    """Run OpenDSS for each forecast hour.

    per_hour_load_kw:  shape [T_out, N] — predicted kW per bus.
    bus_order:         list of length N matching the predictor.
    mode:              "qsts" (default) — daily-mode QSTS, regulators evolve.
                       "snapshot" — independent snapshot per hour.
    inject_kw:         optional bus -> kW dict. Each entry adds a constant-kW
                       Generator at that bus, modelling a battery discharge or
                       a Volt-VAR program's net real-power offset for
                       counterfactual "what-if I deploy this action?" runs.
    disabled_elements: optional list of OpenDSS elements to Open before solving
                       (e.g., ["Transformer.Reg2", "Line.L_832_858"]) for N-1
                       contingency analysis.

    By default each call runs in a fresh `python -m physics._solver_worker`
    subprocess so a SIGILL inside the OpenDSS native engine never crashes
    the Streamlit server. On crash, timeout, or unpickling error we return
    one empty (non-converged) HourResult per hour so the dashboard keeps
    rendering with a friendly warning.
    """
    if not use_subprocess:
        if mode == "qsts":
            return _run_horizon_qsts(per_hour_load_kw, bus_order,
                                      inject_kw=inject_kw,
                                      disabled_elements=disabled_elements)
        return _run_horizon_in_process(per_hour_load_kw, bus_order,
                                        inject_kw=inject_kw,
                                        disabled_elements=disabled_elements)

    payload = pickle.dumps({
        "forecast_kw": np.ascontiguousarray(per_hour_load_kw, dtype=np.float64),
        "bus_order": list(bus_order),
        "mode": mode,
        "inject_kw": dict(inject_kw or {}),
        "disabled_elements": list(disabled_elements or []),
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


def compute_hosting_capacity(
    bus_order: List[str],
    nominal_kw_per_bus: Dict[str, float],
    test_pv_kw: float = 200.0,
) -> Dict[str, float]:
    """Estimate per-bus PV hosting headroom (kW until any voltage hits 1.05 pu).

    APS is required by the Arizona Corporation Commission to publish
    hosting-capacity maps for every feeder. This function exploits the
    near-linear V vs PV-injection relationship at a single bus to compute
    headroom in 2 N + 1 OpenDSS solves rather than a brute-force sweep.

    Returns a dict {bus -> headroom_kw}. Buses already over 1.05 pu return 0.
    """
    deck = _ensure_deck()
    _load_deck(deck)
    nominal = np.array([nominal_kw_per_bus.get(b, 0.0) for b in bus_order], dtype=float)
    safe_nom = np.where(nominal > 0, nominal, 1.0)

    # Baseline solve at nominal load — this is the "starting point" for V.
    mults = {b: 1.0 for b in bus_order if nominal_kw_per_bus.get(b, 0.0) > 0}
    _set_load_multipliers(mults)
    _solve_snapshot()
    base = _collect_results(0)
    v_base = base.bus_voltage_pu

    import opendssdirect as dss
    headroom: Dict[str, float] = {}
    for bus in bus_order:
        if bus not in v_base:
            continue
        v0 = v_base[bus]
        # Inject `test_pv_kw` of PV (negative load) at this bus, re-solve,
        # measure ΔV, then back out headroom to 1.05 pu by linear extrapolation.
        # Unique generator names per iteration so OpenDSS doesn't warn about
        # duplicate element definitions (which the CFFI binding promotes to error).
        gen_name = f"HC_{bus}"
        dss.Text.Command(
            f"New Generator.{gen_name} bus1={bus} phases=3 conn=wye "
            f"kv={4.16 if bus in {'888','890'} else 24.9} "
            f"kw={test_pv_kw:.3f} kvar=0 model=1 status=fixed"
        )
        dss.Solution.Solve()
        ok = dss.Solution.Converged()
        if ok:
            r = _collect_results(0)
            v_new = r.bus_voltage_pu.get(bus, v0)
            dV = v_new - v0
            slope = dV / max(test_pv_kw, 1.0)  # pu per kW
            if slope <= 0:  # injection didn't raise voltage — unbounded headroom
                headroom[bus] = float("inf")
            else:
                hr_kw = max(0.0, (VMAX_PU - v0) / slope)
                # Cap at a sensible 5 MW per single-phase tap
                headroom[bus] = float(min(hr_kw, 5000.0))
        else:
            headroom[bus] = 0.0
        # Disable the test generator before moving to the next bus
        try:
            dss.Text.Command(f"Disable Generator.{gen_name}")
        except Exception:
            pass
    return headroom


def _empty_results(n: int) -> List[HourResult]:
    return [
        HourResult(
            hour_index=t, bus_voltage_pu={}, line_loading_pct={},
            converged=False,
        )
        for t in range(n)
    ]


def run_hosting_capacity_subprocess(
    bus_order: List[str],
    nominal_kw_per_bus: Dict[str, float],
    test_pv_kw: float = 200.0,
    timeout_s: float = 90.0,
) -> Dict[str, float]:
    """Run hosting-capacity computation in a subprocess (for crash isolation)."""
    payload = pickle.dumps({
        "mode": "hosting_capacity",
        "bus_order": list(bus_order),
        "nominal_kw_per_bus": dict(nominal_kw_per_bus),
        "test_pv_kw": float(test_pv_kw),
    })
    cmd = [sys.executable, "-m", "physics._solver_worker"]
    try:
        completed = subprocess.run(
            cmd, input=payload, capture_output=True,
            timeout=timeout_s, cwd=str(REPO), check=False,
        )
    except subprocess.TimeoutExpired:
        print("[hosting_capacity] subprocess timed out", file=sys.stderr)
        return {}
    if completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"[hosting_capacity] subprocess exit={completed.returncode}; stderr:\n{err}",
              file=sys.stderr)
        return {}
    try:
        return pickle.loads(completed.stdout)
    except Exception as e:
        print(f"[hosting_capacity] failed to unpickle: {e}", file=sys.stderr)
        return {}


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
