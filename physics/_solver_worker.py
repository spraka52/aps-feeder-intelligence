"""Standalone OpenDSS solver process.

Reads a pickled (forecast_kw_array, bus_order) tuple from stdin, solves
each forecast hour with OpenDSS, and writes a pickled list-of-dicts to
stdout. Errors and crashes are isolated to this process — the parent
(Streamlit or CLI) only sees a non-zero exit code, which it converts to a
graceful empty result.

Run as:  python -m physics._solver_worker
(intended to be spawned by physics/opendss_runner.run_forecast_horizon)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Ensure the project root is on the path even if launched from elsewhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    payload = pickle.loads(sys.stdin.buffer.read())
    arr = payload["forecast_kw"]
    bus_order = payload["bus_order"]

    from physics.opendss_runner import _run_horizon_in_process, _hourresult_to_dict
    results = _run_horizon_in_process(arr, bus_order)
    out = [_hourresult_to_dict(r) for r in results]
    sys.stdout.buffer.write(pickle.dumps(out))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
