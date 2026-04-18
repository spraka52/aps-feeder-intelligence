# APS Spatio-Temporal Feeder Intelligence

A proof-of-concept spatio-temporal AI system for an APS-style distribution
feeder. It forecasts feeder-level load 24 hours ahead, stress-tests the
network under extreme heat and EV evening-peak growth, validates every
forecast hour with a real OpenDSS power-flow solution, and turns the result
into a prioritized action list a control-room operator could act on.

Built for the **APS / ASU AI for Energy** hackathon track.

---

## What it does

| Layer | Component | Output |
| --- | --- | --- |
| Topology | `data/topology.py` (NetworkX + OpenDSS) | IEEE 34-bus radial feeder, two voltage regulators, in-line 24.9/4.16 kV transformer, geographic coordinates on a Phoenix footprint |
| Time-series | `data/synthesize.py` | Hourly Phoenix-realistic temperature, NSRDB-style GHI, per-bus residential demand with HVAC sensitivity, EV evening-peak shape (NREL EVI-Pro inspired), behind-meter PV |
| AI model | `models/graphsage_gru.py` | **GraphSAGE → GRU → linear head** in PyTorch Geometric. ~27k learned parameters. Inputs: per-bus load, temperature, irradiance, hour-of-day (sin/cos), EV-growth %, baseline-kW. Output: 24-hour per-bus kW forecast |
| Physics | `physics/opendss_runner.py` | OpenDSS snapshot power flow per forecast hour; detects voltage excursions outside [0.95, 1.05] pu and line currents above NormAmps |
| Decisions | `decisions/action_engine.py` | Aggregates violations into ranked operator actions (battery dispatch, Volt-VAR, deferrable-load shed) with sized recommendations |
| UI | `app.py` (Streamlit) | Side-by-side baseline vs. stress map, time-series charts, Action Center |

The forecast model is a **real trainable network with learned weights**,
not a prompt-only LLM wrapper.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (baseline + stress scenarios, ~92 days hourly)
python -m data.synthesize --days 92

# 2. Train the GraphSAGE+GRU forecaster (CPU, ~1 minute / epoch)
python -m models.train --epochs 12

# 3. Run the end-to-end pipeline (no UI, prints KPIs + actions)
python -m scripts.run_pipeline

# 4. Launch the dashboard
streamlit run app.py
```

A trained checkpoint is committed under `models/checkpoints/graphsage_gru.pt`
and the synthetic datasets under `data/synthetic/` so the dashboard can be
launched immediately without retraining.

## Deploy to Streamlit Community Cloud

This repo is set up for one-click deploy:

1. Push this repo to a public GitHub repo (already true if you cloned it).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo / branch / `app.py`.
4. Streamlit Cloud reads `requirements.txt` (Python 3.13 from `runtime.txt`),
   builds the env, and serves the dashboard at a public URL.

The repo intentionally commits the small synthetic `.npz` datasets and the
112 KB model checkpoint so cold deploys work without running training.

---

## Data realism

Synthetic data is used so the pipeline runs end-to-end with one command,
but every driver is climatologically motivated:

- **Temperature**: diurnal cycle peaking at ~16:00 local, summer high
  39–46°C with two ~7-day heatwave events bumping +5 to +8°C.
- **Irradiance**: clear-sky GHI from solar geometry at lat 33.45°N,
  modulated by an AR(1) cloud factor.
- **Demand**: HVAC cooling sensitivity grows quadratically above 24°C;
  per-bus phase shifts simulate geographic variation; weekend / holiday
  factors applied.
- **EV stress**: NREL EVI-Pro style residential evening curve, scaled
  to add up to 35% of nominal at the evening peak.
- **PV offset**: behind-meter solar proportional to GHI / 1000 W/m².

Real NOAA, NREL NSRDB, and EVI-Pro feeds can be dropped in by replacing
the `synth_*` functions with API loaders. Inputs and outputs are all
plain `np.ndarray`s so the rest of the pipeline is unaffected.

---

## Model performance

Held-out validation (last 20% of the 92-day window):

```
Overall   RMSE ~30 kW  •  MAE ~10 kW  •  MAPE ~17%
Heatwave  RMSE ~37 kW  •  MAE ~16 kW  •  MAPE ~22%
Normal    RMSE ~28 kW  •  MAE ~9 kW   •  MAPE ~16%
```

Heatwave error is intentionally surfaced as a separate metric so the model
is judged where it matters — the regime where the feeder is most stressed.

The training script writes `models/checkpoints/training_report.json` with
the full history; the dashboard renders it under "Model performance".

---

## Layout

```
APS/
├── app.py                          # Streamlit dashboard
├── data/
│   ├── topology.py                 # IEEE 34-bus topology (NetworkX + DSS deck writer)
│   ├── synthesize.py               # Arizona-realistic time-series synthesizer
│   ├── synthetic/                  # Generated .npz datasets
│   └── opendss/                    # Generated .dss decks
├── models/
│   ├── graphsage_gru.py            # GraphSAGE + GRU model
│   ├── dataset.py                  # Sliding-window torch Dataset
│   ├── train.py                    # Training loop + metrics
│   ├── predict.py                  # Inference wrapper
│   └── checkpoints/                # Saved weights + training report
├── physics/
│   └── opendss_runner.py           # opendssdirect.py power-flow validator
├── decisions/
│   └── action_engine.py            # Ranked operator actions
├── scripts/
│   └── run_pipeline.py             # CLI smoke test
├── requirements.txt
└── README.md
```

---

## Notes on the OpenDSS interface

The brief mentions `py_dss_interface`. We use **`opendssdirect.py`**
(EPRI-maintained) because it installs cleanly on macOS via pip without
extra system packages. Both are thin wrappers over the same OpenDSS
engine and produce equivalent physics. Swap in `py_dss_interface` by
replacing the import in `physics/opendss_runner.py`.

---

## What "decision-ready" means here

Each violation is grouped by location and time, scored by how far out of
bounds it is and how persistent it is, and turned into a sized
recommendation a dispatcher could act on:

> **[Priority 1] undervoltage @ Bus 890** — sev 1.38, 14 hours.
> Bus 890 dipped to 0.917 pu (limit 0.95). **Dispatch battery / cap
> bank at Bus 890 ≈ 56 kW** to close the voltage gap.

The map highlights flagged buses with a black ✕ so the spatial pattern
is obvious at a glance.

---

## License

Built for the APS / ASU AI for Energy hackathon.
