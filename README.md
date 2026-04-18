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
| Physics | `physics/opendss_runner.py` | OpenDSS quasi-static time-series (QSTS) per 24-hour forecast horizon; regulator tap and capacitor switching states evolve across hours instead of resetting per snapshot. Detects voltage excursions outside [0.95, 1.05] pu, thermal overloads above NormAmps, and tracks regulator tap trajectories |
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

Live demo: **[aps-feeder.streamlit.app](https://aps-feeder.streamlit.app/)**

To redeploy your own:

1. Fork the repo to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → select your fork / `main` / `app.py` → Deploy.

The repo commits the small synthetic `.npz` datasets, the model checkpoint,
the cached NOAA/NSRDB parquets, and the OpenDSS deck — so the app works
**without any API keys at runtime**. To refresh real-data caches you only
need a free NREL API key set as the `NREL_API_KEY` environment variable
(or in Streamlit Cloud's *Settings → Secrets*: `NREL_API_KEY = "..."`).

---

## Data realism

The default mode (`weather_source="noaa"`) drives the load model with
**real Phoenix observations**:

- **Temperature**: hourly KPHX (Phoenix Sky Harbor) air temperature from
  **NOAA NCEI ISD-Lite** for Jun–Aug 2024. Peak 47.2°C, mean 37.1°C, with
  a real-life *42-day continuous heatwave* (Jun 20 – Jul 31 2024, all
  daily highs ≥ 41°C). Heatwave windows are auto-detected (≥ 3 consecutive
  days ≥ 41°C) instead of being hard-coded. No API key required —
  ISD-Lite is public bulk data.
- **Irradiance**: hourly GHI / DNI / DHI from **NREL NSRDB** (GOES
  Aggregated PSM v4.0.0) for Phoenix 2024, satellite-derived from cloud
  imagery. Average daytime GHI 528 W/m², peak 1076 W/m², ~6.5%
  cloud loss vs. clear-sky. Falls back to a NOAA-cloud-attenuated
  clear-sky model if no NSRDB key is configured.
- **Per-bus demand**: per-customer hourly load shapes from **NREL
  SMART-DS** (Austin TX, P1R substation, 2018) — public no-auth dataset
  hosted on the OEDI Open Energy Data Initiative S3 bucket. We sample
  8 residential + 5 commercial customers, then for each IEEE 34 bus
  draw a deterministic mix of customers sized to the bus's nominal kW
  (small residential pockets vs. commercial-dominated load centers).
  This injects realistic customer-class diversity: residential profiles
  peak in the evening, commercial profiles peak mid-morning, mixed
  pockets are flatter. A gentler Phoenix-specific HVAC overlay is
  applied on top so extreme heat still pushes loads above the embedded
  Austin baseline. Real APS AMI traces are not publicly distributed; the
  hackathon brief allows synthetic with documented assumptions.
- **EV stress**: NREL EVI-Pro style residential evening curve, scaled
  to add up to 35% of nominal at the evening peak.
- **PV offset**: behind-meter solar proportional to GHI / 1000 W/m².

All real-data feeds are cached as Parquet files under `data/noaa_cache/`,
`data/nsrdb_cache/`, and `data/smart_ds_cache/`, so cold deploys (e.g. on
Streamlit Cloud) work without runtime API calls or keys. The committed
dataset spans **three summers** (Jun-Aug 2024, 2025, 2026) — 2024 and
2025 use real NOAA observations, 2026 uses a synthetic projection since
real summer data isn't yet observable. To refresh from source:

```bash
python -m data.noaa_real --start 2024-06-01 --end 2024-08-31  # no key
NREL_API_KEY=xxxx python -m data.nsrdb_real \
    --start 2024-06-01 --end 2024-08-31 \
    --email you@example.com                                    # free key
python -m data.smart_ds --n_res 8 --n_com 5                   # no key
python -m data.synthesize --multi                              # build .npz
```

A full **synthetic** mode (`--source synthetic --customers synthetic`)
is preserved for tests and reproducibility.

---

## Model performance

Trained on real NOAA temperature + real NSRDB irradiance for Phoenix
Jun–Aug 2024; held-out validation on the last 20% of the window:

```
Overall   RMSE 19.3 kW  •  MAPE 18.9%
Heatwave  RMSE 17.7 kW  •  MAPE  ~17%
Normal    RMSE 19.7 kW  •  MAPE  ~19%
```

Heatwave error is intentionally surfaced as a separate metric so the model
is judged where it matters — the regime where the feeder is most stressed.
The model performs *better* during heatwaves because heat is a strong
signal: the relationship between temperature, time of day, and HVAC
load is more predictable than non-heatwave variability.

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
