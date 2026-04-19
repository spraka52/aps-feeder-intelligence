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
- **Per-bus demand**: hourly aggregate electricity-demand profiles from
  **NREL ResStock** (4 residential building types) and **NREL ComStock**
  (5 commercial building types) for **ASHRAE climate zone 2B** — the
  hot-dry zone Phoenix sits in. Public no-auth datasets on the OEDI
  S3 bucket. Each profile is the per-unit (per-home / per-1000-sqft)
  hourly aggregate across the entire Maricopa-relevant building stock,
  with all end-uses summed. For each IEEE 34 bus we deterministically
  draw a customer mix sized to the bus's nominal kW: small buses are
  pure single-family-detached, medium buses mix residential + small
  office, large pockets become commercial-dominated (medium office +
  retail + warehouse). A gentler Phoenix-specific HVAC overlay still
  rides on top so out-of-distribution heatwaves push loads above the
  ResStock baseline. Real APS AMI traces are not publicly distributed;
  the hackathon brief allows synthetic with documented assumptions.
- **EV stress**: NREL EVI-Pro style residential evening curve, scaled
  to add up to 35% of nominal at the evening peak.
- **PV offset**: behind-meter solar proportional to GHI / 1000 W/m².

All real-data feeds are cached as Parquet files under `data/noaa_cache/`,
`data/nsrdb_cache/`, `data/resstock_cache/`, and `data/smart_ds_cache/`,
so cold deploys (e.g. on Streamlit Cloud) work without runtime API calls
or keys. The committed dataset spans **three summers** (Jun-Aug 2024,
2025, 2026) — 2024 and 2025 use real NOAA observations, 2026 uses a
synthetic projection since real summer data isn't yet observable. To
refresh from source:

```bash
python -m data.noaa_real    --start 2024-06-01 --end 2024-08-31  # no key
NREL_API_KEY=xxxx python -m data.nsrdb_real \
    --start 2024-06-01 --end 2024-08-31 \
    --email you@example.com                                       # free key
python -m data.resstock_real                                     # no key
python -m data.smart_ds --n_res 8 --n_com 5                      # no key (Austin fallback)
python -m data.synthesize --multi --customers resstock           # build .npz
```

A full **synthetic** mode (`--source synthetic --customers synthetic`)
is preserved for tests and reproducibility.

---

## Model performance

Trained on real NOAA temperature + real NSRDB irradiance + ResStock /
ComStock Phoenix building-stock load shapes, multi-year (2024-2026
summers, 6624 hourly samples); held-out validation on the last 20%:

```
Overall   RMSE 6.3 kW  •  MAPE 13.4%
Heatwave  RMSE 6.3 kW  •  MAPE ~13%
Normal    RMSE 6.4 kW  •  MAPE ~14%
```

Switching from Austin-based SMART-DS profiles to Phoenix-specific
ResStock/ComStock data dropped MAPE from 34% → 13% while keeping the
network architecture unchanged — the bigger win is that the model
now learns *Phoenix* HVAC patterns (climate zone 2B), not Austin's.

The training script writes `models/checkpoints/training_report.json` with
the full history; the dashboard renders it under "Model performance".

---

## Why this model architecture? (and why not alternatives)

We chose **GraphSAGE → GRU → Linear** because the forecasting target is
*spatio-temporal*: every bus has its own 24-hour load curve, but no bus is
electrically isolated — voltage sags propagate, EV adoption tends to
cluster, and a substation transformer constrains the entire downstream
radial. The architecture has to capture both axes.

| Candidate | What we'd lose | Why GraphSAGE+GRU wins |
| --- | --- | --- |
| **Plain MLP / per-bus regression** | Throws away the feeder graph entirely. Each bus is forecast in isolation, ignoring that bus 890's evening EV peak shifts neighbour 888's voltage. | GraphSAGE aggregates neighbour features per layer, so each bus's prediction sees its electrical neighbourhood. |
| **LightGBM / XGBoost per bus** | Same problem as MLP — no spatial information. Also: gradient-boosted trees can't share parameters across buses, so a small bus with sparse data can't borrow signal from a similar large bus. | GraphSAGE shares weights across all 34 buses, which is critical when each bus has only ~6.5k hourly samples. |
| **LSTM (no graph)** | Captures the temporal axis but still per-bus. ~2× the parameters of GRU for marginal accuracy gain on 24-hour horizons (LSTM excels at much longer sequences). | GRU is the right capacity for diurnal cycles + lagged HVAC response. |
| **Transformer / TFT** | Strong on long horizons with lots of data. We have 6,624 samples — Transformers overfit hard at this scale. Also adds 5–10× training cost for no clear win on 24-hour load. | GraphSAGE+GRU has only ~27k parameters and trains in ~12 epochs (≈12 minutes CPU). |
| **Pure physics (OpenDSS only)** | OpenDSS solves the *power flow* given loads — it doesn't *forecast* loads. We need both: a forecaster (this model) and a validator (OpenDSS). | This is exactly the role split we built. |

**Bottom line:** GraphSAGE+GRU is the smallest model that respects both the
feeder topology and the temporal dynamics of load, and at 27k parameters
it's lightweight enough to run in the browser tab on Streamlit Cloud.

---

## How we measure success (and why these metrics)

We report three numbers in the dashboard's *Model performance* card:

| Metric | What it tells you | Why we picked it |
| --- | --- | --- |
| **MAPE** (mean absolute % error) | Forecast error normalised by the bus's actual load. A 5 kW miss matters at a 50 kW bus and is noise at a 500 kW bus — MAPE puts both on the same scale. | Industry standard for distribution-load forecasting. EPRI / NREL day-ahead benchmarks land around 8–15% MAPE; ISO-NE day-ahead system-load is ≈2% MAPE *on the whole system* but degrades sharply at the feeder level. **Our 13.4% is competitive with the published feeder-level state of the art.** |
| **RMSE** (kW) | Absolute error in the unit operators care about. Useful for sizing — "the model can be off by ~6 kW per bus per hour." | Operators size batteries and capacitor banks in kW, not in percent. |
| **Heatwave-vs-normal split** | Reveals whether the model breaks down on the days that matter most (Phoenix peaks happen during 41 °C+ heatwaves). | A model with great average MAPE but terrible heatwave MAPE would be useless to APS. We explicitly checked: 13% heatwave vs 14% normal — the model holds up under stress. |

**Decisions we made because of these metrics:**

- **Switched from Austin-based SMART-DS to Phoenix-specific ResStock/ComStock load shapes** when MAPE on heatwave days was 34% — the Austin profiles weren't learning Phoenix's HVAC patterns. New MAPE: 13%.
- **Clamped negative model outputs to zero** after seeing rare slightly-negative kW predictions hurt RMSE; a load forecast can't physically be negative.
- **Added year-in-window encoding** so 2024-2026 trends don't leak across folds — early experiments showed val MAPE jumped 4% without it.

We deliberately do *not* report:
- *R² alone* — meaningless on highly auto-correlated time series.
- *Single-bus MAPE* — APS cares about the worst-case bus, not the average. The dashboard surfaces worst-case behaviour through the *Bus × Day heatmap* and the *Capital Action Plan* instead.

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
