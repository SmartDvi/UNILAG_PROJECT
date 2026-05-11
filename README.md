# 🇳🇬 Nigeria Retail Fuel Price Intelligence System

> **Uni Dissertation Project — Distinction Track**  
> An end-to-end data engineering and machine learning pipeline for forecasting Nigeria's retail petrol prices, detecting structural market breaks, and generating actionable intelligence for policymakers and MSMEs.

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10%2B-FFA500?style=flat-square&logo=duckdb&logoColor=white)](https://duckdb.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Kernel Map](#kernel-map)
- [Key Insights Implemented](#key-insights-implemented)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Replacing Synthetic Data](#replacing-synthetic-data)
- [Outputs](#outputs)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Policy Applications](#policy-applications)
- [Thesis Citation](#thesis-citation)
- [References](#references)
- [License](#license)

---

## Project Overview

Nigeria's retail fuel market underwent a historic structural transformation in May 2023 when the incoming Tinubu administration removed decades-old petroleum subsidies. This event — combined with the Dangote Refinery coming online in 2024 — shifted the market from a state-controlled import monopoly (NNPCL) to an **asymmetric duopoly** with volatile, market-reflective pricing.

This project builds a **production-grade forecasting and intelligence system** that:

- Engineers a clean analytical data warehouse from raw WFP price data using **SQL (DuckDB)**
- Detects structural market breaks using the **PELT change-point detection algorithm**
- Trains a **2-layer multivariate LSTM** with a custom trust-weighted loss function
- Generates a **3-month early-warning fuel price forecast** to help MSMEs hedge logistics costs
- Maps **regional price disparities** to support targeted government cash-transfer programmes
- Produces **fully interactive Plotly visualisations** for all analytical outputs

---

## Research Questions

1. **Duopoly Shift** — Can data distinguish price hikes caused by global crude shocks (external) from domestic refinery bottlenecks (internal)?
2. **MSME Vulnerability** — Which months trigger a "volatility squeeze" where MoM price surges erode MSME profit margins?
3. **Regional Disparity** — Which Nigerian states pay a persistent premium above the national average, and by how much?
4. **Structural Break** — Where does the data-driven regime boundary fall, and how does it differ from the policy date (May 2023)?
5. **Forecasting** — Can a trust-weighted, post-subsidy-only LSTM predict prices 30 days in advance with MAPE < 5%?

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | [WFP VAM Real-Time Energy Prices](https://dataviz.vam.wfp.org/economic_explorer/fuel-prices) |
| **Coverage** | Nigeria — January 2007 to April 2026 |
| **Granularity** | Monthly, by market (LGA level) |
| **Rows** | 17,168 |
| **Columns** | 55 (8 usable after null audit) |
| **Key columns** | `o/h/l/c_fuel_petrol_gasoline`, `trust_fuel_petrol_gasoline`, `inflation_fuel_petrol_gasoline`, `adm1_name`, `adm2_name`, `geo_id` |
| **Currency** | Nigerian Naira (₦ per litre) |

### Column Coverage Summary

| Column Group | Coverage | Decision |
|---|---|---|
| `o/h/l/c_fuel_petrol_gasoline` | 100% | ✅ Retain — core OHLC series |
| `trust_fuel_petrol_gasoline` | 100% | ✅ Retain — custom loss weight |
| `inflation_fuel_petrol_gasoline` | 94.8% | ✅ Retain — linear interpolation applied |
| `fuel_petrol_gasoline` (raw) | 7.7% | ⚠️ Use only for validation |
| `fuel_diesel / gas / kerosene` | 0% | 🗑️ Dropped |
| `fuel_petrol_gasoline_95_octane` | 0% | 🗑️ Dropped |
| `fuel_super_petrol` | 0% | 🗑️ Dropped |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (DuckDB SQL)                        │
│                                                                     │
│  CSV File  ──►  raw_fuel_prices  ──►  clean_fuel_prices             │
│                     (K1)                     (K2)                   │
│                                               │                     │
│                               ┌───────────────▼──────────────────┐  │
│                               │     engineered_features (K3)     │  │
│                               │  • OHLC derivatives              │  │
│                               │  • Lag features (1,3,6,12m)      │  │
│                               │  • Rolling stats (3m, 12m)       │  │
│                               │  • MoM / YoY % change            │  │
│                               │  • Cyclical month encoding       │  │
│                               │  • Regime flag (PELT)            │  │
│                               └───────────────┬──────────────────┘  │
└───────────────────────────────────────────────┼─────────────────────┘
                                                │
              ┌─────────────────────────────────┼───────────────────┐
              │         ANALYTICAL LAYER        │                   │
              │                                 ▼                   │
              │  K4: Duopoly Analysis    K5: MSME Volatility        │
              │  K6: Regional Disparity  K7: PELT Break Detection   │
              └─────────────────────────────────┬───────────────────┘
                                                │
              ┌─────────────────────────────────▼───────────────────┐
              │              ML LAYER (PyTorch)                     │
              │                                                     │
              │  K8: Sliding-window Dataset (6-month window)        │
              │  K9: 2-Layer LSTM + Trust-Weighted MSE Loss         │
              │      AdamW + ReduceLROnPlateau + Early Stopping     │
              │  K10: Evaluation + 3-Month Forward Forecast         │
              └─────────────────────────────────────────────────────┘
```

---

## Kernel Map

| Kernel | ID | Purpose | Key Output |
|---|---|---|---|
| K0 | `k0-install` / `k0-imports` | Environment, palette, Plotly theme | Global `apply_theme()` helper |
| K1 | `k1-ingest` / `k1-summary` | DuckDB ingestion, SQL schema audit | `raw_fuel_prices` table |
| K2 | `k2-null-audit` / `k2-clean-table` / `k2-interpolate` | NULL audit, clean table, interpolation | `clean_fuel_prices` table |
| K3 | `k3-features` | SQL window-function feature engineering | `engineered_features` table |
| K4 | `k4-duopoly` | Asymmetric duopoly analysis | Candlestick + dispersion chart |
| K5 | `k5-msme` | MSME volatility squeeze + PMI correlation | Spike detection chart |
| K6 | `k6-regional` | Regional price disparity | State bar chart + treemap |
| K7 | `k7-pelt` | PELT structural break detection | Regime segmentation chart |
| K8 | `k8-prepare` / `k8-dataset-arch` | Feature prep, Dataset class, LSTM architecture | `NigeriaFuelLSTM` model |
| K9 | `k9-loss-viz` / `k9-train` | Trust-weighted loss + training loop | `best_nigeria_lstm.pt` |
| K10 | `k10-eval` / `k10-forecast` | Evaluation + 3-month forecast | Forecast chart + CSVs |

---

## Key Insights Implemented

### Economic Insights (Policy & Strategy)

#### Insight 1 — Asymmetric Duopoly Shift (K4)
The 2024–2026 period represents a shift from an NNPCL import monopoly to an NNPCL + Dangote Refinery duopoly. Cross-market price standard deviation rises sharply post-2024, distinguishing external (Brent crude) shocks from internal (domestic supply) bottlenecks. This helps the **FCCPC** identify price manipulation vs. genuine cost pass-through.

#### Insight 2 — Volatility Squeeze on MSMEs (K5)
Months where MoM price change exceeds 5% are flagged as "MSME spike months." Rolling 6-month volatility charts define the planning horizon. When integrated with Stanbic IBTC PMI data, the model quantifies the **Cost-Push PMI Transmission Lag**. Replace the synthetic PMI series before thesis submission.

#### Insight 3 — Regional Price Disparity (K6)
SQL aggregation surfaces state-level price premiums above the national average. An interactive treemap and ranked bar chart reveal which geopolitical zones are persistently above average — candidate targets for **Social Safety Net cash-transfer programmes**.

---

### ML Engineering Insights (Technical)

#### Insight 4 — Structural Break Detection (K7)
The **PELT algorithm** (`ruptures`, RBF cost function, `pen=15`) detects change-points objectively rather than relying on the policy date. The detected boundary updates the `pelt_regime` column in DuckDB and is used to filter training data to the post-subsidy regime only.

#### Insight 5 — Multivariate LSTM with Lagged Exogenous Features (K8–K9)
A 6-month sliding window of 14 features feeds a 2-layer stacked LSTM:

```
Input features:  price_open, price_high, price_low, spread, momentum,
                 mom_pct, roll_mean_3m, roll_std_12m, fuel_inflation_mom,
                 trust_score, month_sin, month_cos,
                 usd_ngn_rate*, brent_crude*
                 (* replace with real CBN / FRED data)
```

Pump prices react to FX and crude prices with a **2–4 week lag** — the 6-month window captures this signal.

#### Insight 6 — Trust-Weighted Custom Loss Function (K9)

```python
class TrustWeightedMSELoss(nn.Module):
    """
    L = mean( trust_norm × (ŷ − y)² )
    trust_norm = trust / (mean(trust) + ε)

    trust = 1.0  →  verified NBS/WFP survey  →  full gradient
    trust = 0.3  →  spatially interpolated   →  30% gradient
    trust = 0.0  →  unknown provenance       →  no gradient
    """
```

This suppresses noise from spatially interpolated LGA observations, making the model learn primarily from **high-confidence survey data**.

---

## Tech Stack

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| SQL Engine | DuckDB | ≥ 0.10 | In-process data warehouse, all ETL |
| Data | Pandas | ≥ 2.0 | Interpolation, pandas↔DuckDB bridge |
| ML Framework | PyTorch | ≥ 2.0 | LSTM, custom loss, training loop |
| Change-point | Ruptures | ≥ 1.1 | PELT algorithm |
| Scaling | scikit-learn | ≥ 1.4 | MinMaxScaler, metrics |
| Visualisation | Plotly | ≥ 5.20 | All interactive charts |
| Static export | Kaleido | latest | PNG/SVG thesis figures |
| Notebook | JupyterLab | ≥ 4.0 | Interactive development |

---

## Project Structure

```
nigeria-fuel-price-intelligence/
│
├── 📓 nigeria_fuel_price.ipynb   # Main dissertation notebook
│
├── 📄 README.md                        # This file
│
├── 📁 data/
│   └── real-time-energy-prices-for-nigeria.csv   # WFP source data (add here)
│
├── 📁 outputs/
│   ├── best_nigeria_lstm.pt                      # Best model checkpoint
│   ├── nigeria_fuel_engineered_features.csv      # Full feature table export
│   └── nigeria_fuel_3m_forecast.csv              # 3-month forward forecast
│
├── 📁 figures/                                   # Auto-generated by notebook
│   ├── k4_duopoly_candlestick.png
│   ├── k5_msme_volatility.png
│   ├── k6_regional_disparity.png
│   ├── k7_structural_break.png
│   ├── k9_trust_loss_surface.png
│   └── k10_predicted_vs_actual.png
│
├── 📁 exogenous/                                 # Add real data here
│   ├── usd_ngn_rate.csv                          # CBN / ABOKIFX FX series
│   ├── brent_crude.csv                           # EIA / FRED DCOILBRENTEU
│   └── stanbic_pmi.csv                           # Stanbic IBTC PMI monthly
│
└── requirements.txt                              # Python dependencies
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone hhttps://github.com/SmartDvi/UNILAG_PROJECT.git
cd UNILAG_PROJECT
```

### 2. Create a virtual environment

```bash
uv init
uv venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

Or let the notebook install them automatically — the K0 cell runs `uv pip install` for all packages.

### 4. Add the dataset

Download `real-time-energy-prices-for-nigeria.csv` from the WFP VAM portal and place it in `data/`. Then update `DATA_PATH` in the K0 cell:

```python
DATA_PATH = Path("data/real-time-energy-prices-for-nigeria.csv")
```

### 5. Launch Jupyter

```bash
jupyter lab
```

Open `nigeria_fuel_price.ipynb` and run all cells top-to-bottom (`Kernel → Restart & Run All`).

---

## Replacing Synthetic Data

The notebook ships with synthetic placeholders for three exogenous variables. Replace them before thesis submission:

### USD/NGN Parallel Market Rate

```python
#  — CBN official rate
# Download from: https://ng.investing.com/currencies/usd-ngn-historical-data
usd_ngn = pd.read_csv("exogenous/usd_ngn_rate.csv", parse_dates=["date"])
lstm_df = lstm_df.merge(usd_ngn, left_on="price_date", right_on="date", how="left")

# Option B — pandas_datareader (FRED official rate proxy)
usd_ngn = pd.read_csv("exogenous/usd_ngn_rate.csv", parse_dates=["date"])
import pandas_datareader.data as pdr
usd_ngn = pdr.get_data_fred("NAEXKP01NGA652S", start="2023-01-01")
```

### Brent Crude Price (USD/bbl)


```python
# Download from: https://fred.stlouisfed.org/series/DCOILBRENTEU
usd_ngn = pd.read_csv("DCOILBRENTEU.csv", parse_dates=["date"])

# Alternatively 
import pandas_datareader.data as pdr
brent = pdr.get_data_fred("DCOILBRENTEU", start="2023-01-01")
brent = brent.resample("MS").mean()  # resample to monthly
```

### Stanbic IBTC PMI

Compile the monthly PMI press releases from [S&P Global / Stanbic IBTC](https://www.markiteconomics.com/Public/Release/PressReleases) into a CSV:

```
date,pmi
2020-01-01,52.3
2020-02-01,51.8
...
```

```python
pmi = pd.read_csv("exogenous/stanbic_pmi.csv", parse_dates=["date"])
vol_df = vol_df.merge(pmi, left_on="price_date", right_on="date", how="left")
```

---

## Outputs

After running all kernels the following files are produced:

| File | Description |
|---|---|
| `best_nigeria_lstm.pt` | Best model checkpoint (PyTorch `state_dict`) |
| `nigeria_fuel_engineered_features.csv` | Full 17k-row feature table with all engineered columns |
| `nigeria_fuel_3m_forecast.csv` | 3-month forward price forecast with MoM % and alert flag |

### Sample Forecast Output

```
════════════════════════════════════════════════════
  🔮  3-MONTH FORWARD FORECAST — EARLY WARNING SYSTEM
════════════════════════════════════════════════════
  May 2026  →  ₦  1,021.40   (+2.1%)  ✅ STABLE
  Jun 2026  →  ₦  1,058.30   (+5.9%)  ⚠️  SPIKE WARNING
  Jul 2026  →  ₦  1,044.10   (+4.4%)  ✅ STABLE
════════════════════════════════════════════════════
```

A "SPIKE WARNING" (MoM > 5%) gives MSMEs a 30-day window to hedge logistics costs or adjust pricing strategies before margin erosion occurs.

---

## Model Architecture

```
NigeriaFuelLSTM
═══════════════════════════════════════════════════
Input shape:   (batch, 6, 14)     ← 6-month window, 14 features

LSTM Layer 1:  hidden=128, dropout=0.25
LSTM Layer 2:  hidden=128, dropout=0.25
                      │
               Last time-step hidden state  →  (batch, 128)
                      │
               Dropout(0.25)
               Linear(128 → 64)
               ReLU
               Linear(64 → 1)
                      │
Output:        (batch, 1)         ← scaled price_close
═══════════════════════════════════════════════════

Loss:          TrustWeightedMSE  L = mean(trust_norm × (ŷ − y)²)
Optimiser:     AdamW  (lr=1e-3, weight_decay=1e-4)
Scheduler:     ReduceLROnPlateau  (factor=0.5, patience=6)
Early stop:    Patience = 12 epochs
Grad clipping: max_norm = 1.0
```

---

## Results

> Note: Final numeric results will appear after training on the full post-subsidy dataset with real exogenous data. The table below shows representative targets for a Distinction-grade submission.

| Metric | Target | Description |
|---|---|---|
| MAE | < ₦30/L | Mean absolute price error |
| RMSE | < ₦45/L | Penalises large outlier spikes |
| MAPE | < 5% | % error relative to actual price |
| R² | > 0.90 | Explained variance on test set |

---

## Policy Applications

| Stakeholder | Use Case | Kernel |
|---|---|---|
| **FCCPC** | Distinguish price manipulation from genuine cost pass-through | K4 |
| **MSMEs / VentureRoot** | 30-day early-warning system for logistics cost hedging | K5, K10 |
| **Federal Government / NNPCL** | Identify regions requiring targeted fuel subsidy interventions | K6 |
| **CBN / Monetary Policy** | Quantify fuel-driven cost-push inflation pressure | K5 |
| **NBS** | Validate reported price series against WFP trust-weighted estimates | K2, K9 |

---

## Thesis Citation

If you use this codebase or methodology in academic work, please cite as:

```bibtex
@mastersthesis{nigeria_fuel_intelligence_2026,
  author  = {[Your Name]},
  title   = {Nigeria Retail Fuel Price Intelligence System:
             A Multivariate LSTM Approach with Trust-Weighted Training
             for Post-Subsidy Market Forecasting},
  school  = {[Your Institution]},
  year    = {2026},
  type    = {Postgraduate Diploma Dissertation},
}
```

---

## References

| # | Citation |
|---|---|
| 1 | Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. |
| 2 | Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590–1598. |
| 3 | World Food Programme (2024). *WFP VAM Real-Time Monitoring — Energy Price Data, Nigeria*. VAM Food Security Analysis. |
| 4 | National Bureau of Statistics Nigeria (2026). *Petrol Price Watch*. NBS. |
| 5 | Stanbic IBTC Bank (2026). *Nigeria PMI Monthly Report*. S&P Global Market Intelligence. |
| 6 | Dangote Group (2024). *Dangote Petroleum Refinery — Commercial Operations Update*. |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for academic distinction. Designed for real-world policy impact.

**Nigeria Fuel Price Intelligence System** ·  Dissertation · 2026

</div>