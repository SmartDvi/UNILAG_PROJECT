# Nigeria Retail Petrol Prices, 2007–2026

Regime analysis, regional disparity and short-term forecasting of retail petrol prices in Nigeria, built as an end-to-end pipeline: SQL data engineering in DuckDB, descriptive and change-point analysis, and a forecasting comparison of an LSTM against statistical benchmarks.

The full analysis, with code, figures and interpretation, is in [nigeria_fuel_price.ipynb](nigeria_fuel_price.ipynb). An interactive, multipage [dashboard](#dashboard) lets readers explore the results themselves.

## Dashboard demo:
https://github.com/user-attachments/assets/153574ed-f22d-45a6-a2ac-d4dbaa0aaeb2

## Key findings

1. **The data is regional, not national.** The source file has 67 individual markets in 14 states, and 79% of them are in Borno, Yobe and Adamawa. It also contains 7 aggregate series that must be excluded to avoid double-counting. All national figures use a *state-balanced* index, in which each state counts once.
2. **Subsidy removal changed price behaviour.** From Jun 2023 to Aug 2024 the index rose 5.1% per month on average (0.6% under regulation), and 8 of 15 months had a rise above 5%. After domestic refining began (Sep 2024), growth slowed to 2.1% per month and only 2 of 20 months had such a rise.
3. **States are drifting apart.** The cross-state coefficient of variation averaged 7.5% under regulation and 15.1% since Sep 2024, peaking near 30% in early 2026. Borno's premium over the index rose from +6.8% to +22.2%.
4. **Data-driven breaks are March 2016, January 2022 and December 2023** (PELT on log prices, stable across penalties 0.5–2). The June 2023 policy date is not itself a break in this series.
5. **Simple models are as good as, or better than, the LSTM.** On the held-out test period (May 2024 – Apr 2026):

   | Model | Test MAPE | Test MAE (₦/L) | Better than naive? (Diebold–Mariano) |
   |---|---|---|---|
   | Naive (no change) | 2.27% | 12.64 | – |
   | Drift | 2.54% | 13.45 | no (p = 0.71) |
   | **ARIMA(1,0,0) on returns** | **1.94%** | **10.79** | **yes (p = 0.016)** |
   | Ridge regression | 2.09% | 11.55 | no (p = 0.22) |
   | LSTM ensemble (5 seeds) | 2.74% | 15.61 | no, slightly worse (p = 0.054) |

6. **The trust-weighted loss has no measurable effect.** The trust score only ranges from 9.7 to 10 (on a 0–10 scale), so the weights differ by less than 1%.

**Important caveat:** the prices are model-based estimates. After subsidy removal they sit well below reported official pump prices; for example, Lagos is ₦256/L in June 2023 against roughly ₦488–568/L reported. Treat naira levels as levels of this series, not as pump prices. See section K2 of the notebook.

---

## Project structure

```
UNI_LAG_PROJECT/
├── nigeria_fuel_price.ipynb     # the analysis (K0–K10), executed with outputs
├── main.py                      # runs the notebook (--test) or the dashboard (--dashboard)
├── dashboard/                   # Dash + Dash Mantine Components + Dash AG Grid app
│   ├── app.py                   #   app shell, navigation, entry point
│   ├── data.py                  #   loads outputs/tables, shared calculations
│   ├── components.py            #   cards, insight callouts, date controls, grids
│   └── pages/                   #   one module per page (8 pages)
├── src/fuel_forecast/           # reusable code imported by the notebook
│   ├── config.py                #   paths, policy dates, split dates, seeds
│   ├── exogenous.py             #   USD/NGN and Brent loaders (monthly)
│   ├── features.py              #   returns, sliding windows, chronological split
│   ├── models.py                #   LSTM, trust-weighted loss, training loop
│   ├── baselines.py             #   naive, drift, ARIMA, ridge benchmarks
│   ├── evaluation.py            #   metrics, Diebold–Mariano test
│   └── plotting.py              #   figure style and saving
├── tests/
│   ├── test_pipeline.py         # unit tests (date parsing, leakage, loss, metrics)
│   └── test_dashboard.py        # every dashboard callback runs for many control settings
├── data/
│   ├── raw/                     # source files (committed)
│   └── processed/               # DuckDB database (generated, git-ignored)
├── outputs/
│   ├── figures/                 # all figures (PNG)
│   ├── tables/                  # result tables (CSV)
│   └── models/                  # LSTM weights, one file per seed
├── MODEL_CARD.md
├── REPRODUCIBILITY.md
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
python main.py --test                     # run unit tests, then execute the notebook
```

Then start the dashboard with `python main.py --dashboard` and open <http://127.0.0.1:8050>.

To work interactively, run `jupyter lab` **from the project root** and open the notebook. The full run takes about one minute on a laptop CPU. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details.

## Dashboard

A multipage web app built with **Dash**, **Dash Mantine Components** (layout, inputs, cards) and **Dash AG Grid** (every table: sortable, filterable, paginated, CSV export). It reads the tables the notebook writes to `outputs/tables/`, so run the notebook first.

```bash
cd dashboard
python app.py --dashboard            # or: python -m dashboard.app [--debug] [--port 8050]
```

| Page | What you can do | Controls |
|---|---|---|
| **Overview** | KPI cards, the national index with low–high band and policy periods, optional USD/NGN and Brent overlays, period summary | date-range calendar, quick-range presets, driver multiselect, log-scale switch |
| **Market Explorer** | Compare states or individual markets as prices, rebased indices, premiums or YoY change; OHLC candlesticks; change ranking | date range, states/markets multiselects, measure select |
| **Regional Disparity** | Market map, state ranking against the regulated-era baseline, state × year premium heatmap | date range, metric select, highlight-states multiselect |
| **Price Shocks** | Shock months above an adjustable threshold, share of markets hit, rolling volatility, planning rule of thumb | date range, series select, threshold slider, window select |
| **Structural Breaks** | Live PELT change-point detection with penalty-sensitivity chart and regime statistics | series, cost function, penalty, minimum segment length |
| **Price Drivers** | Petrol vs exchange-rate/Brent changes: scatter with fits per period, correlation by lead time, rolling correlation, full r/p/β table | date range, series, driver, lead, periods multiselect |
| **Forecast Models** | Actual vs forecasts, cumulative error, metrics recomputed for any window, 3-month ARIMA outlook with interval | evaluation set, window calendar, models multiselect, rank-by select |
| **Data & Method** | Coverage KPIs and caveats, market map and register, pipeline summary | states multiselect, colour-by select |

Every page includes *insight* callouts that are recalculated from the current selection, for example which state rose fastest in the chosen range, or which model is best in the chosen window.

## Data sources

| Data | File | Notes |
|---|---|---|
| Petrol prices (monthly OHLC, YoY inflation, trust score), Jan 2007 – Apr 2026 | `data/raw/real-time-energy-prices-for-nigeria.csv` | *Real-Time Energy Prices — Nigeria*, Humanitarian Data Exchange (HDX). Layout follows the World Bank Real-Time Prices methodology (Andrée, 2021). |
| USD/NGN monthly close | `data/raw/USD_NGN Historical Data.csv` | Investing.com export. Dates are **day-first** (`01/04/2026` = April 2026). |
| Brent crude, daily (USD/bbl) | `data/raw/DCOILBRENTEU.csv` | FRED series `DCOILBRENTEU`, averaged to monthly. |

## Notebook sections

| Section | Content | Main outputs |
|---|---|---|
| K0 | Set-up | – |
| K1 | Ingestion and profiling (SQL) | series types, market coverage |
| K2 | Quality audit and cleaning (SQL) | `clean_fuel_prices`, plausibility check |
| K3 | Feature engineering (SQL window functions) | `market_features`, `state_monthly`, `national_monthly`; `k3_index_check.png` |
| K4 | Insight 1: price regimes and market structure | `k4_price_regimes.png`, `k4_period_stats.csv`, `k4_driver_correlations.csv` |
| K5 | Insight 2: price shocks and volatility | `k5_price_shocks.png` |
| K6 | Insight 3: regional disparity | `k6_regional_disparity.png`, `k6_state_premiums.csv` |
| K7 | Insight 4: structural breaks (PELT) | `k7_structural_breaks.png` |
| K8 | Insight 5: forecasting set-up | chronological split, scaling on training data only |
| K9 | Insight 6: LSTM with trust-weighted loss | `k9_loss_curves.png`, `outputs/models/*.pt` |
| K10 | Benchmark evaluation and 3-month forecast | `k10_test_forecasts.png`, `k10_forward_forecast.png`, `k10_*_results.csv`, `k10_predictions.csv`, `k10_forecast_3m.csv` |

## Limitations

* Prices are estimates and appear to understate official post-2023 pump prices.
* Coverage is limited to 14 states, mostly in the north-east.
* The post-subsidy sample is short (35 months), so tests have low power.
* The trust score is almost constant, so trust weighting cannot be evaluated properly on this data.

## References

* Andrée, B. P. J. (2021). *Estimating Food Price Inflation from Partial Surveys.* World Bank Policy Research Working Paper 9886.
* Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3).
* Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2).
* Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
* Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA*, 107(500).
* Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning forecasting methods: Concerns and ways forward. *PLOS ONE*, 13(3).
