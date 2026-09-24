# Model Card: Nigeria Petrol Price Index Forecasters

## Overview

| | |
|---|---|
| **Task** | One-month-ahead forecast of a state-balanced national petrol price index for Nigeria |
| **Target** | Next month's log return of the index; forecasts are converted back to ₦/L |
| **Models** | LSTM ensemble (5 seeds) and four benchmarks: naive, drift, ARIMA(1,0,0), ridge regression |
| **Recommended model** | **ARIMA(1,0,0) on returns.** It is the most accurate on the test period, the only model significantly better than naive, and it provides prediction intervals |
| **Code** | `src/fuel_forecast/`, notebook sections K8–K10 |

## Intended use

* Short-term (1–3 month) planning signals on the direction and rough size of petrol price changes in the covered markets, for example MSME cost planning.
* Research and teaching on forecasting short, regime-shifting economic series.

**Not intended for:** quoting official pump prices, pricing contracts, or making claims about states and markets outside the 14 covered states.

## Data

* **Prices:** *Real-Time Energy Prices — Nigeria* (HDX), Jan 2007 – Apr 2026. The file has 67 individual markets in 14 states, 79% of them in Borno, Yobe and Adamawa. The 7 aggregate series in the file are excluded. The index is the mean of the 14 state means.
* **Exogenous:** USD/NGN monthly close (Investing.com) and Brent crude (FRED, monthly mean), both used as monthly log returns.
* **Nature of the prices:** model-based estimates (World Bank Real-Time Prices layout). After June 2023 they are well below reported official pump prices.

## Inputs

A 6-month window of 7 features, all known by the end of the forecast origin month: index return, USD/NGN return, return of Brent priced in naira, intra-month range ÷ close, cross-state coefficient of variation, and sine/cosine of the calendar month.

## Training and selection

| Split (by target month) | Months | Use |
|---|---|---|
| Train | Sep 2007 – Apr 2022 (176) | fitting; scalers fitted here only |
| Validation | May 2022 – Apr 2024 (24) | LSTM early stopping and hidden size (ARIMA orders are chosen by AIC on train, not here) |
| Test | May 2024 – Apr 2026 (24) | final evaluation, used once |

* **LSTM:** 1 layer, hidden size 64 (chosen from {16, 32, 64} on validation MAPE), dropout 0.1, linear head. AdamW (lr 1e-3, weight decay 1e-3), batch size 32, up to 300 epochs, early stopping with patience 30. Trained with 5 seeds, predictions averaged.
* **Loss:** trust-weighted MSE. The trust score varies only from 9.7 to 10, so the weights differ by under 1%. The ablation with plain MSE gives the same validation MAPE (4.116% vs 4.117%).
* **ARIMA:** ARMA(p, q) on returns with p ≤ 3 and q ≤ 2, chosen by AIC on training data (result: (1, 0, 0)). Parameters are fixed when forecasting the validation and test periods.
* **Ridge:** L2-penalised regression on the flattened 6 × 7 window, with the penalty strength chosen by cross-validation on training data.

## Performance (one month ahead, ₦/L)

| Model | Val MAPE | Test MAPE | Test MAE | Test RMSE | Test direction correct | DM vs naive (test) |
|---|---|---|---|---|---|---|
| Naive | 5.05% | 2.27% | 12.64 | 17.77 | – | – |
| Drift | 4.79% | 2.54% | 13.45 | 16.71 | 87.5% | p = 0.71 |
| **ARIMA(1,0,0)** | 4.22% | **1.94%** | **10.79** | **15.89** | 79.2% | **p = 0.016 (better)** |
| Ridge | 4.27% | 2.09% | 11.55 | 16.37 | 75.0% | p = 0.22 |
| LSTM ensemble | 4.12% | 2.74% | 15.61 | 22.46 | 41.7% | p = 0.054 (worse) |

The LSTM was the best model on validation but did not generalise: on the test period it is less accurate than the naive forecast. All models under-forecast on average (positive bias) because the index rose steadily over the test period.

## Forward forecast (from Apr 2026, ₦696.79/L)

| Month | ARIMA | 95% interval |
|---|---|---|
| May 2026 | ₦705.55 | ₦666.84 – ₦746.51 |
| Jun 2026 | ₦713.07 | ₦647.23 – ₦785.61 |
| Jul 2026 | ₦720.12 | ₦631.96 – ₦820.59 |

## Limitations and risks

* **Estimated prices:** errors in the source model carry through to these forecasts. Validate against NBS *PMS Price Watch* before relying on levels.
* **Short regime:** only 35 post-subsidy months are available, so accuracy estimates rest on 24 test points.
* **Structural breaks:** sudden policy or supply shocks (such as those in October 2025 and March 2026) are not forecastable from past prices, and all models miss them.
* **Recursive multi-step forecasts:** the LSTM path assumes the exchange rate and Brent stay flat.
* **Coverage bias** towards the north-east.

## Reproducibility

Seeds are fixed (`config.SEED = 42`, seeds 42–46 for the ensemble) and PyTorch deterministic algorithms are enabled. Re-running the notebook reproduces every number above. See `REPRODUCIBILITY.md`.
