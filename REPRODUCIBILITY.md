# Reproducibility Guide

## Requirements

* Python 3.10 or newer (developed on 3.12)
* About 1 GB of free disk space for PyTorch; no GPU needed

## Set-up

```bash
cd UNI_LAG_PROJECT
uv venv && source .venv/bin/activate        # or: python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # or: pip install -r requirements.txt
```

All input data is already in `data/raw/`. Nothing needs downloading and there are no absolute paths.

## Run

```bash
python main.py --test     # unit tests, then executes the notebook in place (~1 minute on CPU)
```

Equivalent manual steps:

```bash
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace nigeria_fuel_price.ipynb
```

For interactive work, start `jupyter lab` **from the project root** (the notebook imports `src/` relative to the working directory) and choose *Restart Kernel and Run All Cells*.

## Dashboard

```bash
cd dashboard
python app.py --dashboard      # http://127.0.0.1:8050
```

The dashboard needs the CSVs in `outputs/tables/`, so run the notebook first. It stops with a clear message if they are missing. Map tiles load from the internet (CARTO); everything else works offline.

## What gets generated

| Location | Contents |
|---|---|
| `data/processed/nigeria_fuel.duckdb` | Database with `raw_fuel_prices`, `clean_fuel_prices`, `market_features`, `state_monthly`, `national_monthly`, `usd_ngn_monthly`, `brent_monthly` |
| `outputs/figures/` | `k3_index_check`, `k4_price_regimes`, `k5_price_shocks`, `k6_regional_disparity`, `k7_structural_breaks`, `k9_loss_curves`, `k10_test_forecasts`, `k10_forward_forecast` (PNG) |
| `outputs/tables/` | Period statistics, driver correlations, state premiums, validation and test results, 3-month forecast, national and market feature tables (CSV) |
| `outputs/models/` | `lstm_h64_seed{0..4}.pt`: LSTM state dicts |

## Expected key numbers

A correct run reproduces these exactly:

| Check | Value |
|---|---|
| Clean table | 15,544 rows, 67 markets, 14 states |
| PELT breaks (penalty 1.0) | Mar 2016, Jan 2022, Dec 2023 |
| Sample split | train 176 / validation 24 / test 24 |
| ARIMA order | (1, 0, 0) |
| Selected LSTM hidden size | 64 |
| Test MAPE: naive / ARIMA / LSTM | 2.269% / 1.939% / 2.744% |
| May 2026 ARIMA forecast | ₦705.55 |

Determinism comes from fixed seeds (`SEED = 42`), `torch.use_deterministic_algorithms(True)` and a seeded mini-batch shuffle. Different PyTorch or BLAS versions, or a GPU, may change the LSTM numbers in the last decimals. The statistical benchmarks and all SQL results do not depend on these settings.

## Data provenance

| File | SHA-256 |
|---|---|
| `real-time-energy-prices-for-nigeria.csv` | `4e4a4e3ab97f62eb27d3b6521fc631d9dd50ea8364df38528fd80790bc0dc7c5` |
| `USD_NGN Historical Data.csv` | `ec409ed23588714e2dffcb98cc413362b33abd4e6d41733fbf69d5e9c928c8c5` |
| `DCOILBRENTEU.csv` | `2bf63bb2e0b32e886d886c9b81439fc2c013aa79877b21bdb9135d6efbbd5397` |

Check with `sha256sum data/raw/*`. If you download newer versions of these files, the results will change. Re-run the notebook and update the write-up.

## Tests

`tests/test_dashboard.py` calls every dashboard callback with a range of control settings and checks that the output serialises; it is skipped if the notebook outputs are missing.

`tests/test_pipeline.py` covers the parts where silent errors are most likely:

* USD/NGN dates parse day-first, giving one row per month with no gaps
* Brent monthly averages cover every month
* sliding windows never include the target month (no look-ahead)
* returns convert back to the exact prices
* the train, validation and test splits are chronological and do not overlap
* the trust-weighted loss equals MSE when trust is constant
* metric values and the Diebold–Mariano test behave as expected
