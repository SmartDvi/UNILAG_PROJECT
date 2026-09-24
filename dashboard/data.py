"""Load the tables produced by the notebook and provide shared calculations.

The dashboard never recomputes the pipeline: it reads the CSVs in
``outputs/tables``. Run the notebook (``python main.py``) first.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuel_forecast import config as cfg  # noqa: E402

REQUIRED_TABLES = [
    "national_monthly.csv",
    "market_features.csv",
    "k10_predictions.csv",
    "k10_forecast_3m.csv",
]

missing = [name for name in REQUIRED_TABLES if not (cfg.TABLE_DIR / name).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing {missing} in {cfg.TABLE_DIR}. Run the notebook first: python main.py"
    )

PERIOD_SHORT = cfg.PERIOD_SHORT
SPIKE_THRESHOLD = cfg.SPIKE_THRESHOLD_PCT
PERIOD_COLORS = {0: cfg.PALETTE["gray"], 1: cfg.PALETTE["amber"], 2: cfg.PALETTE["coral"]}


def _read(name: str, date_col: str = "month") -> pd.DataFrame:
    df = pd.read_csv(cfg.TABLE_DIR / name)
    df[date_col] = pd.to_datetime(df[date_col])
    return df


# ── Core tables ─────────────────────────────────────────────────────────────
NATIONAL = _read("national_monthly.csv")
NATIONAL["brent_ngn"] = NATIONAL["brent_usd"] * NATIONAL["usd_ngn"]
NATIONAL["yoy_pct"] = NATIONAL["price_close"].pct_change(12) * 100

MARKETS = _read("market_features.csv")

STATES = (
    MARKETS.groupby(["state", "month", "period"], as_index=False)
    .agg(price_open=("price_open", "mean"), price_high=("price_high", "mean"),
         price_low=("price_low", "mean"), price_close=("price_close", "mean"),
         trust_score=("trust_score", "mean"), markets=("geo_id", "nunique"))
    .sort_values(["state", "month"])
)
STATES["mom_pct"] = STATES.groupby("state")["price_close"].pct_change() * 100
STATES["yoy_pct"] = STATES.groupby("state")["price_close"].pct_change(12) * 100
STATES = STATES.merge(NATIONAL[["month", "price_close"]].rename(columns={"price_close": "national"}), on="month")
STATES["premium_pct"] = 100 * (STATES["price_close"] / STATES["national"] - 1)
STATES = STATES.sort_values(["state", "month"]).reset_index(drop=True)

PREDICTIONS = _read("k10_predictions.csv")
MODEL_NAMES = [c for c in PREDICTIONS.columns if c not in ("month", "split", "actual", "previous")]

FORECAST = _read("k10_forecast_3m.csv")

MARKET_INFO = (
    MARKETS.groupby(["geo_id", "market", "lga", "state"], as_index=False)
    .agg(lat=("lat", "first"), lon=("lon", "first"), months=("month", "size"),
         first_month=("month", "min"), last_month=("month", "max"),
         mean_trust=("trust_score", "mean"), latest_price=("price_close", "last"))
)

STATE_LIST = sorted(STATES["state"].unique())
MIN_MONTH, MAX_MONTH = NATIONAL["month"].min(), NATIONAL["month"].max()


# ── Helpers shared by pages ──────────────────────────────────────────────────
def parse_range(value) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Turn a DatePickerInput range value into a pair of month-start timestamps."""
    if not value or len(value) != 2 or not all(value):
        return MIN_MONTH, MAX_MONTH
    start, end = (pd.Timestamp(v).to_period("M").to_timestamp() for v in value)
    return max(start, MIN_MONTH), min(end, MAX_MONTH)


def between(df: pd.DataFrame, start, end, col: str = "month") -> pd.DataFrame:
    return df[(df[col] >= start) & (df[col] <= end)]


def period_spans(start=None, end=None) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(first, last, period) spans of the policy periods, clipped to a range."""
    start = start or MIN_MONTH
    end = end or MAX_MONTH
    spans = []
    for period, group in NATIONAL.groupby("period"):
        first, last = max(group["month"].min(), start), min(group["month"].max(), end)
        if first <= last:
            spans.append((first, last + pd.offsets.MonthEnd(0), period))
    return spans


def pct_change(first: float, last: float) -> float:
    return 100 * (last / first - 1)


def shock_streak(flags: pd.Series) -> int:
    """Longest run of consecutive True values."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def series_for(key: str) -> pd.Series:
    """Monthly closing price of the national index or of one state."""
    if key == "National index":
        return NATIONAL.set_index("month")["price_close"]
    return STATES[STATES["state"] == key].set_index("month")["price_close"]


@lru_cache(maxsize=256)
def pelt_breaks(series_key: str, model: str, penalty: float, min_size: int = 6) -> tuple:
    """Change-points (first month of each new regime) for the national index or a state."""
    import ruptures as rpt

    s = series_for(series_key)
    signal = np.log(s.to_numpy()).reshape(-1, 1)
    idx = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal).predict(pen=penalty)[:-1]
    return tuple(s.index[i] for i in idx)


def forecast_metrics(df: pd.DataFrame, model: str) -> dict:
    err = df["actual"] - df[model]
    true_dir = np.sign(df["actual"] - df["previous"])
    pred_dir = np.sign(df[model] - df["previous"])
    moves = pred_dir != 0
    return {
        "Model": model,
        "MAE (₦/L)": np.abs(err).mean(),
        "RMSE (₦/L)": np.sqrt((err**2).mean()),
        "MAPE (%)": (np.abs(err) / df["actual"]).mean() * 100,
        "Bias (₦/L)": err.mean(),
        "Direction correct (%)": (true_dir[moves] == pred_dir[moves]).mean() * 100 if moves.any() else np.nan,
        "Months": len(df),
    }
