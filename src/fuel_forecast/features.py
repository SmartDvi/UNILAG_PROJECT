"""Turn the monthly national series into supervised-learning samples.

The models predict next month's *log return* of the petrol price index,

    r[t+1] = log(P[t+1]) - log(P[t]),

rather than the price level. Prices rose roughly tenfold over 2007–2026, so a
model trained on levels has to extrapolate outside the range it was trained
on; returns are much closer to stationary. A predicted return converts back to
a price with P_hat[t+1] = P[t] * exp(r_hat[t+1]).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "ret",            # log return of the petrol index this month
    "ret_usd_ngn",    # log return of the exchange rate
    "ret_brent_ngn",  # log return of Brent crude priced in naira
    "spread_pct",     # (high - low) / close within the month
    "cross_state_cv", # price dispersion across states
    "month_sin",
    "month_cos",
]


def add_model_columns(national: pd.DataFrame) -> pd.DataFrame:
    """Add returns, cyclical month encoding and the target column.

    ``national`` must have one row per month, sorted, with columns
    ``month, price_close, price_high, price_low, cross_state_cv, usd_ngn,
    brent_usd``.
    """
    df = national.sort_values("month").reset_index(drop=True).copy()
    brent_ngn = df["brent_usd"] * df["usd_ngn"]

    df["ret"] = np.log(df["price_close"]).diff()
    df["ret_usd_ngn"] = np.log(df["usd_ngn"]).diff()
    df["ret_brent_ngn"] = np.log(brent_ngn).diff()
    df["spread_pct"] = (df["price_high"] - df["price_low"]) / df["price_close"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"].dt.month / 12)
    return df


@dataclass
class Samples:
    """Sliding-window samples.

    ``windows[i]`` holds features for months ``end[i]-window+1 .. end[i]``;
    ``target[i]`` is the return in month ``end[i]+1``. No feature is ever
    taken from the target month, so there is no look-ahead.
    """

    windows: np.ndarray       # (n, window, n_features)
    target: np.ndarray        # (n,) next-month log return
    target_month: np.ndarray  # (n,) datetime64 of the month being predicted
    prev_price: np.ndarray    # (n,) price in the last month of the window
    true_price: np.ndarray    # (n,) actual price in the target month
    trust: np.ndarray         # (n,) mean trust score of the target month


def make_samples(df: pd.DataFrame, features: list[str], window: int) -> Samples:
    """Build look-ahead-free sliding windows from a frame made by ``add_model_columns``."""
    usable = df.dropna(subset=features).reset_index(drop=True)
    X = usable[features].to_numpy(dtype=np.float64)
    ends = np.arange(window - 1, len(usable) - 1)
    targets = ends + 1

    return Samples(
        windows=np.stack([X[e - window + 1 : e + 1] for e in ends]),
        target=usable["ret"].to_numpy()[targets],
        target_month=usable["month"].to_numpy()[targets],
        prev_price=usable["price_close"].to_numpy()[ends],
        true_price=usable["price_close"].to_numpy()[targets],
        trust=usable["trust_score"].to_numpy()[targets],
    )


def split_masks(target_month: np.ndarray, val_start, test_start) -> dict[str, np.ndarray]:
    """Boolean masks for a chronological train / validation / test split."""
    months = pd.to_datetime(target_month)
    return {
        "train": months < val_start,
        "val": (months >= val_start) & (months < test_start),
        "test": months >= test_start,
    }
