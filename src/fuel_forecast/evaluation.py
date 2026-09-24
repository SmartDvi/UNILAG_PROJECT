"""Forecast accuracy metrics and the Diebold–Mariano test."""

import numpy as np
import pandas as pd
from scipy import stats


def returns_to_prices(prev_price: np.ndarray, predicted_return: np.ndarray) -> np.ndarray:
    """Convert predicted log returns back to price levels."""
    return prev_price * np.exp(predicted_return)


def price_metrics(true_price: np.ndarray, pred_price: np.ndarray) -> dict[str, float]:
    """MAE and RMSE in ₦/L, MAPE in %, and bias (mean of actual − predicted)."""
    error = true_price - pred_price
    return {
        "MAE (₦/L)": float(np.mean(np.abs(error))),
        "RMSE (₦/L)": float(np.sqrt(np.mean(error**2))),
        "MAPE (%)": float(np.mean(np.abs(error) / true_price) * 100),
        "Bias (₦/L)": float(np.mean(error)),
    }


def directional_accuracy(true_return: np.ndarray, pred_return: np.ndarray) -> float:
    """Share of months where the predicted direction of change is correct (%).

    Undefined (NaN) for a model that never predicts a change, such as the naive forecast.
    """
    if np.all(pred_return == 0):
        return float("nan")
    return float(np.mean(np.sign(true_return) == np.sign(pred_return)) * 100)


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """Diebold–Mariano test of equal accuracy under squared-error loss.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction and a
    Student-t reference distribution. A negative statistic means model A has
    the lower loss. Returns ``(statistic, two-sided p-value)``.
    """
    d = np.asarray(errors_a) ** 2 - np.asarray(errors_b) ** 2
    n = len(d)
    d_bar = d.mean()

    # Long-run variance with autocovariances up to lag horizon-1.
    gamma = [np.mean((d[k:] - d_bar) * (d[: n - k] - d_bar)) for k in range(horizon)]
    long_run_var = gamma[0] + 2 * sum(gamma[1:])

    dm = d_bar / np.sqrt(long_run_var / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def comparison_table(
    true_price: np.ndarray,
    prev_price: np.ndarray,
    true_return: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str = "Naive (no change)",
) -> pd.DataFrame:
    """One row per model: accuracy metrics plus a DM test against ``reference``.

    ``predictions`` maps model name to predicted log returns.
    """
    ref_error = true_price - returns_to_prices(prev_price, predictions[reference])
    rows = []
    for name, pred_return in predictions.items():
        pred_price = returns_to_prices(prev_price, pred_return)
        row = {"Model": name, **price_metrics(true_price, pred_price)}
        row["Direction correct (%)"] = directional_accuracy(true_return, pred_return)
        if name == reference:
            row["DM stat vs naive"], row["DM p-value"] = np.nan, np.nan
        else:
            row["DM stat vs naive"], row["DM p-value"] = diebold_mariano(true_price - pred_price, ref_error)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
