"""Benchmark forecasters that any machine-learning model has to beat.

All functions return predicted next-month log returns, aligned with the
samples produced by ``features.make_samples``.
"""

import warnings

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


def naive(n: int) -> np.ndarray:
    """Random walk: next month's price equals this month's (zero return)."""
    return np.zeros(n)


def drift(windows: np.ndarray, return_col: int = 0) -> np.ndarray:
    """Random walk with drift: next return = mean return over the input window."""
    return windows[:, :, return_col].mean(axis=1)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray):
    """Ridge regression on the flattened feature window, alpha chosen by CV."""
    scaler = StandardScaler().fit(x_train.reshape(len(x_train), -1))
    model = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(
        scaler.transform(x_train.reshape(len(x_train), -1)), y_train
    )

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(x.reshape(len(x), -1)))

    return predict, model.alpha_


def select_arima(returns_train: np.ndarray, max_p: int = 3, max_q: int = 2):
    """Pick ARMA(p, q) orders for the return series by AIC on training data only."""
    best_aic, best_order, best_fit = np.inf, None, None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = ARIMA(returns_train, order=(p, 0, q)).fit()
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, (p, 0, q), fit
    return best_fit, best_order


def arima_one_step(fit, returns_full: np.ndarray) -> np.ndarray:
    """One-step-ahead predictions over the full return series.

    Parameters stay fixed at the training estimates; the filter only uses
    returns up to month t to predict month t+1. Element i is the forecast for
    ``returns_full[i]``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(fit.apply(returns_full).predict())
