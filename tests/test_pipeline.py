import numpy as np
import pandas as pd
import pytest
import torch

from fuel_forecast.config import BRENT_CSV, FX_CSV
from fuel_forecast.evaluation import diebold_mariano, price_metrics, returns_to_prices
from fuel_forecast.exogenous import load_brent, load_usd_ngn
from fuel_forecast.features import add_model_columns, make_samples, split_masks
from fuel_forecast.models import FuelLSTM, TrustWeightedMSELoss


# ── Exogenous data ───────────────────────────────────────────────────────────
def test_usd_ngn_dates_are_day_first():
    fx = load_usd_ngn(FX_CSV)
    # One row per month, every row on the 1st, no gaps.
    assert (fx["month"].dt.day == 1).all()
    assert fx["month"].is_unique
    expected = pd.date_range(fx["month"].min(), fx["month"].max(), freq="MS")
    assert len(fx) == len(expected)
    # "01/04/2026" must be April 2026, not January.
    assert fx["month"].max() == pd.Timestamp("2026-04-01")


def test_brent_has_every_month():
    brent = load_brent(BRENT_CSV)
    expected = pd.date_range(brent["month"].min(), brent["month"].max(), freq="MS")
    assert len(brent) == len(expected)
    assert brent["brent_usd"].between(5, 200).all()


# ── Features ─────────────────────────────────────────────────────────────────
def _toy_national(n=30):
    months = pd.date_range("2020-01-01", periods=n, freq="MS")
    price = 100 * np.exp(np.cumsum(np.full(n, 0.01)))
    return pd.DataFrame({
        "month": months,
        "price_close": price,
        "price_high": price * 1.02,
        "price_low": price * 0.98,
        "cross_state_cv": 0.1,
        "trust_score": 9.8,
        "usd_ngn": np.linspace(300, 400, n),
        "brent_usd": 80.0,
    })


def test_windows_never_include_the_target_month():
    df = add_model_columns(_toy_national())
    df["marker"] = np.arange(len(df), dtype=float)  # row number as a feature
    samples = make_samples(df, ["marker", "ret"], window=6)
    usable = df.dropna(subset=["marker", "ret"]).reset_index(drop=True)
    for window, target_month in zip(samples.windows, samples.target_month):
        target_row = usable.index[usable["month"] == target_month][0]
        assert window[-1, 0] == usable.loc[target_row - 1, "marker"]


def test_returns_round_trip_to_prices():
    df = add_model_columns(_toy_national())
    samples = make_samples(df, ["ret"], window=3)
    rebuilt = returns_to_prices(samples.prev_price, samples.target)
    np.testing.assert_allclose(rebuilt, samples.true_price)


def test_split_is_chronological_and_complete():
    months = pd.date_range("2020-01-01", periods=24, freq="MS").to_numpy()
    masks = split_masks(months, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-07-01"))
    assert (masks["train"].astype(int) + masks["val"] + masks["test"] == 1).all()
    assert months[masks["train"]].max() < months[masks["val"]].min()
    assert months[masks["val"]].max() < months[masks["test"]].min()


# ── Model and loss ───────────────────────────────────────────────────────────
def test_trust_loss_equals_mse_when_trust_is_constant():
    pred, target = torch.randn(10), torch.randn(10)
    trust = torch.full((10,), 9.8)
    expected = torch.mean((pred - target) ** 2)
    assert torch.isclose(TrustWeightedMSELoss()(pred, target, trust), expected)


def test_lstm_output_shape():
    model = FuelLSTM(n_features=7, hidden_size=8)
    assert model(torch.randn(4, 6, 7)).shape == (4,)


# ── Evaluation ───────────────────────────────────────────────────────────────
def test_price_metrics_known_values():
    m = price_metrics(np.array([100.0, 200.0]), np.array([90.0, 220.0]))
    assert m["MAE (₦/L)"] == pytest.approx(15.0)
    assert m["MAPE (%)"] == pytest.approx(10.0)
    assert m["Bias (₦/L)"] == pytest.approx(-5.0)


def test_diebold_mariano_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0, 1, 200), rng.normal(0, 3, 200)
    stat, p = diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01
