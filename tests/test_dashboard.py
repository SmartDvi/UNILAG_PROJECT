"""Smoke tests: every dashboard page callback runs for a range of control settings.

These need the notebook's output tables (outputs/tables); they are skipped if absent.
"""

import json
import sys

import plotly
import pytest

from fuel_forecast.config import TABLE_DIR

pytestmark = pytest.mark.skipif(not (TABLE_DIR / "k10_predictions.csv").exists(),
                                reason="run the notebook first to create outputs/tables")

ALL = ["2007-01-01", "2026-04-30"]
POST = ["2023-06-01", "2026-04-30"]
MODELS = ["Naive (no change)", "ARIMA(1, 0, 0)", "LSTM ensemble (h=64)"]

CASES = {
    "overview": [
        dict(dates=ALL, overlays=["usd_ngn", "brent_ngn"], log_scale=True),
        dict(dates=["2024-09-01", "2025-03-31"], overlays=[], log_scale=False),
        dict(dates=None, overlays=None, log_scale=False),
    ],
    "markets": [
        *[dict(dates=ALL, level="state", states=["Borno", "Lagos"], markets=[], measure=m, show_national=True,
               candle_choice=None) for m in ("price_close", "rebased", "premium_pct", "yoy_pct")],
        dict(dates=POST, level="market", states=["Borno"], markets=["gid_101700000127400000"],
             measure="premium_pct", show_national=False, candle_choice=None),
        dict(dates=POST, level="state", states=[], markets=[], measure="price_close", show_national=True,
             candle_choice=None),
    ],
    "regional": [dict(dates=POST, metric=m, highlight=h)
                 for m in ("premium_pct", "price_close", "change_pct") for h in ([], ["Borno"])],
    "volatility": [
        dict(dates=ALL, scope="National index", threshold=5, window="6"),
        dict(dates=POST, scope="Borno", threshold=2.5, window="24"),
    ],
    "breaks": [dict(series_key=s, model=m, penalty=p, min_size=6, show_events=True)
               for s in ("National index", "Lagos") for m in ("l2", "l1", "rbf") for p in ("1.0", "0.1")],
    "drivers": [
        *[dict(dates=ALL, target="National index", driver=d, lag=lag, periods=["0", "1", "2"], window="24")
          for d in ("usd_ngn", "brent_ngn", "brent_usd") for lag in ("0", "3")],
        dict(dates=POST, target="Kano", driver="usd_ngn", lag="1", periods=["1"], window="12"),
    ],
    "forecast": [dict(split=s, dates=d, models=MODELS, metric="MAPE (%)")
                 for s in ("validation", "test", "both") for d in (None, ["2024-05-01", "2025-04-30"])],
    "data_quality": [dict(states=s, color_by=c)
                     for s in ([], ["Borno", "Lagos"]) for c in ("state", "latest_price", "mean_trust")],
}


@pytest.fixture(scope="module")
def pages():
    import dashboard.app  # noqa: F401  (registers the pages)

    return {name.split(".")[-1]: module for name, module in sys.modules.items() if name.startswith("pages.")}


def test_all_pages_registered(pages):
    assert set(CASES) <= set(pages)


@pytest.mark.parametrize("page,kwargs", [(p, c) for p, cases in CASES.items() for c in cases])
def test_callbacks_run_and_serialise(pages, page, kwargs):
    output = pages[page].update(**kwargs)
    json.dumps(output, cls=plotly.utils.PlotlyJSONEncoder)


def test_secondary_callbacks(pages):
    pages["markets"].candles("Borno", ALL, "state", ["Borno"], [])
    groups, value, disabled = pages["markets"].market_options(["Borno", "Lagos"], "market", [])
    assert value and not disabled and {g["group"] for g in groups} == {"Borno", "Lagos"}
    pages["forecast"].outlook("test")
