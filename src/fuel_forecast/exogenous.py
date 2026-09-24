"""Loaders for the exogenous drivers: the USD/NGN exchange rate and Brent crude.

Both loaders return one row per calendar month, keyed by the first day of the
month, so they join cleanly onto the monthly fuel-price series.
"""

from pathlib import Path

import pandas as pd


def load_usd_ngn(path: Path) -> pd.DataFrame:
    """Load the monthly USD/NGN file (Investing.com export).

    The ``Date`` column is written day-first (``01/04/2026`` is 1 April 2026).
    Parsing it month-first silently turns most rows into the wrong month, so
    ``dayfirst=True`` is essential. ``Price`` is the month-end closing rate.

    Returns columns ``month`` and ``usd_ngn``.
    """
    fx = pd.read_csv(path)
    month = pd.to_datetime(fx["Date"], format="%d/%m/%Y")
    rate = fx["Price"].astype(str).str.replace(",", "", regex=False).astype(float)

    out = pd.DataFrame({"month": month.dt.to_period("M").dt.to_timestamp(), "usd_ngn": rate})
    if out["month"].duplicated().any():
        raise ValueError("USD/NGN file contains more than one row per month")
    return out.sort_values("month").reset_index(drop=True)


def load_brent(path: Path) -> pd.DataFrame:
    """Load daily Brent crude (FRED ``DCOILBRENTEU``) and average it by month.

    Daily data has gaps on weekends and holidays, so joining on an exact date
    misses many months. Averaging within each calendar month avoids that.

    Returns columns ``month`` and ``brent_usd``.
    """
    brent = pd.read_csv(path)
    brent["brent_usd"] = pd.to_numeric(brent["DCOILBRENTEU"], errors="coerce")
    brent["month"] = pd.to_datetime(brent["observation_date"]).dt.to_period("M").dt.to_timestamp()
    return (
        brent.dropna(subset=["brent_usd"])
        .groupby("month", as_index=False)["brent_usd"]
        .mean()
    )
