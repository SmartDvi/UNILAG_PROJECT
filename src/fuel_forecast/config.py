"""Project-wide paths, dates and constants.

Everything that a reader might want to change (file locations, split dates,
random seed) lives here so the notebook itself contains no magic values.
"""

from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

FUEL_CSV = RAW_DIR / "real-time-energy-prices-for-nigeria.csv"
FX_CSV = RAW_DIR / "USD_NGN Historical Data.csv"
BRENT_CSV = RAW_DIR / "DCOILBRENTEU.csv"
DUCKDB_PATH = PROCESSED_DIR / "nigeria_fuel.duckdb"

# ── Series in the raw file that are aggregates, not individual markets ──────
# Including them in cross-market averages would double-count markets.
AGGREGATE_ADM1 = ("Market Average", "National Average", "Geopolitical Zone")

# ── Policy / market-structure periods (first month of each period) ──────────
# Subsidy removal was announced on 29 May 2023; June 2023 is the first full
# month of deregulated pricing. The Dangote refinery began supplying petrol to
# the domestic market in September 2024.
SUBSIDY_REMOVAL = pd.Timestamp("2023-06-01")
DOMESTIC_REFINING = pd.Timestamp("2024-09-01")

PERIOD_LABELS = {
    0: "Regulated (Jan 2007 – May 2023)",
    1: "Post-subsidy, import-dependent (Jun 2023 – Aug 2024)",
    2: "Post-subsidy, domestic refining (Sep 2024 – Apr 2026)",
}
PERIOD_SHORT = {0: "Regulated", 1: "Post-subsidy (import)", 2: "Post-subsidy (domestic refining)"}

# ── Forecasting set-up ───────────────────────────────────────────────────────
# Chronological split on the *target* month. Validation is used for early
# stopping and model selection; the test period is touched once, at the end.
VAL_START = pd.Timestamp("2022-05-01")
TEST_START = pd.Timestamp("2024-05-01")

WINDOW = 6            # months of history fed to the sequence models
FORECAST_HORIZON = 3  # months ahead for the forward forecast
SPIKE_THRESHOLD_PCT = 5.0  # month-on-month rise treated as a cost shock

SEED = 42
N_SEEDS = 5           # LSTM ensemble size

PALETTE = {
    "teal": "#1D9E75",
    "blue": "#378ADD",
    "coral": "#D85A30",
    "amber": "#EF9F27",
    "purple": "#7F77DD",
    "gray": "#888780",
}
