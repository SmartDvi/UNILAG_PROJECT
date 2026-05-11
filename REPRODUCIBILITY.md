# 🔄 Reproducibility Guide

## Quick Start (30 Minutes)

### Prerequisites
```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
# OR
pip install -e .
```

### Run the Full Pipeline
```bash
# Option 1: Via Jupyter (interactive, shows plots)
jupyter notebook nigeria_fuel_price.ipynb

# Option 2: Via CLI (non-interactive, generates all artifacts)
python main.py --run-all

# Option 3: Via nbconvert (fastest, no browser)
jupyter nbconvert --to notebook --execute --inplace nigeria_fuel_price.ipynb
```

### Expected Output
After execution, you should see:

**Console Output:**
```
✅ Imports complete. DuckDB: 0.10.X | PyTorch: 2.X.X | Ruptures: 1.1.9

✅ Table created and saved to nigeria_fuel.duckdb
✅ clean_fuel_prices created — 17,168 rows
✅ Feature table: 17,168 rows × 36 columns

🚀 Starting LSTM Training
...training progress table...
⏹  Early stopping at epoch 45

✅ Training complete.

📊 TEST SET EVALUATION METRICS
   Mean Absolute Error (MAE):              ₦XX.XX/L
   Root Mean Squared Error (RMSE):         ₦XX.XX/L
   Mean Absolute % Error (MAPE):           X.XX%
   
🔮 3-MONTH FORWARD FORECAST
   Jun 2026 | ₦XXX.XX | 95% CI [XX, XX] | +X.X% | STATUS

🔬 REPRODUCIBILITY & ARTIFACT VALIDATION
✅ All artifacts present and ready for dissertation submission!
```

**Generated Files:**
```
nigeria_fuel.duckdb                  # Persistent DuckDB database
best_nigeria_lstm.pt                 # Best model checkpoint
nigeria_fuel_engineered_features.csv # Feature table (for analysis)
nigeria_fuel_3m_forecast.csv         # 3-month forecast (for stakeholders)

k4_duopoly_analysis.png              # Market structure shift
k5_msme_volatility.png               # Volatility squeeze analysis
k6_regional_disparity.png            # State-level price gaps
k7_structural_break.png              # PELT breakpoint detection
k9_trust_weighted_loss.png           # Loss weighting visualization
k10_loss_curve.png                   # Training convergence
k10_evaluation.png                   # Actual vs predicted plots
```

---

## Detailed Execution Steps

### Step 1: Environment Setup (K0)
- Installs all packages via pip
- Imports libraries
- Sets SEED=42 for reproducibility
- Configures matplotlib/seaborn styles

### Step 2: Data Ingestion (K1–K2)
- Loads CSV into DuckDB persistent database
- Validates schema (55 columns → 8 usable)
- Performs null audit (100% OHLC coverage)
- Creates `clean_fuel_prices` table

### Step 3: Feature Engineering (K3)
- Applies SQL window functions for lagged features
- Computes OHLC derivatives (spread, momentum)
- Calculates rolling statistics (3m, 12m)
- Creates cyclical time encoding (sin/cos for months)
- Generates `engineered_features` table (36 columns)

### Step 4–7: Exploratory Analysis (K4–K7)
- **K4:** Duopoly market structure analysis
- **K5:** MSME volatility squeeze analysis
- **K6:** Regional price disparity mapping
- **K7:** PELT structural break detection

### Step 8–9: Model Building (K8–K9)
- Loads exogenous data (USD/NGN FX, Brent Crude)
- Creates sliding-window dataset
- Initializes 2-layer LSTM
- Defines trust-weighted loss function

### Step 10: Training & Evaluation (K10)
- **Trains LSTM with detailed epoch-by-epoch logging** (new!)
- Visualises loss convergence
- Generates predictions on test set
- Computes MAPE, MAE, RMSE, R²
- Creates actual vs predicted plots
- Generates 3-month forecast with confidence intervals
- **Validates all artifacts** (new!)
- Prints model card summary

---

## Troubleshooting

### Error: "DuckDB database locked"
```
Solution: Delete old .duckdb files and restart
rm nigeria_fuel.duckdb*
jupyter restart kernel
```

### Error: "CUDA out of memory"
```
Solution: CPU is fine for this model size (~200K params)
DEVICE will automatically fall back to CPU
```

### Error: "FileNotFoundError: best_nigeria_lstm.pt"
```
Solution: Training must complete before evaluation runs
Ensure training cell completes without errors
```



### Slow training on CPU?
```
Expected: ~2–3 minutes total (60 epochs)
Reason: 60 epochs × 16 batch size × ~10 batches
Improvement: Install GPU (CUDA for NVIDIA; ROCm for AMD)
```

---

## Data Integrity Checks

Run this Python snippet to verify data integrity:

```python
import duckdb
import pandas as pd

con = duckdb.connect("nigeria_fuel.duckdb")

# Check table exists and has rows
count = con.execute("SELECT COUNT(*) FROM engineered_features").fetchone()[0]
print(f"✅ Engineered features: {count:,} rows")

# Check no NaNs in target variable
nans = con.execute("""
    SELECT COUNT(*) FROM engineered_features 
    WHERE price_close IS NULL
""").fetchone()[0]
print(f"✅ NaNs in target: {nans}")

# Check model file
import os
if os.path.exists("best_nigeria_lstm.pt"):
    size_mb = os.path.getsize("best_nigeria_lstm.pt") / (1024**2)
    print(f"✅ Model checkpoint: {size_mb:.1f} MB")

# Check forecast
forecast_df = pd.read_csv("nigeria_fuel_3m_forecast.csv")
print(f"✅ Forecast rows: {len(forecast_df)}")
print(forecast_df)
```

---

## Performance Expectations

### Training
- **Time:** 2–3 minutes on CPU (M1 Mac: ~45 sec)
- **Memory:** ~1–2 GB RAM
- **Convergence:** Early stop at ~45 epochs (PATIENCE=10)

### Inference
- **Test Set:** 42 samples → predictions in <1 sec
- **3-Month Forecast:** Generated in <100ms

### Model Quality
- **MAPE:** Target <5% (production-grade)
- **R²:** Target >0.80 (explains 80%+ variance)
- **RMSE:** ±₦XX/L (depends on regime volatility)

---

## Reproducibility Notes

### Random Seeding
```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```
All stochastic operations use this seed. Results are **deterministic** across runs.

### Data Leakage Prevention
✅ Scalers fit on train set only, applied to test  
✅ Early stopping monitors validation loss only  
✅ Test dates strictly after train dates (temporal split)  
✅ No information flows from test to train  

### Exogenous Data
- USD/NGN: Real FRED/CBN historical data (daily → monthly aggregation)
- Brent Crude: Real FRED historical data (daily → monthly aggregation)

### Computational Reproducibility
- PyTorch: Deterministic algorithms enabled where possible
- NumPy: Fixed random seed (SEED=42)
- Pandas: No randomisation (deterministic operations)

---

## Citation

If you use this pipeline in your work:

```bibtex
@software{nigeria_fuel_2026,
  author = {Your Name},
  title = {Nigeria Retail Fuel Price Intelligence System},
  year = {2026},
  url = {https://github.com/your-username/uni-lag-project},
  note = {End-to-end ML pipeline for fuel price forecasting}
}
```

---

## Support

For issues, check:
1. **requirements.txt** installed correctly
2. **SEED=42** set before training
3. **Database file** not corrupted (try deleting & regenerating)
4. **GPU/CUDA** not required (CPU works fine)

**Last tested:** May 11, 2026  
**Python version:** 3.10–3.12  
**OS:** Linux, macOS, Windows (via WSL2)
