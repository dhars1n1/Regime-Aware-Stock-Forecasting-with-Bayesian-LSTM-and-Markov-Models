# Data Folder Runbook 🚦

## Quick Overview

The data processing pipeline consists of four main scripts that should be run in order:

```
data_creation.py → download + enrich raw market + macro → data.csv
eda.py → clean & explore → data_cleaned.csv
sentiment.py (optional) → (hybrid offline or Twitter + FinBERT) → data_with_sentiment.csv
feature_regime_preprocessing.py → scale, build lags, run multivariate MSM → data_with_regimes.csv
```

## Files & What They Do

### `data_creation.py`
- **Purpose**: Download S&P 500 (^GSPC) & ^VIX with yfinance, compute technical indicators (RSI, MACD, BBands, OBV), pull FRED macro series, add simple flags (is_crisis, earnings_season), forward-fill and save
- **Inputs**: None (reads environment variable for FRED API key if using FRED)
- **Outputs**: `data.csv` (raw enriched CSV)
- **Run if**: You need to regenerate market + macro features from the web

### `eda.py`
- **Purpose**: Fix CSV header (handles two header rows), parse Date, compute log returns, run stationarity test (ADF), basic plots, and save cleaned file
- **Inputs**: `data.csv`
- **Outputs**: `data_cleaned.csv`
- **Run if**: You want a cleaned, indexed dataset ready for modeling

### `sentiment.py`
- **Purpose**: Merge historical sentiment data from the **American Association of Individual Investors** Excel file into your main dataset.
- **Logic**:
  1. Reads `sentiment.xls` (skips the first 4 non-data rows).
  2. Cleans `"xx.x%"` strings into floats for **Bullish**, **Neutral**, **Bearish**.
  3. Drops rows with no sentiment data.
  4. Creates `sentiment_score` = Bullish − Bearish.
  5. Merges sentiment into `data_cleaned.csv` using nearest past date (`merge_asof`), forward-filling values.
- **Inputs**:
  - `data_cleaned.csv` (from `eda.py`)
  - `sentiment.xls` (historical AAII sentiment Excel file)
- **Outputs**: `data_with_sentiment.csv` — main dataset with added `Bullish`, `Neutral`, `Bearish`, and `sentiment_score`.
- **Run if**: You have the AAII sentiment Excel and want it included as a predictive feature.
- **Python version**: Works fine on Python 3.13
- **Run**:
  ```bash
  python data/sentiment.py
  ```

### `msm/msm.py`
- **Purpose**: Contains the from-scratch multivariate MSM implementation (Baum–Welch EM, forward/backward, Viterbi). Exposes function `baum_welch_multivariate(obs, K=3, ...)`
- **Inputs**: Numeric observation array (T × D) — the preprocessing script passes [log_return, VIX] by default
- **Outputs**: Dict containing pi, A, means, covs, gamma, viterbi

### `feature_regime_preprocessing.py`
- **Purpose**: Orchestrates final preprocessing:
  - Loads `data_with_sentiment.csv`
  - Builds log_return, lag features, normalizes features
  - Prepares multivariate observation array (default: log_return + VIX)
  - Calls `msm.baum_welch_multivariate(...)`
  - Attaches regime_k_prob, regime_viterbi, regime_label
  - Saves `data_with_regimes.csv`
- **Inputs**: `data_with_sentiment.csv` (and `msm/msm.py` in parent folder — imports via sys.path adjustment)
- **Outputs**: `data_with_regimes.csv` ready for modeling

## Exact Run Order (Recommended)

1. **(Optional)** Create virtual environment (see next section)

2. **Step 1**: Generate market and macro data
   ```bash
   python data/data_creation.py
   ```
   Produces `data/data.csv`

3. **Step 2**: Clean and explore data
   ```bash
   python data/eda.py
   ```
   Produces `data/data_cleaned.csv`

4. **Step 3** Add sentiment analysis
   ```bash
   python data/sentiment.py
   ```
   Produces `data/data_with_sentiment.csv` (or overwrites/merges into cleaned file)

5. **Step 4**: Feature engineering and regime detection
   ```bash
   python data/feature_regime_preprocessing.py
   ```
   Requires `msm/msm.py` accessible from project root (see import help)
   Produces `data/data_with_regimes.csv`

> **Note**: You can run step 3 (sentiment) any time before step 4. If you skip sentiment, step 4 still works — sentiment will be a placeholder.

## Virtual Environments & Python Versions (Recommended)

Create and activate a venv in the project root:

```bash
# macOS (example)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
```
## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

**Mac M4 / PyTorch note**: Install PyTorch following the official instructions at https://pytorch.org — select macOS/arm64 (MPS) if you want hardware acceleration. If unsure, `pip install torch` will get a compatible wheel for many setups; follow PyTorch site if issues occur.

## Environment Variables / Config Files

Create `.env` at project root if needed:

```bash
FRED_API_KEY=your_fred_api_key_here
```

- `data_creation.py` needs `FRED_API_KEY` to successfully fetch CPI / UNRATE / FEDFUNDS. If not present, that code sets NaNs (or you can edit to skip FRED)
- `sentiment.py` (online NewsAPI mode) needs `NEWSAPI_KEY`. Offline hybrid uses `news_archive.csv` so you don't need the NewsAPI key

## Example Run Commands

### From Project Root

```bash
# activate venv
source .venv/bin/activate

# 1. create market + macros
python data/data_creation.py

# 2. EDA & clean
python data/eda.py

# 3. sentiment addition
python data/sentiment.py

# 4. feature engineering + regimes
python data/feature_regime_preprocessing.py
```

### From Data Directory

```bash
cd data
python data_creation.py
python eda.py
python sentiment.py
python feature_regime_preprocessing.py
```

## Import / Path Note (MSM Directory)

If `msm/msm.py` is in a sibling folder to `data/`, `feature_regime_preprocessing.py` must add the project root to sys.path before importing:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from msm.msm import baum_welch_multivariate
```

## Expected Outputs (Summary)

- `data/data.csv` — enriched raw CSV (from `data_creation.py`)
- `data/data_cleaned.csv` — cleaned, indexed CSV with returns (from `eda.py`)
- `data/data_with_sentiment.csv` — cleaned + sentiment merged (if sentiment run)
- `data/data_with_regimes.csv` — final CSV with scaled features, lag features, regime_k_prob, regime_viterbi, regime_label (from `feature_regime_preprocessing.py`)

## Runtime Expectations / Tips

- **`data_creation.py`**: Minutes (downloads market+macro data)
- **`eda.py`**: Seconds to minutes
- **`sentiment.py`**: Slow for long ranges (FinBERT inference on CPU): batches & yearly chunks recommended. Use `max_tweets_per_day=50` for speed
- **`feature_regime_preprocessing.py`**: EM on ~9k rows with multivariate MSM usually tens of seconds to a few minutes, depending on max_iter and dimensionality; add logging for iterations to monitor convergence

## Troubleshooting Checklist

- **If import snscrape fails**: Use Python 3.11 venv or call snscrape CLI with subprocess
- **If HuggingFace model downloads hang**: Check internet or allow the large pytorch_model.bin download (approx 400+ MB). Cache is in `~/.cache/huggingface/transformers`
- **If msm import fails**: Ensure sys.path is set in `feature_regime_preprocessing.py`, or run the script from project root
- **If the MSM EM doesn't converge**: Try different initialization (KMeans) or increase max_iter