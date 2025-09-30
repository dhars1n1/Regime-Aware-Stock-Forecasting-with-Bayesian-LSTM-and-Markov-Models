import ssl
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from fredapi import Fred
from dotenv import load_dotenv

# =====================
# CONFIG
# =====================
start = "2008-01-02"
end = "2024-12-31"

# Load environment variables from .env file
load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found. Please set it in your .env file.")

fred = Fred(api_key=FRED_API_KEY)


# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

# =====================
# Flatten columns helper
# =====================
def flatten_df(df):
    """Ensures no column contains (N,1) shaped data."""
    for col in df.columns:
        vals = df[col].values
        if hasattr(vals, 'shape') and len(vals.shape) == 2 and vals.shape[1] == 1:
            df[col] = vals.ravel()
    return df

# =====================
# 1. Download Data
# =====================
sp = yf.download("^GSPC", start=start, end=end)
vix_df = yf.download("^VIX", start=start, end=end)

# Flatten immediately
sp = flatten_df(sp)
vix_df = flatten_df(vix_df)

# Select necessary columns
sp = sp[['Open', 'High', 'Low', 'Close', 'Volume']]
vix_df = vix_df[['Close']].rename(columns={'Close': 'VIX'})

# =====================
# 2. Indicators
# =====================
# Ensure Close and Volume are 1D Series
close = sp['Close'].squeeze()
volume = sp['Volume'].squeeze()

# Technical indicators
sp['RSI'] = RSIIndicator(close).rsi()
sp['MACD_diff'] = MACD(close).macd_diff()

bb = BollingerBands(close)
sp['BB_high'] = bb.bollinger_hband()
sp['BB_low'] = bb.bollinger_lband()

sp['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

# =====================
# 3. Merge VIX
# =====================
df = sp.merge(vix_df, left_index=True, right_index=True, how='left')

# =====================
# 4. Macro data (FRED)
# =====================
fred_series = {
    'CPI': 'CPIAUCSL',
    'Unemployment': 'UNRATE',
    'FedFunds': 'FEDFUNDS',
}

for col, series_id in fred_series.items():
    try:
        s = fred.get_series(series_id, start, end)
        df[col] = s
    except Exception as e:
        print(f"Could not fetch {col}: {e}")
        df[col] = np.nan

# =====================
# 5. Extra Features
# =====================
df['sentiment'] = 0.0
df['is_crisis'] = (df['VIX'] > 40).astype(int)
df['fed_meeting'] = 0
df['earnings_season'] = (
    df.index.month.isin([1, 4, 7, 10]) & (df.index.day <= 20)
).astype(int)

# =====================
# Final Save
# =====================
df = df.ffill().dropna()
print(f"Final dataset shape: {df.shape}")
df.to_csv("data.csv")
print("Saved enriched dataset to data.csv")
