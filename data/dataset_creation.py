import ssl
import os
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from fredapi import Fred
from dotenv import load_dotenv

# Config
start = "2008-01-02"
end = "2024-12-31"

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file")

fred = Fred(api_key=FRED_API_KEY)
ssl._create_default_https_context = ssl._create_unverified_context

# Download S&P 500 and VIX
sp = yf.download("^GSPC", start=start, end=end)[['Open', 'High', 'Low', 'Close', 'Volume']]
vix = yf.download("^VIX", start=start, end=end)[['Close']].rename(columns={'Close': 'VIX'})

# Technical indicators
close = sp['Close'].squeeze()
volume = sp['Volume'].squeeze()

sp['RSI'] = RSIIndicator(close).rsi()
sp['MACD_diff'] = MACD(close).macd_diff()

bb = BollingerBands(close)
sp['BB_high'] = bb.bollinger_hband()
sp['BB_low'] = bb.bollinger_lband()

sp['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

# Merge VIX
df = sp.merge(vix, left_index=True, right_index=True, how='left')

# Add macro data from FRED
macro_series = {
    'CPI': 'CPIAUCSL',
    'Unemployment': 'UNRATE',
    'FedFunds': 'FEDFUNDS',
}

for col, series_id in macro_series.items():
    try:
        df[col] = fred.get_series(series_id, start, end)
    except Exception as e:
        print(f"Could not fetch {col}: {e}")
        df[col] = np.nan

df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['target'] = df['log_return'].shift(-1)

# Clean data
df = df.ffill().dropna()

print(f"Final dataset shape: {df.shape}")
df.to_csv("data.csv")
print("Saved to data.csv")