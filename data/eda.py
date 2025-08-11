# =========================
# 📊 EDA for S&P 500 Enriched Dataset
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# -------------------------
# 1. Load & Clean Dataset
# -------------------------

# Read CSV with 2 header rows
df = pd.read_csv("Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data.csv", header=[0,1])

# Drop the first row (tickers) and flatten headers
df.columns = df.columns.get_level_values(0)
df = df.drop(index=0)

# Convert Date column to datetime and set as index
df['Price'] = pd.to_datetime(df['Price'])
df = df.rename(columns={'Price': 'Date'})
df = df.set_index('Date')

# Convert numeric columns to float
df = df.apply(pd.to_numeric, errors='coerce')

print("Shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())

# -------------------------
# 2. Missing Values Check
# -------------------------
missing = df.isna().sum()
print("\nMissing values per column:\n", missing)

# -------------------------
# 3. Basic Statistics
# -------------------------
print("\nSummary statistics:\n", df.describe().T)

# -------------------------
# 4. Plot Key Features
# -------------------------
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.title("S&P 500 Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(df.index, df['VIX'], color='orange', label='VIX')
plt.title("Volatility Index (VIX)")
plt.axhline(40, color='red', linestyle='--', label='Crisis threshold (VIX>40)')
plt.legend()
plt.show()

# -------------------------
# 5. Crisis Periods
# -------------------------
crisis_days = df[df['VIX'] > 40]
print("\nNumber of crisis days (VIX>40):", len(crisis_days))
print(crisis_days.head())

# -------------------------
# 6. Correlation Heatmap
# -------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------
# 7. Returns & Stationarity Test
# -------------------------
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

result = adfuller(df['returns'])
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("✅ Returns are stationary")
else:
    print("⚠️ Returns are not stationary")

plt.figure(figsize=(14,6))
plt.plot(df.index, df['returns'])
plt.title("Log Returns")
plt.show()

# -------------------------
# 8. Save cleaned dataset
# -------------------------
df.to_csv("Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_cleaned.csv")
print("\n✅ Cleaned dataset saved as data_cleaned.csv")
