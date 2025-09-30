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
df = pd.read_csv("data.csv", header=[0,1])

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
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
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
df.to_csv("data_cleaned.csv")
print("\n✅ Cleaned dataset saved as data_cleaned.csv")

# -------------------------
# 9. Remove Redundant Features (Correlation > 0.9)
# -------------------------
print("\nFeatures BEFORE redundancy cleaning:")
print(df.columns.tolist())
print("\nTotal features before:", len(df.columns))

# Compute correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify features with correlation higher than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print("\nHighly correlated features to drop (|ρ| > 0.9):")
print(to_drop)

# Drop them
df_reduced = df.drop(columns=to_drop)

print("\nFeatures AFTER redundancy cleaning:")
print(df_reduced.columns.tolist())
print("\nTotal features after:", len(df_reduced.columns))

# Save reduced dataset
df_reduced.to_csv("data_reduced.csv")
print("\n✅ Reduced dataset saved as data_reduced.csv")

# -------------------------
# 10. Correlation Heatmap After Feature Reduction
# -------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df_reduced.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap (After Reducing Features)")
plt.show()

