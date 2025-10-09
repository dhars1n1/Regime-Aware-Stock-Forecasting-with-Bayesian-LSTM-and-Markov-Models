import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os

# Create output folder
os.makedirs("eda_results", exist_ok=True)
print("Created folder: eda_results/")

# Load data - handle yfinance multi-index format
df = pd.read_csv("data.csv")

# Check if first row contains ticker info
if df.iloc[0].astype(str).str.contains('GSPC|Ticker', case=False).any():
    df = df.iloc[1:]  # Skip ticker row
    
# Set date as index
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
df = df.set_index(df.columns[0])
df.index.name = 'Date'

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with invalid dates
df = df[df.index.notna()]

print("Shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())
print("\nColumns:", df.columns.tolist())

# Missing values
missing = df.isna().sum()
if missing.sum() > 0:
    print("\nMissing values:\n", missing[missing > 0])
else:
    print("\n✅ No missing values")

# Summary statistics
print("\nSummary:\n", df.describe().T)

# =============================
# 1. TIME SERIES VISUALIZATION
# =============================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df.index, df['Close'], linewidth=0.8)
axes[0].set_title("S&P 500 Close Price", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Price")
axes[0].grid(alpha=0.3)

axes[1].plot(df.index, df['VIX'], color='orange', linewidth=0.8)
axes[1].axhline(40, color='red', linestyle='--', linewidth=1, label='Crisis (VIX>40)')
axes[1].set_title("VIX (Volatility Index)", fontsize=12, fontweight='bold')
axes[1].set_ylabel("VIX")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(df.index, df['target'], linewidth=0.5, alpha=0.7)
axes[2].set_title("Target: Next Day Log Returns", fontsize=12, fontweight='bold')
axes[2].set_ylabel("Log Return")
axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/01_time_series.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: eda_results/01_time_series.png")

# =============================
# 2. TARGET VARIABLE ANALYSIS
# =============================
print(f"\n{'='*50}")
print("TARGET VARIABLE ANALYSIS")
print(f"{'='*50}")
print(f"\nTarget stats:\n{df['target'].describe()}")

# Stationarity test
result = adfuller(df['target'].dropna())
print(f"\nADF test - Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
print("✅ Target is stationary (good for LSTM)" if result[1] < 0.05 else "⚠️ Target not stationary")

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['target'].dropna(), bins=100, edgecolor='black', alpha=0.7)
axes[0].set_title("Distribution of Target", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Log Return")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df['target'].mean(), color='red', linestyle='--', label=f'Mean: {df["target"].mean():.6f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Q-Q plot for normality check
stats.probplot(df['target'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/02_target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/02_target_distribution.png")

# Check for outliers
Q1 = df['target'].quantile(0.25)
Q3 = df['target'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['target'] < Q1 - 3*IQR) | (df['target'] > Q3 + 3*IQR)]
print(f"\nExtreme outliers (3×IQR): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# =============================
# 3. AUTOCORRELATION ANALYSIS
# =============================
print(f"\n{'='*50}")
print("AUTOCORRELATION ANALYSIS")
print(f"{'='*50}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# ACF and PACF for target
plot_acf(df['target'].dropna(), lags=40, ax=axes[0, 0])
axes[0, 0].set_title("ACF: Target (Next Day Log Returns)", fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

plot_pacf(df['target'].dropna(), lags=40, ax=axes[0, 1])
axes[0, 1].set_title("PACF: Target (Next Day Log Returns)", fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# ACF and PACF for Close price
plot_acf(df['Close'].dropna(), lags=40, ax=axes[1, 0])
axes[1, 0].set_title("ACF: Close Price", fontsize=11, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

plot_pacf(df['Close'].dropna(), lags=40, ax=axes[1, 1])
axes[1, 1].set_title("PACF: Close Price", fontsize=11, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/03_autocorrelation.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/03_autocorrelation.png")

# Calculate ACF values
acf_vals = acf(df['target'].dropna(), nlags=20)
print("\nACF values (first 10 lags):")
for i, val in enumerate(acf_vals[:11]):
    print(f"Lag {i}: {val:.4f}")

# =============================
# 4. ROLLING STATISTICS
# =============================
print(f"\n{'='*50}")
print("ROLLING STATISTICS ANALYSIS")
print(f"{'='*50}")

# Calculate rolling statistics
windows = [20, 60, 252]  # ~1 month, 3 months, 1 year
rolling_stats = {}

for window in windows:
    rolling_stats[f'mean_{window}'] = df['target'].rolling(window=window).mean()
    rolling_stats[f'std_{window}'] = df['target'].rolling(window=window).std()

# Plot rolling mean
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df.index, df['target'], label='Target', alpha=0.3, linewidth=0.5)
for window in windows:
    axes[0].plot(df.index, rolling_stats[f'mean_{window}'], 
                label=f'{window}-day Mean', linewidth=1.5)
axes[0].set_title("Rolling Mean of Target", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Log Return")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot rolling std (volatility)
for window in windows:
    axes[1].plot(df.index, rolling_stats[f'std_{window}'], 
                label=f'{window}-day Std', linewidth=1.5)
axes[1].set_title("Rolling Standard Deviation (Volatility)", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Std Dev")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/04_rolling_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/04_rolling_statistics.png")

# Rolling statistics for key features
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

features_to_plot = ['RSI', 'VIX', 'MACD_diff', 'Volume']
for idx, feature in enumerate(features_to_plot):
    if feature in df.columns:
        ax = axes[idx // 2, idx % 2]
        ax.plot(df.index, df[feature], alpha=0.4, linewidth=0.5, label='Original')
        ax.plot(df.index, df[feature].rolling(window=60).mean(), 
               linewidth=1.5, label='60-day Mean')
        ax.set_title(f"{feature} with Rolling Mean", fontsize=11, fontweight='bold')
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/05_feature_rolling_stats.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/05_feature_rolling_stats.png")

# =============================
# 5. TRAIN/VAL/TEST SPLIT VISUALIZATION
# =============================
print(f"\n{'='*50}")
print("TRAIN/VALIDATION/TEST SPLIT")
print(f"{'='*50}")

# Split ratios: 70% train, 15% validation, 15% test
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_size = len(df) - train_size - val_size

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size + val_size]
test_data = df.iloc[train_size + val_size:]

print(f"\nTotal samples: {len(df)}")
print(f"Train: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%) | {train_data.index[0]} to {train_data.index[-1]}")
print(f"Val:   {len(val_data)} ({len(val_data)/len(df)*100:.1f}%) | {val_data.index[0]} to {val_data.index[-1]}")
print(f"Test:  {len(test_data)} ({len(test_data)/len(df)*100:.1f}%) | {test_data.index[0]} to {test_data.index[-1]}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Split on Close price
axes[0].plot(train_data.index, train_data['Close'], label='Train', linewidth=1)
axes[0].plot(val_data.index, val_data['Close'], label='Validation', linewidth=1)
axes[0].plot(test_data.index, test_data['Close'], label='Test', linewidth=1)
axes[0].set_title("Train/Val/Test Split - Close Price", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Split on target
axes[1].plot(train_data.index, train_data['target'], label='Train', alpha=0.7, linewidth=0.8)
axes[1].plot(val_data.index, val_data['target'], label='Validation', alpha=0.7, linewidth=0.8)
axes[1].plot(test_data.index, test_data['target'], label='Test', alpha=0.7, linewidth=0.8)
axes[1].set_title("Train/Val/Test Split - Target", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Log Return")
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1].legend()
axes[1].grid(alpha=0.3)

# Split on VIX
axes[2].plot(train_data.index, train_data['VIX'], label='Train', linewidth=1)
axes[2].plot(val_data.index, val_data['VIX'], label='Validation', linewidth=1)
axes[2].plot(test_data.index, test_data['VIX'], label='Test', linewidth=1)
axes[2].axhline(40, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[2].set_title("Train/Val/Test Split - VIX", fontsize=12, fontweight='bold')
axes[2].set_ylabel("VIX")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_results/06_train_val_test_split.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/06_train_val_test_split.png")

# Statistics by split
print("\nTarget statistics by split:")
print(f"Train - Mean: {train_data['target'].mean():.6f}, Std: {train_data['target'].std():.6f}")
print(f"Val   - Mean: {val_data['target'].mean():.6f}, Std: {val_data['target'].std():.6f}")
print(f"Test  - Mean: {test_data['target'].mean():.6f}, Std: {test_data['target'].std():.6f}")

# =============================
# 6. CORRELATION ANALYSIS
# =============================
print(f"\n{'='*50}")
print("CORRELATION ANALYSIS")
print(f"{'='*50}")

# Correlation heatmap
fig = plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=False, 
            fmt=".2f", square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_results/07_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/07_correlation_heatmap.png")

# Target correlations
target_corr = corr['target'].drop('target').sort_values(key=abs, ascending=False)
print("\nTop 10 features correlated with target:")
print(target_corr.head(10))

# Plot target correlations
fig, ax = plt.subplots(figsize=(10, 8))
target_corr.head(15).plot(kind='barh', ax=ax)
ax.set_title("Top 15 Features Correlated with Target", fontsize=12, fontweight='bold')
ax.set_xlabel("Correlation")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eda_results/08_target_correlations.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: eda_results/08_target_correlations.png")

# Remove highly correlated features (>0.95)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col].abs() > 0.95)]
to_drop = [col for col in to_drop if col != 'target']

if to_drop:
    print(f"\nDropping highly correlated features (|ρ|>0.95): {to_drop}")
    df_reduced = df.drop(columns=to_drop)
else:
    print("\nNo highly correlated features to drop")
    df_reduced = df.copy()

print(f"\nFeatures: {len(df.columns)} → {len(df_reduced.columns)}")

# =============================
# 7. SAVE PROCESSED DATA
# =============================
df_reduced.to_csv("data_reduced.csv")
print("\n✅ Saved data_reduced.csv")
print(f"\n{'='*50}")
print("EDA COMPLETE - All plots saved in eda_results/")
print(f"{'='*50}")