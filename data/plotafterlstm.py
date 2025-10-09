import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# CONFIG
# ============================
CSV_FILE = 'bayesian_lstm_predictions_regularized.csv'  # your predictions CSV
REGIME_COLUMN = 'regime'  # set to None if no regime column

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(CSV_FILE)
print(f"✅ Loaded {len(df)} rows from '{CSV_FILE}'")
print(df.head())

# ============================
# RESIDUAL ANALYSIS
# ============================
df['residual'] = df['predicted_mean'] - df['actual_log_return']

# Residual histogram
plt.figure(figsize=(8,5))
sns.histplot(df['residual'], bins=50, kde=True, color='orange')
plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Residual Distribution (Predicted - Actual)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Rolling MAE
rolling_window = 50
df['abs_error'] = np.abs(df['residual'])
df['rolling_mae'] = df['abs_error'].rolling(rolling_window).mean()

plt.figure(figsize=(10,5))
plt.plot(df['time_step'], df['rolling_mae'], color='blue', linewidth=2)
plt.title(f'Rolling MAE (window={rolling_window})')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ============================
# UNCERTAINTY VISUALIZATION
# ============================
plt.figure(figsize=(12,6))
plt.plot(df['time_step'], df['predicted_mean'], color='orange', label='Predicted')
plt.plot(df['time_step'], df['actual_log_return'], color='red', alpha=0.6, label='Actual')
plt.fill_between(df['time_step'], df['lower_bound_95'], df['upper_bound_95'],
                 color='orange', alpha=0.2, label='95% Credible Interval')
plt.title('Predictions with Uncertainty')
plt.xlabel('Time Step')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Plot prediction uncertainty (std)
plt.figure(figsize=(10,5))
plt.plot(df['time_step'], df['uncertainty_std'], color='orange', linewidth=2, label='Std Dev')
plt.axhline(df['actual_log_return'].std(), color='red', linestyle='--', linewidth=2, label='Actual Return Std')
plt.title('Prediction Uncertainty (Std Dev)')
plt.xlabel('Time Step')
plt.ylabel('Std Dev')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ============================
# REGIME-WISE ANALYSIS (OPTIONAL)
# ============================
# ============================
# DYNAMIC REGIME CREATION
# ============================
quantiles = df['actual_log_return'].quantile([0.33, 0.66]).values
df['regime'] = pd.cut(
    df['actual_log_return'],
    bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
    labels=['Bear', 'Neutral', 'Bull']
)

print("\n📊 Regime distribution:")
print(df['regime'].value_counts())

# Regime-wise metrics
regimes = df['regime'].unique()
metrics = []
for r in regimes:
    subset = df[df['regime'] == r]
    mae = np.mean(np.abs(subset['residual']))
    mse = np.mean(subset['residual']**2)
    coverage = np.mean(subset['within_ci']) * 100
    mean_uncertainty = subset['uncertainty_std'].mean()
    metrics.append((r, len(subset), mae, mse, coverage, mean_uncertainty))
    print(f" Regime {r}: N={len(subset)}, MAE={mae:.6f}, MSE={mse:.6f}, Coverage={coverage:.2f}%, Avg Std={mean_uncertainty:.6f}")

# Bar plot: MAE per regime
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.barplot(x=[m[0] for m in metrics], y=[m[2] for m in metrics], palette='viridis')
plt.title('Regime-wise MAE')
plt.xlabel('Regime')
plt.ylabel('MAE')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.show()
