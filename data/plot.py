import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv("data_final_with_regime.csv", parse_dates=['Date'], index_col='Date')

# =========================
# 1. Create dominant regime label
# =========================
df['dominant_regime'] = df[['regime_prob_0', 'regime_prob_1', 'regime_prob_2']].idxmax(axis=1)

# =========================
# 2. Define lag windows
# =========================
short_lags = range(1, 6)
long_lags = range(10, 21)

# =========================
# 3. Compute lagged correlation + p-values
# =========================
def compute_lagged_corr_pval(df, lags):
    results = []
    for lag in lags:
        shifted_returns = df['log_return'].shift(-lag)
        valid = df[['sentiment_score']].join(shifted_returns).dropna()
        if len(valid) > 10:  # Ensure enough samples
            corr, pval = pearsonr(valid['sentiment_score'], valid['log_return'])
        else:
            corr, pval = np.nan, np.nan
        results.append({'Lag_Days': lag, 'Correlation': corr, 'p_value': pval})
    return pd.DataFrame(results)

short_corr = compute_lagged_corr_pval(df, short_lags)
long_corr = compute_lagged_corr_pval(df, long_lags)

# =========================
# 4. Regime-wise correlation with p-values
# =========================
regime_corrs = {}
for regime in df['dominant_regime'].unique():
    subset = df[df['dominant_regime'] == regime]
    regime_corrs[regime] = compute_lagged_corr_pval(subset, short_lags)

# =========================
# 5. Display results
# =========================
print("\n" + "="*60)
print("SHORT-TERM LAGGED CORRELATION + P-VALUE (1–5 days)")
print("="*60)
print(short_corr.to_string(index=False, formatters={'p_value': '{:.4f}'.format}))

print("\n" + "="*60)
print("LONG-TERM LAGGED CORRELATION + P-VALUE (10–20 days)")
print("="*60)
print(long_corr.to_string(index=False, formatters={'p_value': '{:.4f}'.format}))

print("\n" + "="*60)
print("REGIME-WISE SHORT-TERM CORRELATION + P-VALUE")
print("="*60)
for regime, corr_df in regime_corrs.items():
    print(f"\n{regime}")
    print(corr_df.to_string(index=False, formatters={'p_value': '{:.4f}'.format}))

# =========================
# 6. Visualization with significance markers
# =========================
plt.figure(figsize=(10, 5))
plt.plot(short_corr['Lag_Days'], short_corr['Correlation'], 'o-', label='Short-term (1–5)')
plt.plot(long_corr['Lag_Days'], long_corr['Correlation'], 'x--', label='Long-term (10–20)')

# Highlight significant points (p < 0.05)
sig_short = short_corr[short_corr['p_value'] < 0.05]
sig_long = long_corr[long_corr['p_value'] < 0.05]
plt.scatter(sig_short['Lag_Days'], sig_short['Correlation'], color='green', s=80, label='Significant (short)')
plt.scatter(sig_long['Lag_Days'], sig_long['Correlation'], color='red', s=80, label='Significant (long)')

plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
plt.title("Lagged Correlation & Significance: Sentiment vs Future log_return")
plt.xlabel("Lag (days)")
plt.ylabel("Pearson Correlation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
