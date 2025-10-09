import pandas as pd
import numpy as np

print("="*60)
print("MERGING ALL FEATURES")
print("="*60)

# ----------------------------
# 1. Load base data
# ----------------------------
df = pd.read_csv("data_reduced.csv", index_col=0, parse_dates=True)
print(f"\n1. Base data: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# ----------------------------
# 2. Load sentiment scores
# ----------------------------
try:
    sentiment_df = pd.read_csv("sentiment_scores.csv", parse_dates=['Date'])
    sentiment_df = sentiment_df.set_index('Date')
    print(f"\n2. Sentiment data: {sentiment_df.shape}")
    
    # Merge sentiment (forward fill)
    df = df.merge(sentiment_df[['sentiment_score']], 
                  left_index=True, right_index=True, how='left')
    df['sentiment_score'] = df['sentiment_score'].ffill()
    print(f"   Added: sentiment_score")
except FileNotFoundError:
    print("\n⚠️ sentiment_scores.csv not found, skipping...")

# ----------------------------
# 3. Load regime probabilities
# ----------------------------
regime_cols = []
try:
    regime_df = pd.read_csv("regime_data.csv", index_col=0, parse_dates=True)
    print(f"\n3. Regime data: {regime_df.shape}")
    
    # Keep only regime probabilities
    regime_cols = ['regime_prob_0', 'regime_prob_1', 'regime_prob_2']
    df = df.merge(regime_df[regime_cols], 
                  left_index=True, right_index=True, how='left')
    print(f"   Added: {', '.join(regime_cols)}")
except FileNotFoundError:
    print("\n⚠️ regime_data.csv not found, skipping...")

# ----------------------------
# 4. Select base + sentiment columns
# ----------------------------
base_cols = [
    'Open', 'Volume', 'RSI', 'MACD_diff', 'OBV', 'VIX',
    'Unemployment', 'FedFunds', 'log_return', 'target', 'sentiment_score'
]

# ----------------------------
# 5. Create datasets
# ----------------------------
# Dataset with regime probabilities
cols_with_regime = base_cols + regime_cols
df_with_regime = df[cols_with_regime].dropna()
df_with_regime.to_csv("data_final_with_regime.csv")
print(f"\n✅ Saved data_final_with_regime.csv ({df_with_regime.shape[1]} features)")

# Dataset without regime probabilities
df_without_regime = df[base_cols].dropna()
df_without_regime.to_csv("data_final_without_regime.csv")
print(f"✅ Saved data_final_without_regime.csv ({df_without_regime.shape[1]} features)")

# ----------------------------
# 6. Print feature summary
# ----------------------------
print("\n" + "="*60)
print("FEATURE SUMMARY (WITH REGIME)")
print("="*60)
for i, col in enumerate(df_with_regime.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "="*60)
print("FEATURE SUMMARY (WITHOUT REGIME)")
print("="*60)
for i, col in enumerate(df_without_regime.columns, 1):
    print(f"  {i:2d}. {col}")

print("\nReady for LSTM training!")

# ----------------------------
# 6. Correlation Analysis
# ----------------------------
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

if 'sentiment_score' in df.columns:
    # Drop rows with NaNs in relevant columns
    corr_df = df[['sentiment_score', 'log_return', 'target']].dropna()
    
    # Compute pairwise correlations
    corr_matrix = corr_df.corr()
    sentiment_log_corr = corr_matrix.loc['sentiment_score', 'log_return']
    sentiment_target_corr = corr_matrix.loc['sentiment_score', 'target']

    print(f"Correlation between sentiment_score and log_return: {sentiment_log_corr:.4f}")
    print(f"Correlation between sentiment_score and target:     {sentiment_target_corr:.4f}")
else:
    print("⚠️ sentiment_score not found in dataset. Skipping correlation analysis.")
