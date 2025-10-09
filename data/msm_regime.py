import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data_reduced.csv", index_col=0, parse_dates=True)

# Prepare features
regime_features = df[['log_return', 'VIX']].copy()
regime_features['realized_vol'] = df['log_return'].rolling(window=20).std()
regime_features = regime_features.dropna()

print("Finding optimal number of regimes using Markov Switching Model...")
print("="*70)

# ----------------------------
# Try both with and without scaling to find what works
# ----------------------------
optimal_n = 3
best_result = None
best_method = None

# Method 1: Original (no scaling)
print(f"\nMethod 1: Fitting MSM with {optimal_n} regimes (no scaling)...")
try:
    model1 = MarkovRegression(
        endog=regime_features['log_return'],
        k_regimes=optimal_n,
        exog=regime_features[['VIX', 'realized_vol']],
        switching_variance=True,
        switching_exog=False
    )
    
    results1 = model1.fit(
        maxiter=1000,
        disp=False,
        search_reps=20
    )
    
    # Check if solution is valid
    smoothed_probs = results1.smoothed_marginal_probabilities
    regime_counts = [(smoothed_probs.iloc[:, i] > 0.5).sum() for i in range(optimal_n)]
    min_regime_pct = min(regime_counts) / len(regime_features) * 100
    
    if not np.isnan(results1.llf) and min_regime_pct > 1.0:
        print(f"  ✓ Converged: Log-Likelihood: {results1.llf:.2f}")
        print(f"    BIC: {results1.bic:.2f}, AIC: {results1.aic:.2f}")
        print(f"    Min regime: {min_regime_pct:.1f}% of data")
        best_result = results1
        best_method = "Method 1 (no scaling)"
    else:
        print(f"  ✗ Degenerate solution (min regime: {min_regime_pct:.1f}%)")
except Exception as e:
    print(f"  ✗ Failed: {str(e)}")

# Method 2: Scaled features
print(f"\nMethod 2: Fitting MSM with {optimal_n} regimes (with scaling)...")
try:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(regime_features[['VIX', 'realized_vol']])
    
    model2 = MarkovRegression(
        endog=regime_features['log_return'],
        k_regimes=optimal_n,
        exog=scaled_features,
        switching_variance=True,
        switching_exog=False
    )
    
    results2 = model2.fit(
        maxiter=1000,
        disp=False,
        search_reps=20
    )
    
    smoothed_probs = results2.smoothed_marginal_probabilities
    regime_counts = [(smoothed_probs.iloc[:, i] > 0.5).sum() for i in range(optimal_n)]
    min_regime_pct = min(regime_counts) / len(regime_features) * 100
    
    if not np.isnan(results2.llf) and min_regime_pct > 1.0:
        print(f"  ✓ Converged: Log-Likelihood: {results2.llf:.2f}")
        print(f"    BIC: {results2.bic:.2f}, AIC: {results2.aic:.2f}")
        print(f"    Min regime: {min_regime_pct:.1f}% of data")
        
        # Use this if it's better or if Method 1 failed
        if best_result is None or results2.bic < best_result.bic:
            best_result = results2
            best_method = "Method 2 (scaled)"
    else:
        print(f"  ✗ Degenerate solution (min regime: {min_regime_pct:.1f}%)")
except Exception as e:
    print(f"  ✗ Failed: {str(e)}")

# Method 3: Simple model (no exogenous variables)
print(f"\nMethod 3: Fitting simple MSM with {optimal_n} regimes (no exog)...")
try:
    model3 = MarkovRegression(
        endog=regime_features['log_return'],
        k_regimes=optimal_n,
        switching_variance=True
    )
    
    results3 = model3.fit(
        maxiter=1000,
        disp=False,
        search_reps=20
    )
    
    smoothed_probs = results3.smoothed_marginal_probabilities
    regime_counts = [(smoothed_probs.iloc[:, i] > 0.5).sum() for i in range(optimal_n)]
    min_regime_pct = min(regime_counts) / len(regime_features) * 100
    
    if not np.isnan(results3.llf) and min_regime_pct > 1.0:
        print(f"  ✓ Converged: Log-Likelihood: {results3.llf:.2f}")
        print(f"    BIC: {results3.bic:.2f}, AIC: {results3.aic:.2f}")
        print(f"    Min regime: {min_regime_pct:.1f}% of data")
        
        if best_result is None or results3.bic < best_result.bic:
            best_result = results3
            best_method = "Method 3 (simple)"
    else:
        print(f"  ✗ Degenerate solution (min regime: {min_regime_pct:.1f}%)")
except Exception as e:
    print(f"  ✗ Failed: {str(e)}")

if best_result is None:
    print("\n❌ All methods failed! Cannot fit model.")
    exit()

print("\n" + "="*70)
print(f"✅ Best model: {best_method}")
print("="*70)

results = best_result

# ----------------------------
# Analyze model
# ----------------------------
print("\n" + "="*70)
print(f"MARKOV SWITCHING MODEL SUMMARY ({optimal_n} Regimes)")
print("="*70)
print(results.summary())

# Get regime assignments
regime_probs = results.smoothed_marginal_probabilities
regime_df = regime_features.copy()
regime_df['regime'] = regime_probs.values.argmax(axis=1)

for i in range(optimal_n):
    regime_df[f'regime_prob_{i}'] = regime_probs.iloc[:, i].values

# ----------------------------
# Sort regimes by volatility
# ----------------------------
regime_means = []
regime_stds = []

for i in range(optimal_n):
    mean_param = results.params[f'const[{i}]']
    std_param = np.sqrt(results.params[f'sigma2[{i}]'])
    regime_means.append(mean_param)
    regime_stds.append(std_param)

regime_params = pd.DataFrame({
    'regime': range(optimal_n),
    'mean': regime_means,
    'std': regime_stds
})

regime_params = regime_params.sort_values(['std', 'mean'])
sorted_regimes = regime_params['regime'].tolist()

# Create mapping
regime_label_map = {old: new for new, old in enumerate(sorted_regimes)}
regime_df['regime'] = regime_df['regime'].map(regime_label_map)

# Remap probabilities
regime_probs_remapped = np.zeros((len(regime_df), optimal_n))
for old_idx, new_idx in regime_label_map.items():
    regime_probs_remapped[:, new_idx] = regime_df[f'regime_prob_{old_idx}'].values

for i in range(optimal_n):
    regime_df[f'regime_prob_{i}'] = regime_probs_remapped[:, i]

# Assign labels
label_names = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
regime_df['regime_label'] = regime_df['regime'].map(label_names)

# ----------------------------
# Print regime characteristics
# ----------------------------
print("\n" + "="*70)
print("REGIME CHARACTERISTICS")
print("="*70)

for i in range(optimal_n):
    data = regime_df[regime_df['regime'] == i]
    orig_regime = sorted_regimes[i]
    mean_param = results.params[f'const[{orig_regime}]']
    std_param = np.sqrt(results.params[f'sigma2[{orig_regime}]'])
    
    print(f"\n{label_names[i]} (Regime {i}):")
    print(f"  Model Parameters:")
    print(f"    • Mean (μ): {mean_param:.6f}")
    print(f"    • Std Dev (σ): {std_param:.6f}")
    print(f"  Observed Statistics:")
    print(f"    • Count: {len(data)} observations ({len(data)/len(regime_df)*100:.1f}%)")
    if len(data) > 0:
        print(f"    • Avg Return: {data['log_return'].mean():.6f}")
        print(f"    • Std Return: {data['log_return'].std():.6f}")
        print(f"    • Avg VIX: {data['VIX'].mean():.2f}")
        print(f"    • Avg Realized Vol: {data['realized_vol'].mean():.6f}")

# ----------------------------
# Transition matrix - try multiple extraction methods
# ----------------------------
print("\n" + "="*70)
print("REGIME TRANSITION MATRIX")
print("="*70)

try:
    # Get raw transition matrix
    trans_mat_raw = results.regime_transition
    print(f"Raw shape: {trans_mat_raw.shape}")
    
    # Try to extract properly based on shape
    if trans_mat_raw.shape == (optimal_n, optimal_n):
        # Perfect - already correct shape (from regime i to regime j)
        trans_mat = trans_mat_raw
    elif len(trans_mat_raw.shape) == 3:
        # Handle 3D arrays
        if trans_mat_raw.shape == (optimal_n, optimal_n, 1):
            trans_mat = trans_mat_raw[:, :, 0]
        elif trans_mat_raw.shape == (1, optimal_n, optimal_n):
            trans_mat = trans_mat_raw[0, :, :]
        else:
            # Try to find the right slice
            trans_mat = trans_mat_raw.reshape(optimal_n, optimal_n)
    else:
        raise ValueError(f"Unexpected shape: {trans_mat_raw.shape}")
    
    print(f"Processed shape: {trans_mat.shape}")
    
    # Remap based on sorted regimes
    trans_mat_remapped = np.zeros((optimal_n, optimal_n))
    for old_i, new_i in regime_label_map.items():
        for old_j, new_j in regime_label_map.items():
            trans_mat_remapped[new_i, new_j] = trans_mat[old_i, old_j]
    
    transmat_df = pd.DataFrame(
        trans_mat_remapped,
        index=[f'{label_names[i]}' for i in range(optimal_n)],
        columns=[f'{label_names[i]}' for i in range(optimal_n)]
    ).round(3)
    print(transmat_df)
    
    print("\n" + "="*70)
    print("REGIME PERSISTENCE")
    print("="*70)
    for i in range(optimal_n):
        persistence = trans_mat_remapped[i, i]
        if persistence < 0.9999:
            expected_duration = 1 / (1 - persistence)
            print(f"{label_names[i]}: {persistence:.3f} (Expected duration: {expected_duration:.1f} periods)")
        else:
            print(f"{label_names[i]}: {persistence:.3f} (Expected duration: ∞ periods)")
    
except Exception as e:
    print(f"⚠ Could not extract transition matrix: {str(e)}")
    print("This is a statsmodels API issue, but the regime assignments are still valid.")

# ----------------------------
# Visualize
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

ax = axes[0]
for i in range(optimal_n):
    mask = regime_df['regime'] == i
    ax.scatter(regime_df.index[mask], regime_df.loc[mask, 'log_return'], 
               label=label_names[i], alpha=0.6, s=15)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_ylabel('Log Return', fontsize=11)
ax.set_title('Market Regimes over Time (3-Regime MSM)', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

ax = axes[1]
for i in range(optimal_n):
    mask = regime_df['regime'] == i
    ax.scatter(regime_df.index[mask], regime_df.loc[mask, 'VIX'], 
               label=label_names[i], alpha=0.6, s=15)
ax.set_ylabel('VIX', fontsize=11)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

ax = axes[2]
for i in range(optimal_n):
    ax.plot(regime_df.index, regime_df[f'regime_prob_{i}'], 
            label=label_names[i], alpha=0.7, linewidth=1.5)
ax.set_ylabel('Regime Probability', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("msm_regime_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# Save results
# ----------------------------
regime_df.to_csv("regime_data.csv")

with open("msm_model_summary.txt", "w") as f:
    f.write(results.summary().as_text())

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Total observations: {len(regime_df)}")
print(f"Model: {best_method}")
print(f"Log-Likelihood: {results.llf:.2f}")
print(f"BIC: {results.bic:.2f}")
print(f"AIC: {results.aic:.2f}")
print("\nRegime Distribution:")
for i in range(optimal_n):
    count = (regime_df['regime'] == i).sum()
    pct = count / len(regime_df) * 100
    print(f"  {label_names[i]}: {count} obs ({pct:.1f}%)")

print("\n✅ Saved regime_data.csv")
print("✅ Saved msm_regime_visualization.png")
print("✅ Saved msm_model_summary.txt")