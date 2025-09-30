"""
Add regimes (Crisis, Normal, Bull) to data_with_news_sentiment.csv
using custom multivariate MSM (returns + VIX).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msm.msm import baum_welch_multivariate  # custom MSM implementation

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("data_with_news_sentiment.csv", parse_dates=["Date"], index_col="Date")
print(f"Loaded dataset shape: {df.shape}")

# -------------------------
# 2. Compute log returns
# -------------------------
df["log_return"] = np.log1p(df["returns"])   # safe log(1+r)
df.dropna(inplace=True)

# -------------------------
# 3. Prepare observations for MSM
# -------------------------
# Use returns + VIX as the multivariate signal to detect regimes
obs = df[["log_return", "VIX"]].values

# -------------------------
# 4. Fit MSM (3 regimes)
# -------------------------
print("Fitting MSM with 3 regimes...")
results = baum_welch_multivariate(obs, K=3, verbose=True)

gamma = results["gamma"]      # regime probabilities
viterbi = results["viterbi"]  # hard regime assignments
means = results["means"][:, 0]  # mean returns per regime (for ranking)

# -------------------------
# 5. Add regime info to dataframe
# -------------------------
# Add regime probabilities
for k in range(gamma.shape[1]):
    df[f"regime_{k}_prob"] = gamma[:, k]

# Add hard regime assignment
df["regime_viterbi"] = viterbi

# Label regimes by sorting mean returns
regime_order = np.argsort(means)  # lowest return → highest return
labels = ["Crisis", "Normal", "Bull"]
mapping = {regime_order[i]: labels[i] for i in range(len(labels))}
df["regime_label"] = df["regime_viterbi"].map(mapping)

print("\n✅ Regimes added:")
print(df[["log_return", "VIX", "regime_viterbi", "regime_label"]].head(10))


# -------------------------
# 5b. One-hot encode regimes
# -------------------------
regime_dummies = pd.get_dummies(df["regime_label"], prefix="regime")
df = pd.concat([df, regime_dummies], axis=1)

print("\n✅ One-hot encoded regimes added:")
print(df[["regime_label"] + list(regime_dummies.columns)].head(10))

# -------------------------
# 6. Save dataset with regimes
# -------------------------
df.to_csv("data_with_regimes.csv")
print("\n✅ Saved dataset with regimes as data_with_regimes.csv")

# -------------------------
# 7. Visualization
# -------------------------
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["returns"], label="Returns", alpha=0.6, color="blue")

colors = {"Crisis": "red", "Normal": "orange", "Bull": "green"}
for label in df["regime_label"].unique():
    idx = df["regime_label"] == label
    plt.scatter(df.index[idx], df["returns"][idx], s=6, color=colors[label], label=label)

plt.title("Market Regimes (MSM on Returns + VIX)")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.show()
