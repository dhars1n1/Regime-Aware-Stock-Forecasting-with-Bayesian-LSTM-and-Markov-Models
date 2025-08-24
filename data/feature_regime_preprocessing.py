"""
Feature engineering + Regime detection using custom Multivariate MSM from msm.py.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msm.msm import baum_welch_multivariate

def main():
    # Load cleaned data - handle both relative and absolute paths
    data_file = "data_with_sentiment.csv"
    if not os.path.exists(data_file):
        data_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_with_sentiment.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found. Please ensure data_with_sentiment.csv exists.")
        print("Run the earlier steps of the pipeline first:")
        print("1. dataset_creation.py")
        print("2. eda.py") 
        print("3. sentiment.py")
        return
    
    df = pd.read_csv(data_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    print(f"Original shape: {df.shape}")

    # Log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
    df.dropna(inplace=True)

    # Normalize continuous features (z-score)
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD_diff",
                    "BB_high", "BB_low", "OBV", "VIX", "CPI", "Unemployment", "FedFunds"]
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    # Prepare multivariate observations (log_return + VIX)
    obs = df[["log_return", "VIX"]].values

    # Fit MSM
    print("Fitting custom multivariate MSM...")
    results = baum_welch_multivariate(obs, K=3, verbose=True)

    gamma = results["gamma"]
    viterbi = results["viterbi"]

    for k in range(gamma.shape[1]):
        df[f"regime_{k}_prob"] = gamma[:, k]
    df["regime_viterbi"] = viterbi

    # Label regimes
    means = results["means"][:, 0]  # use returns dimension to rank
    regime_order = np.argsort(means)
    labels = ["Crisis", "Normal", "Bull"]
    mapping = {regime_order[i]: labels[i] for i in range(len(labels))}
    df["regime_label"] = df["regime_viterbi"].map(mapping)

    # Save - try relative path first, then absolute
    output_file = "data_with_regimes.csv"
    try:
        df.to_csv(output_file)
        print(f"✅ Saved {output_file}")
    except:
        output_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_with_regimes.csv"
        df.to_csv(output_file)
        print(f"✅ Saved {output_file}")

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df["Close"], label="Close", alpha=0.6)
    colors = {"Crisis": "red", "Normal": "orange", "Bull": "green"}
    for label in df["regime_label"].unique():
        idx = df["regime_label"] == label
        plt.scatter(df.index[idx], df["Close"][idx], s=6, color=colors[label], label=label)
    plt.legend()
    plt.title("Market Regimes (Custom Multivariate MSM)")
    plt.show()

if __name__ == "__main__":
    main()
