import pandas as pd

# =========================
# CONFIG
# =========================
sentiment_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/sentiment.xls"  # Path to your sentiment Excel
main_data_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_cleaned.csv"  # Path to your main stock data
output_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_with_sentiment.csv"

# =========================
# LOAD SENTIMENT DATA
# =========================
print("📥 Loading sentiment data...")
sent_df = pd.read_excel(sentiment_file, skiprows=4)  # Skip header rows

# Keep only relevant columns
sent_df = sent_df.iloc[:, 0:4]  # Date, Bullish, Neutral, Bearish
sent_df.columns = ["Date", "Bullish", "Neutral", "Bearish"]

# Convert date
sent_df["Date"] = pd.to_datetime(sent_df["Date"], errors="coerce")

# Remove rows without a date
sent_df = sent_df.dropna(subset=["Date"])

# Function to convert percentage strings like "36.0%" → 36.0
def pct_to_float(x):
    if isinstance(x, str) and "%" in x:
        return float(x.replace("%", ""))
    return pd.to_numeric(x, errors="coerce")

for col in ["Bullish", "Neutral", "Bearish"]:
    sent_df[col] = sent_df[col].apply(pct_to_float)

# Drop rows where sentiment columns are all NaN
sent_df = sent_df.dropna(subset=["Bullish", "Neutral", "Bearish"], how="all")

# Create sentiment score
sent_df["sentiment_score"] = sent_df["Bullish"] - sent_df["Bearish"]

print(f"✅ Sentiment data cleaned. Rows: {len(sent_df)}")

# =========================
# LOAD MAIN STOCK DATA
# =========================
print("📥 Loading stock data...")
main_df = pd.read_csv(main_data_file, parse_dates=["Date"])

# =========================
# MERGE & FORWARD-FILL
# =========================
print("🔄 Merging sentiment into stock data...")
merged = pd.merge_asof(
    main_df.sort_values("Date"),
    sent_df.sort_values("Date"),
    on="Date",
    direction="backward"
)

# Forward-fill
merged[["Bullish", "Neutral", "Bearish", "sentiment_score"]] = merged[["Bullish", "Neutral", "Bearish", "sentiment_score"]].fillna(method="ffill")

# Save result
merged.to_csv(output_file, index=False)
print(f"✅ Sentiment merged and saved to {output_file}")
