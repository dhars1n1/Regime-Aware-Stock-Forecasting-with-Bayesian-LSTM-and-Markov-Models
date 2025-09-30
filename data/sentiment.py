import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.preprocessing import StandardScaler

# Download VADER lexicon if not already
nltk.download('vader_lexicon')

# -------------------------
# 1. Load datasets
# -------------------------
df = pd.read_csv("data_reduced.csv", index_col=0, parse_dates=True)  # your enriched dataset
news = pd.read_csv("sp500_headlines.csv")  # Kaggle dataset (2008–2024)

# Convert date
news['Date'] = pd.to_datetime(news['Date'])

# -------------------------
# 2. Sentiment analysis
# -------------------------
sia = SentimentIntensityAnalyzer()

def get_daily_sentiment(row):
    headlines = row.drop('Date').dropna().astype(str).tolist()
    scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    return sum(scores)/len(scores) if scores else 0

news['sentiment_news'] = news.apply(get_daily_sentiment, axis=1)

# Keep only Date + sentiment
news_sent = news[['Date','sentiment_news']].set_index('Date')

# -------------------------
# 3. Merge with main dataset
# -------------------------
df = df.join(news_sent, how='left')

# Fill NaN (before 2008 or missing headlines) with 0
df['sentiment_news'] = df['sentiment_news'].fillna(0)

print("\nSample sentiment columns (before dropping):")
print(df[['sentiment', 'sentiment_news']].head(20))

# -------------------------
# 4. Drop columns that are fully zero
# -------------------------
zero_cols = [col for col in df.columns if (df[col] == 0).all()]
print("\nColumns with only zeros:", zero_cols)

df = df.drop(columns=zero_cols)

# -------------------------
# 5. Feature scaling
# -------------------------
scaler = StandardScaler()

# Don’t scale the target variable (returns), but scale exogenous features
features_to_scale = [col for col in df.columns if col != 'returns']

df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\n✅ Features scaled using StandardScaler")
print(df_scaled.head())

# -------------------------
# 6. Save merged + scaled dataset
# -------------------------
df_scaled.to_csv("data_with_news_sentiment.csv")
print("\n✅ Final dataset saved as data_with_news_sentiment.csv")
print("Final dataset shape:", df_scaled.shape)
