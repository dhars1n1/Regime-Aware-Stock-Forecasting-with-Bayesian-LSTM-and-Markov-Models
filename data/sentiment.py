import pandas as pd
import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

# =======================
# CONFIG
# =======================
keywords_twitter = '"S&P 500" OR SPX OR "stock market" lang:en'
keywords_news = "S&P 500 OR SPX OR stock market"
start_date = "1990-01-01"
end_date = "2025-07-31"
max_tweets_per_day = 200
data_file = "Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_cleaned.csv"

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise ValueError("❌ Missing NEWSAPI_KEY in .env")

# =======================
# LOAD FINBERT
# =======================
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_score(text):
    """Return sentiment score between -1 and +1."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return probs[2].item() - probs[0].item()

# =======================
# GET NEWS SENTIMENT
# =======================
def get_news_sentiment(date):
    """Fetch headlines from NewsAPI and return average sentiment score."""
    url = f"https://newsapi.org/v2/everything?q={keywords_news}&from={date}&to={date}&language=en&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
    r = requests.get(url)
    data = r.json()
    if "articles" not in data:
        return None
    headlines = [a["title"] for a in data["articles"]]
    if not headlines:
        return None
    scores = [finbert_score(h) for h in headlines]
    return sum(scores) / len(scores)

# =======================
# GET TWITTER SENTIMENT
# =======================
def get_twitter_sentiment(date):
    """Fetch tweets and return average sentiment score."""
    query = f'{keywords_twitter} since:{date} until:{(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).date()}'
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        tweets.append(tweet.content)
        if i >= max_tweets_per_day - 1:
            break
    if not tweets:
        return None
    scores = [finbert_score(t) for t in tweets]
    return sum(scores) / len(scores)

# =======================
# HYBRID LOOP
# =======================
all_scores = []
start_dt = datetime.strptime(start_date, "%Y-%m-%d")
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

for single_date in pd.date_range(start_dt, end_dt):
    date_str = single_date.strftime("%Y-%m-%d")
    if single_date.year < 2006:
        score = get_news_sentiment(date_str)
    else:
        score = get_twitter_sentiment(date_str)
    if score is not None:
        all_scores.append({"Date": single_date.date(), "sentiment": score})
    print(f"{date_str} → {score}")

sentiment_df = pd.DataFrame(all_scores)

# =======================
# MERGE WITH MAIN DATA
# =======================
df = pd.read_csv(data_file, parse_dates=["Date"])
merged = df.merge(sentiment_df, on="Date", how="left")
merged["sentiment"] = merged["sentiment_y"].combine_first(merged["sentiment_x"])
merged.drop(columns=["sentiment_x", "sentiment_y"], inplace=True)

merged.to_csv("Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/data/data_with_sentiment.csv", index=False)
print("✅ Saved data_with_sentiment.csv with hybrid sentiment")
