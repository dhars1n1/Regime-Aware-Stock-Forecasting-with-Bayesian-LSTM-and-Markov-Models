import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from datetime import datetime

# Load FinBERT
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

# Load news headlines dataset
# Assuming format: Date, Headline columns
news_df = pd.read_csv("news_headlines.csv")
news_df['Date'] = pd.to_datetime(news_df['Date'])

print(f"Loaded {len(news_df)} news headlines")
print(f"Date range: {news_df['Date'].min()} to {news_df['Date'].max()}")

def get_sentiment_scores(text):
    """Get FinBERT sentiment scores (negative, neutral, positive)"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
        
        # Returns: [negative, neutral, positive]
        scores = probs[0].cpu().numpy()
        return scores[0], scores[1], scores[2]
    except:
        return np.nan, np.nan, np.nan

# Process headlines in batches
print("\nExtracting sentiment scores...")
sentiments = []

for idx, row in tqdm(news_df.iterrows(), total=len(news_df)):
    neg, neu, pos = get_sentiment_scores(row['Title'])
    sentiments.append({
        'Date': row['Date'],
        'sentiment_neg': neg,
        'sentiment_neu': neu,
        'sentiment_pos': pos
    })

sentiment_df = pd.DataFrame(sentiments)

# Calculate daily aggregate sentiment
daily_sentiment = sentiment_df.groupby('Date').agg({
    'sentiment_neg': 'mean',
    'sentiment_neu': 'mean', 
    'sentiment_pos': 'mean'
}).reset_index()

# Composite sentiment score: positive - negative
daily_sentiment['sentiment_score'] = (
    daily_sentiment['sentiment_pos'] - daily_sentiment['sentiment_neg']
)

# Add sentiment polarity (more interpretable)
daily_sentiment['sentiment_polarity'] = daily_sentiment['sentiment_score'].apply(
    lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
)

print(f"\nProcessed sentiment for {len(daily_sentiment)} unique dates")
print("\nSentiment statistics:")
print(daily_sentiment[['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_score']].describe())

# Save sentiment data
daily_sentiment.to_csv("sentiment_scores.csv", index=False)
print("\n✅ Saved sentiment_scores.csv")