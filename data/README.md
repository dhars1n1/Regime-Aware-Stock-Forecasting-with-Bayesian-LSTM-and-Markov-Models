# README: Workflow for Creating the Final Dataset

This folder processes financial data, performs sentiment analysis, and applies regime-switching models to create a final enriched dataset. Below is a step-by-step guide to running the code files in the correct order, along with detailed explanations of what each file does.

---

## **1. `data/dataset_creation.py`**

### **Purpose:**
This script is the starting point for creating the dataset. It downloads financial data, computes technical indicators, and merges macroeconomic data from 2008-2024.

### **Steps Performed:**

1. **Download Financial Data:**
   - Downloads S&P 500 (`^GSPC`) and VIX (`^VIX`) data using the `yfinance` library.
   - Extracts key columns like `Open`, `High`, `Low`, `Close`, `Volume`, and `VIX`.

2. **Compute Technical Indicators:**
   - Calculates indicators such as RSI, MACD, Bollinger Bands, and On-Balance Volume (OBV) using the `ta` library.

3. **Merge Macroeconomic Data:**
   - Fetches macroeconomic data (CPI, Unemployment, Fed Funds Rate) from the FRED API.
   - Adds these as columns to the dataset.

4. **Add Extra Features:**
   - Adds placeholder columns like `sentiment` (set to 0.0 initially), `is_crisis` (binary indicator for VIX > 40), `fed_meeting`, and `earnings_season`.

5. **Save the Dataset:**
   - Saves the enriched dataset as `data.csv`.

### **Output:**
- `data.csv`: The enriched dataset with financial, technical, and macroeconomic data.

### **Run Command:**
```bash
python dataset_creation.py
```

---

## **2. `data/eda.py`**

### **Purpose:**
This script performs exploratory data analysis (EDA) on the dataset created in `data/dataset_creation.py`. It cleans the data, removes redundant features, and prepares a reduced dataset.

### **Steps Performed:**

1. **Load and Clean Data:**
   - Reads `data.csv` and flattens multi-level headers.
   - Converts the `Date` column to a datetime index.

2. **Check for Missing Values:**
   - Identifies and reports missing values in the dataset.

3. **Basic Statistics:**
   - Computes summary statistics for all columns.

4. **Visualize Key Features:**
   - Plots the S&P 500 closing price and the VIX index.
   - Highlights crisis periods where VIX > 40.

5. **Remove Redundant Features:**
   - Computes a correlation matrix and removes features with a correlation > 0.9.

6. **Save Reduced Dataset:**
   - Saves the cleaned and reduced dataset as `data_cleaned.csv` and `data_reduced.csv`.

### **Output:**
- `data_cleaned.csv`: Cleaned dataset with all features.
- `data_reduced.csv`: Reduced dataset with redundant features removed.

### **Run Command:**
```bash
python eda.py
```

---

## **3. `data/sentiment.py`**

### **Purpose:**
This script performs sentiment analysis on S&P 500 news headlines and merges the sentiment scores with the reduced dataset.

### **Steps Performed:**

1. **Load Datasets:**
   - Reads `data_reduced.csv` (main dataset) and `data/sp500_headlines.csv` (news headlines).
   - Source of news headlines: [Kaggle News Sentiment Dataset](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024?resource=download)

2. **Sentiment Analysis:**
   - Uses the VADER sentiment analyzer to compute sentiment scores for each headline.
   - Aggregates daily sentiment scores.

3. **Merge Sentiment with Main Dataset:**
   - Joins the daily sentiment scores with the main dataset.
   - Fills missing sentiment values with 0.

4. **Feature Scaling:**
   - Scales all features (except returns) using StandardScaler.

5. **Save Final Dataset:**
   - Saves the merged and scaled dataset as `data_with_news_sentiment.csv`.

### **Output:**
- `data_with_news_sentiment.csv`: Dataset with sentiment scores added.

### **Run Command:**
```bash
python sentiment.py
```

---

## **4. `data/regime.py`**

### **Purpose:**
This script applies a custom multivariate Markov Switching Model (MSM) to identify market regimes (Crisis, Normal, Bull) based on returns and VIX.

### **Steps Performed:**

1. **Load Dataset:**
   - Reads `data_with_news_sentiment.csv`.

2. **Compute Log Returns:**
   - Calculates log returns of the S&P 500.

3. **Fit MSM:**
   - Uses the `baum_welch_multivariate` function to fit a 3-regime MSM.
   - Identifies regime probabilities and hard regime assignments.

4. **Add Regime Information:**
   - Adds regime probabilities and labels (Crisis, Normal, Bull) to the dataset.
   - One-hot encodes the regime labels.

5. **Save Dataset with Regimes:**
   - Saves the dataset with regime information as `data_with_regimes.csv`.

6. **Visualize Regimes:**
   - Plots the S&P 500 returns with regime labels.

### **Output:**
- `data_with_regimes.csv`: Dataset with market regimes added.

### **Run Command:**
```bash
python regime.py
```

---

## **5. `data/check_sentiment.py`**

### **Purpose:**
This script evaluates the predictive value of sentiment scores using correlation analysis, Granger causality tests, and feature importance analysis.

### **Steps Performed:**

1. **Correlation Analysis:**
   - Computes correlations between sentiment scores and key variables (e.g., returns, VIX).

2. **Granger Causality Test:**
   - Tests whether sentiment scores can predict returns.

3. **Feature Importance:**
   - Uses a Random Forest model to evaluate the importance of sentiment scores in predicting returns.

4. **Rolling Regression:**
   - Computes rolling regression betas of returns on sentiment scores.

5. **Summary Heuristic:**
   - Summarizes the findings and provides insights into the usefulness of sentiment scores.

### **Output:**
- Visualizations of sentiment vs. returns, feature importance, and rolling regression betas.

### **Run Command:**
```bash
python check_sentiment.py
```

---

## **6. `data/no.py`**

### **Purpose:**
This script tests different numbers of regimes for a Markov Switching Model using exogenous variables (e.g., RSI, MACD, VIX, sentiment).

### **Steps Performed:**

1. **Load Dataset:**
   - Reads `data_with_news_sentiment.csv`.

2. **Fit Markov Switching Models:**
   - Fits models with 1 to 10 regimes using `MarkovRegression`.
   - Includes exogenous variables like RSI, MACD, VIX, and sentiment.

3. **Evaluate Models:**
   - Computes AIC and BIC for each model.
   - Identifies the optimal number of regimes based on BIC.

### **Output:**
- Optimal number of regimes based on BIC.

### **Run Command:**
```bash
python no.py
```

---

## **Final Dataset:**

The final dataset, `data_with_regimes.csv`, contains:

- Financial data (e.g., returns, VIX, RSI, MACD).
- Macroeconomic data (e.g., CPI, Unemployment, Fed Funds Rate).
- Sentiment scores from news headlines.
- Market regimes (Crisis, Normal, Bull) with probabilities and one-hot encoding.

---

## **Order of Execution:**

1. `data/dataset_creation.py`
2. `data/eda.py`
3. `data/sentiment.py`
4. `data/regime.py`
5. `data/check_sentiment.py` (optional, for analysis)
6. `data/no.py` (optional, for regime testing)