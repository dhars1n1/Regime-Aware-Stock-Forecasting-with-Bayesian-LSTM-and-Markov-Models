import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
# import warnings
# warnings.filterwarnings("ignore")

# Load your CSV
df = pd.read_csv("data_with_news_sentiment.csv")

# Define returns as target
returns = df['returns']

# Exogenous features to include
exog_vars = ['RSI', 'MACD_diff', 'VIX', 'Unemployment', 'FedFunds', 'sentiment_news']
exog = df[exog_vars]

# Range of regimes to test
regime_range = range(1, 11)  # 1 to 4 regimes

results = []

for k in regime_range:
    try:
        # Fit Markov Switching model with exogenous variables
        model = MarkovRegression(returns, k_regimes=k, exog=exog, trend='c', switching_variance=True)
        fit = model.fit(disp=False)
        
        results.append({
            'regimes': k,
            'AIC': fit.aic,
            'BIC': fit.bic
        })
        print(f"Regimes: {k}, AIC: {fit.aic:.2f}, BIC: {fit.bic:.2f}")
    except Exception as e:
        print(f"Failed for {k} regimes: {e}")

# Convert results to DataFrame for easier comparison
results_df = pd.DataFrame(results)

# Find optimal number of regimes based on BIC (usually preferred)
optimal = results_df.loc[results_df['BIC'].idxmin()]
print("\nOptimal number of regimes based on BIC:", optimal['regimes'])
