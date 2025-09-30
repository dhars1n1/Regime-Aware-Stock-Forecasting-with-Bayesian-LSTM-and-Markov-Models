import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset with sentiment already merged
df = pd.read_csv("data_with_news_sentiment.csv", index_col=0, parse_dates=True)

# -------------------------
# 1. Correlation Analysis
# -------------------------
print("📌 Correlation Checks")
print("Correlation (sentiment vs returns):", df['sentiment_news'].corr(df['returns']))
print("Correlation (sentiment vs VIX):", df['sentiment_news'].corr(df['VIX']))

plt.figure(figsize=(6,4))
plt.scatter(df['sentiment_news'], df['returns'], alpha=0.3)
plt.xlabel("Sentiment (news)")
plt.ylabel("Returns")
plt.title("Sentiment vs Returns")
plt.show()

# -------------------------
# 2. Granger Causality Test
# -------------------------
print("\n📌 Granger Causality Test (Does sentiment help predict returns?)")
try:
    granger_result = grangercausalitytests(
        df[['returns','sentiment_news']].dropna(), maxlag=5, verbose=False
    )
    for lag in granger_result:
        p_val = granger_result[lag][0]['ssr_ftest'][1]
        print(f"Lag {lag}: p-value = {p_val:.4f}")
except Exception as e:
    print("Granger causality test failed:", e)

# -------------------------
# 3. Feature Importance Test
# -------------------------
print("\n📌 Feature Importance with RandomForest")

X = df.drop(columns=['returns'])
y = df['returns']

train_size = int(0.8*len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# With sentiment
model_with = RandomForestRegressor(random_state=42).fit(X_train, y_train)
pred_with = model_with.predict(X_test)
mse_with = mean_squared_error(y_test, pred_with)

# Without sentiment
X_train_wo = X_train.drop(columns=['sentiment_news'])
X_test_wo = X_test.drop(columns=['sentiment_news'])
model_wo = RandomForestRegressor(random_state=42).fit(X_train_wo, y_train)
pred_wo = model_wo.predict(X_test_wo)
mse_wo = mean_squared_error(y_test, pred_wo)

print("MSE with sentiment:", mse_with)
print("MSE without sentiment:", mse_wo)

# Feature importance
importances = pd.Series(model_with.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10,5))
plt.title("Feature Importances (with sentiment)")
plt.show()

# -------------------------
# 4. Rolling Regression Beta
# -------------------------
print("\n📌 Rolling Regression of Returns on Sentiment")
window = 252  # ~1 year
betas = []

for i in range(window, len(df)):
    y_window = df['returns'].iloc[i-window:i]
    X_window = sm.add_constant(df['sentiment_news'].iloc[i-window:i])
    model = sm.OLS(y_window, X_window).fit()
    betas.append(model.params['sentiment_news'])

plt.plot(df.index[window:], betas)
plt.axhline(0, color='red', linestyle='--')
plt.title("Rolling Beta of Returns on Sentiment")
plt.show()

# -------------------------
# 5. Summary Heuristic
# -------------------------
print("\n📌 Summary:")
if mse_with < mse_wo:
    print("✅ Sentiment seems to add predictive value (lower MSE).")
else:
    print("⚠️ Sentiment might be adding noise (higher MSE).")

avg_corr = abs(df['sentiment_news'].corr(df['returns']))
if avg_corr < 0.05:
    print("⚠️ Very weak correlation with returns (<0.05).")
else:
    print("✅ Some linear relationship with returns.")

print("Check rolling beta plot: if it hovers around 0, sentiment may be noisy.")
