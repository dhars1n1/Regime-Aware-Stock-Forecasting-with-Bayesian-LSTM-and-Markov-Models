# Regime-Aware Bayesian LSTM for Stock Forecasting

This module implements a sophisticated stock price forecasting system that combines Bayesian Deep Learning for uncertainty quantification, LSTM networks for sequential pattern recognition, and market regime awareness for adaptive predictions.

## Overview

The system combines:
- **Bayesian Deep Learning** for uncertainty quantification
- **LSTM networks** for sequential pattern recognition
- **Market regime awareness** for adaptive predictions

## Detailed Workflow Explanation

### 1. Class Initialization and Configuration

The `BayesianLSTM` class is initialized with carefully chosen parameters:

```python
class BayesianLSTM:
    def __init__(self, sequence_length: int = 20, lstm_units: int = 64, 
                 dropout_rate: float = 0.3, n_features: Optional[int] = None, 
                 monte_carlo_samples: int = 100, use_regime_label: bool = True):
```

**Parameter Reasoning:**
- `sequence_length=20`: Uses 20 trading days of history to predict the next day (approximately 1 month of trading data)
- `dropout_rate=0.3`: High dropout for regularization - crucial for Bayesian inference as it enables Monte Carlo sampling during prediction
- `monte_carlo_samples=100`: Number of forward passes with different dropout masks to estimate uncertainty
- `use_regime_label=True`: Incorporates explicit regime information as a feature

### 2. Feature Engineering Pipeline

The system creates a comprehensive feature set across multiple categories:

#### Market Features
- `log_return`: Log returns are preferred over simple returns because they're additive and normalize the scale
- `Volume`: Trading volume indicates market activity and liquidity
- `VIX`: Volatility index captures market fear/uncertainty

#### Technical Indicators
- `RSI`: Relative Strength Index (momentum oscillator)
- `MACD_diff`: Moving Average Convergence Divergence difference
- `BB_high/BB_low`: Bollinger Bands capture volatility and mean reversion

#### Regime Features
The system intelligently handles regime information:

```python
if self.use_regime_label and 'regime_label' in df.columns:
    regime_features = ['regime_label']
else:
    regime_features = []
    for i in range(3):
        prob_col = f'regime_{i}_prob'
        if prob_col in df.columns:
            regime_features.append(prob_col)
```

**Why regime awareness is crucial:**
- Markets behave differently in bull, bear, and crisis periods
- Same technical indicators have different predictive power across regimes
- Model can adapt its predictions based on current market state

#### Label Encoding for Regimes
Neural networks need numerical inputs, so categorical regime labels are encoded as integers using `LabelEncoder`.

### 3. Data Preparation and Safeguards

The system implements critical safeguards:

```python
# Safeguard: Handle duplicate indices
if not feature_data.index.is_unique:
    dup_count = feature_data.index.duplicated().sum()
    print(f"Warning: {dup_count} duplicate indices found in features. Dropping duplicates...")
    feature_data = feature_data.loc[~feature_data.index.duplicated(keep="first")]
```

**Critical safeguards implemented:**
1. **Duplicate index handling**: Prevents pandas reindexing errors
2. **Length validation**: Ensures feature and target alignment
3. **Temporal splitting**: Maintains chronological order (no data leakage)

#### Temporal Split Strategy
```python
split_idx = int(len(feature_data) * (1 - test_size))
train_idx = feature_data.index[:split_idx]
test_idx = feature_data.index[split_idx:]
```

**Why temporal split?** Financial data has time dependencies - using random splits would create data leakage where future information influences past predictions.

### 4. Feature Scaling Strategy

```python
self.scalers['features'] = StandardScaler()
train_features_scaled = self.scalers['features'].fit_transform(train_features)

self.scalers['target'] = StandardScaler()
train_target_scaled = self.scalers['target'].fit_transform(train_target_array).flatten()
```

**Why StandardScaler?**
- Neural networks perform better with normalized inputs (mean=0, std=1)
- Different features have vastly different scales (VIX ~20, Volume ~millions)
- Separate scalers for features and targets allow proper inverse transformation

### 5. Sequence Creation for LSTM

The system creates sequences for LSTM processing:

```python
def create_sequences(self, feature_data: pd.DataFrame, target_data: pd.Series):
    for i in range(self.sequence_length, len(feature_data)):
        sequence = feature_data.iloc[i-self.sequence_length:i].values
        X.append(sequence)
        y.append(target_data.iloc[i])
```

**Sequence Logic:**
- Input: 20 timesteps of features (sliding window)
- Output: Next day's return
- Shape: `(n_samples, sequence_length, n_features)`

**Example:** To predict return on day 21, use features from days 1-20.

### 6. Bayesian LSTM Architecture

The neural network architecture is carefully designed:

```python
def build_model(self) -> keras.Model:
    model = keras.Sequential([
        layers.LSTM(self.lstm_units, return_sequences=True, ...),
        layers.Dropout(self.dropout_rate),
        layers.LSTM(self.lstm_units // 2, return_sequences=False, ...),
        layers.Dropout(self.dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dropout(self.dropout_rate),
        layers.Dense(1)
    ])
```

**Architecture Decisions:**

#### Two LSTM layers with decreasing size:
- First LSTM (64 units): Captures complex temporal patterns
- Second LSTM (32 units): Refines and summarizes information
- `return_sequences=True/False`: First layer passes sequences, second outputs single vector

#### Multiple Dropout Layers:
- **Standard use**: Prevents overfitting during training
- **Bayesian use**: Enables Monte Carlo sampling during inference

#### Dense Layers:
- 32 → 16 → 1 units: Gradual compression to final prediction
- ReLU activation: Introduces non-linearity

#### Huber Loss:
```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='huber')
```
**Why Huber?** More robust to outliers than MSE, important for financial data with extreme events.

### 7. Bayesian Inference via Monte Carlo Dropout

The core of uncertainty quantification:

```python
def predict_with_uncertainty(self, X: np.ndarray, return_raw: bool = False):
    predictions = []
    for i in range(self.monte_carlo_samples):
        y_pred = self.model(X, training=True)  # Key: training=True
        predictions.append(y_pred.numpy().flatten())
```

**Critical Insight: `training=True`**
- During inference, we set `training=True` to keep dropout active
- Each forward pass produces different predictions due to random dropout
- Collection of predictions forms a distribution for uncertainty estimation

#### Uncertainty Metrics Calculated:
```python
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
lower_ci_95 = np.percentile(predictions, 2.5, axis=0)
upper_ci_95 = np.percentile(predictions, 97.5, axis=0)
```

- **Mean**: Point prediction (expected value)
- **Standard deviation**: Prediction uncertainty
- **Confidence intervals**: Range of plausible values

### 8. Regime-Aware Performance Evaluation

The system evaluates performance across different market regimes:

```python
def evaluate_regime_performance(self, X_test, y_test, test_dates, df):
    # Align regimes with test samples
    test_regimes = df_unique['regime_label'].reindex(test_dates).values
    
    # Overall metrics
    overall_results = {
        'mse': mean_squared_error(y_test_orig, y_pred_orig),
        'coverage_95': np.mean((y_test_orig >= lower_95_orig) & (y_test_orig <= upper_95_orig))
    }
```

**Key Evaluation Aspects:**

#### Coverage Metrics:
- **95% Coverage**: Percentage of actual values within 95% confidence intervals
- **Interval Width**: Average width of confidence intervals
- **Good model**: High coverage (~95%) with narrow intervals

#### Regime-Specific Analysis:
```python
for regime in unique_regimes:
    mask = test_regimes == regime
    if np.sum(mask) > 5:  # Skip regimes with few samples
        regime_results[regime] = {...}
```

**Why regime-specific evaluation?**
- Different regimes have different prediction difficulty
- Crisis periods typically have higher uncertainty
- Model should adapt prediction confidence to regime

### 9. Comprehensive Visualization System

The system creates multiple visualizations:

```python
def create_comprehensive_visualizations(self, test_dates, evaluation_results, save_path="results"):
    # Plot 1: Predictions with uncertainty bands
    ax1.fill_between(test_dates, lower_95, upper_95, alpha=0.3, color='blue')
    
    # Plot 2: Regime-colored actual returns
    for regime in np.unique(test_regimes):
        mask = test_regimes == regime
        ax2.scatter(test_dates[mask], y_actual[mask], c=regime_colors.get(regime, 'gray'))
```

**Visualization Components:**
1. **Time series with uncertainty**: Shows prediction accuracy and confidence over time
2. **Regime visualization**: Reveals how market conditions affect returns
3. **Scatter plots**: Prediction vs actual correlation by regime
4. **Uncertainty analysis**: Distribution of model confidence across regimes

### 10. Inference Artifacts Management

For production deployment, the system saves all necessary artifacts:

```python
def save_all_inference_artifacts(bayesian_lstm_model, save_path: str = "results"):
    # 1. Save scalers
    with open(scalers_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.scalers, f)
    
    # 2. Save regime encoder
    with open(encoder_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.regime_encoder, f)
    
    # 3. Save metadata
    metadata = {
        'feature_columns': bayesian_lstm_model.feature_columns,
        'n_features': bayesian_lstm_model.n_features,
        ...
    }
```

**Why save these artifacts?**
- **Scalers**: New data must be scaled using same parameters as training data
- **Regime encoder**: Consistent encoding of regime labels
- **Metadata**: Ensures feature alignment and model configuration consistency

### 11. Main Training Pipeline

The complete training workflow:

```python
def main():
    # 1. Load regime-labeled data
    df = pd.read_csv("../data/data_with_regimes.csv", parse_dates=["Date"])
    
    # 2. Initialize model
    model = BayesianLSTM(sequence_length=20, lstm_units=64, ...)
    
    # 3. Prepare sequences
    X_train, y_train, X_test, y_test, test_dates = model.prepare_data(df)
    
    # 4. Train with callbacks
    history = model.train(X_train, y_train, X_test, y_test, epochs=100)
```

**Training Callbacks:**
```python
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)
```

- **Early stopping**: Prevents overfitting by monitoring validation loss
- **Learning rate reduction**: Adapts learning rate when loss plateaus

## Comprehensive Evaluation Metrics

The system employs a rich set of evaluation metrics specifically chosen for financial forecasting and uncertainty quantification. These metrics assess both prediction accuracy and the quality of uncertainty estimates.

### Traditional Prediction Accuracy Metrics

#### 1. Mean Squared Error (MSE)
**Formula:** 
```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

**Implementation:**
```python
mse = mean_squared_error(y_test_orig, y_pred_orig)
```

**Why chosen:**
- Penalizes large errors more heavily than small ones
- Standard metric for regression problems
- Differentiable (useful for gradient-based optimization)
- **Financial relevance**: Large prediction errors in financial markets can be catastrophic

#### 2. Mean Absolute Error (MAE)
**Formula:**
```
MAE = (1/n) × Σ|y_actual - y_predicted|
```

**Implementation:**
```python
mae = mean_absolute_error(y_test_orig, y_pred_orig)
```

**Why chosen:**
- More robust to outliers than MSE
- Easier to interpret (same units as the target variable)
- **Financial relevance**: Provides average magnitude of prediction errors in return terms

#### 3. Root Mean Squared Error (RMSE)
**Formula:**
```
RMSE = √(MSE) = √[(1/n) × Σ(y_actual - y_predicted)²]
```

**Implementation:**
```python
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
```

**Why chosen:**
- Same units as the target variable (unlike MSE)
- Maintains the penalty for large errors from MSE
- **Financial relevance**: Directly interpretable as prediction error in return percentage

#### 4. Coefficient of Determination (R²)
**Formula:**
```
R² = 1 - (SS_res / SS_tot)
where:
SS_res = Σ(y_actual - y_predicted)²  (residual sum of squares)
SS_tot = Σ(y_actual - ȳ_actual)²     (total sum of squares)
```

**Implementation:**
```python
r2 = 1 - np.sum((y_test_orig - y_pred_orig)**2) / np.sum((y_test_orig - np.mean(y_test_orig))**2)
```

**Why chosen:**
- Scale-independent measure of prediction quality
- Values range from -∞ to 1, where 1 indicates perfect prediction
- **Financial relevance**: Shows how much of the return variance the model explains

### Bayesian/Uncertainty-Specific Metrics

#### 5. Coverage Probability (95% and 68%)
**Formula:**
```
Coverage_95% = (1/n) × Σ[I(y_actual_i ∈ [CI_lower_95_i, CI_upper_95_i])]
where I(·) is the indicator function
```

**Implementation:**
```python
coverage_95 = np.mean((y_test_orig >= lower_95_orig) & (y_test_orig <= upper_95_orig))
coverage_68 = np.mean((y_test_orig >= lower_68_orig) & (y_test_orig <= upper_68_orig))
```

**Why chosen:**
- **Critical for Bayesian models**: Validates that confidence intervals are well-calibrated
- Good coverage ≈ 95% for 95% intervals, ≈ 68% for 68% intervals
- **Financial relevance**: Essential for risk management - tells you if your uncertainty estimates are trustworthy

#### 6. Average Interval Width
**Formula:**
```
Avg_Width_95% = (1/n) × Σ(CI_upper_95_i - CI_lower_95_i)
```

**Implementation:**
```python
interval_width_95 = np.mean(upper_95_orig - lower_95_orig)
```

**Why chosen:**
- Measures the precision of uncertainty estimates
- **Trade-off with coverage**: Wider intervals → higher coverage but less useful predictions
- **Financial relevance**: Narrow intervals with good coverage indicate confident, reliable predictions

#### 7. Prediction Standard Deviation
**Formula:**
```
σ_pred_i = std([pred_1_i, pred_2_i, ..., pred_M_i])
where M is the number of Monte Carlo samples
```

**Implementation:**
```python
std_pred = np.std(predictions, axis=0)  # predictions shape: (monte_carlo_samples, n_test_samples)
```

**Why chosen:**
- Direct measure of prediction uncertainty
- **Financial interpretation**: Higher σ indicates the model is less confident about that prediction
- Used for dynamic position sizing and risk management

### Advanced Statistical Metrics

#### 8. Interquartile Range (IQR) of Predictions
**Formula:**
```
IQR = Q3 - Q1 = P75 - P25
where P75 is 75th percentile, P25 is 25th percentile
```

**Implementation:**
```python
iqr = np.percentile(predictions, 75, axis=0) - np.percentile(predictions, 25, axis=0)
```

**Why chosen:**
- Robust measure of uncertainty (less sensitive to outliers than standard deviation)
- **Financial relevance**: Provides middle 50% range of model predictions

#### 9. Median Prediction
**Formula:**
```
Median_pred = P50(predictions)
```

**Implementation:**
```python
median_pred = np.median(predictions, axis=0)
```

**Why chosen:**
- More robust central tendency than mean when prediction distribution is skewed
- **Financial relevance**: In volatile markets, median can be more stable than mean

### Regime-Specific Metrics

All above metrics are calculated separately for each market regime:

```python
for regime in unique_regimes:
    mask = test_regimes == regime
    if np.sum(mask) > 5:  # Skip regimes with few samples
        regime_results[regime] = {
            'mse': mean_squared_error(regime_actual, regime_pred),
            'mae': mean_absolute_error(regime_actual, regime_pred),
            'coverage_95': regime_coverage,
            'interval_width_95': regime_width,
            'count': np.sum(mask),
            'std_actual': np.std(regime_actual),
            'std_pred': np.std(regime_pred)
        }
```

**Why regime-specific evaluation is crucial:**
- **Bull markets**: Typically easier to predict, expect lower RMSE, narrower intervals
- **Crisis periods**: Higher volatility, expect higher RMSE, wider intervals but still good coverage
- **Model adaptation**: Good model should show different uncertainty levels across regimes

### Derived Metrics for Analysis

#### 10. Absolute Error
**Formula:**
```
AE_i = |y_actual_i - y_predicted_i|
```

**Implementation:**
```python
results_df['Absolute_Error'] = np.abs(results_df['Actual_Return'] - results_df['Predicted_Return'])
```

**Used for**: Error distribution analysis, regime-specific error comparison

#### 11. Squared Error
**Formula:**
```
SE_i = (y_actual_i - y_predicted_i)²
```

**Implementation:**
```python
results_df['Squared_Error'] = (results_df['Actual_Return'] - results_df['Predicted_Return'])**2
```

**Used for**: Identifying periods of high prediction error

#### 12. Coverage Indicator
**Formula:**
```
Coverage_i = I(y_actual_i ∈ [CI_lower_95_i, CI_upper_95_i])
```

**Implementation:**
```python
results_df['In_CI_95'] = ((results_df['Actual_Return'] >= results_df['Lower_95_CI']) & 
                         (results_df['Actual_Return'] <= results_df['Upper_95_CI']))
```

**Used for**: Time-series analysis of coverage quality

### Metric Interpretation Guidelines

#### For Production Systems:
- **MSE/RMSE**: Should be minimized, but not at the expense of coverage
- **Coverage_95**: Should be close to 0.95 (±0.05 acceptable)
- **Interval_Width**: Should be as narrow as possible while maintaining good coverage
- **R²**: Higher is better, but >0.1 is reasonable for daily financial returns

#### Red Flags:
- **Coverage << 0.95**: Model is overconfident (intervals too narrow)
- **Coverage >> 0.95**: Model is underconfident (intervals too wide)
- **Very different regime performance**: May indicate regime detection issues
- **High MSE with very narrow intervals**: Model is both inaccurate and overconfident

### Why These Metrics Matter for Financial Applications

1. **Risk Management**: Coverage and interval width directly inform position sizing
2. **Model Confidence**: Standard deviation helps identify when the model is uncertain
3. **Regime Adaptation**: Regime-specific metrics ensure the model adapts to market conditions
4. **Backtesting**: Comprehensive metrics prevent overfitting to any single measure
5. **Regulatory Compliance**: Many financial models require uncertainty quantification

## Why This Approach Works

### 1. Uncertainty Quantification
Traditional models give point predictions without confidence measures. This Bayesian approach provides:
- Confidence intervals for risk management
- Adaptive uncertainty based on market conditions
- Better decision-making support

### 2. Regime Awareness
Markets have different behaviors in different periods:
- **Bull markets**: Momentum strategies work
- **Bear markets**: Mean reversion dominates  
- **Crisis periods**: High volatility, different correlations

### 3. Robust Architecture
- **LSTM**: Captures long-term dependencies in financial time series
- **Dropout regularization**: Prevents overfitting to specific market conditions
- **Huber loss**: Robust to outliers (market crashes, gaps)

### 4. Production Ready
- Proper train/test splitting (no data leakage)
- Artifact saving for deployment
- Comprehensive evaluation metrics
- Error handling and data validation

## Files in this Module

- `bayesian_lstm.py`: Main implementation of the Bayesian LSTM model
- `lstm.py`: Standard LSTM implementation for comparison
- `compare_models.py`: Script to compare different model performances
- `results/`: Directory containing trained models, predictions, and visualizations
- `comparison_results/`: Directory containing model comparison results

## Usage

```python
# Basic usage
from bayesian_lstm import BayesianLSTM
import pandas as pd

# Load data
df = pd.read_csv("../data/data_with_regimes.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Initialize model
model = BayesianLSTM(
    sequence_length=20,
    lstm_units=64,
    dropout_rate=0.3,
    monte_carlo_samples=100,
    use_regime_label=True
)

# Prepare data and train
X_train, y_train, X_test, y_test, test_dates = model.prepare_data(df)
model.build_model()
history = model.train(X_train, y_train, X_test, y_test)

# Evaluate with uncertainty
evaluation_results = model.evaluate_regime_performance(X_test, y_test, test_dates, df)
```

This implementation represents a state-of-the-art approach to financial forecasting that balances predictive accuracy with uncertainty quantification, making it suitable for real-world trading applications where risk management is crucial.