# 🧠 Bayesian LSTM Models

This folder contains the implementation of the regime-aware Bayesian LSTM model for stock price forecasting with uncertainty quantification.

---

## 📁 Files Overview

### **`bayesian_lstm.py`**
The main implementation of the Bayesian LSTM model with the following components:

#### **BayesianLSTM Class**
- **Purpose**: Complete implementation of regime-aware Bayesian LSTM
- **Key Features**:
  - Monte Carlo Dropout for uncertainty estimation
  - Regime-aware feature engineering
  - Multi-source data integration
  - Comprehensive evaluation capabilities

#### **Core Methods**

```python
# Initialize the model
model = BayesianLSTM(
    sequence_length=60,        # Look-back window
    lstm_units=64,            # LSTM hidden units
    dropout_rate=0.3,         # MC Dropout rate
    monte_carlo_samples=100   # Uncertainty samples
)

# Prepare regime-aware features
features = model.prepare_regime_aware_features(df)

# Train with uncertainty quantification
history = model.train(X_train, y_train, X_val, y_val)

# Predict with confidence intervals
predictions = model.predict_with_uncertainty(X_test)
```

### **`bayesian_lstm_model.h5`** *(Generated after training)*
- Trained TensorFlow/Keras model file
- Contains learned weights and architecture
- Can be loaded for inference or further training

---

## 🎯 Architecture Details

### **Network Structure**
```
Input Shape: (sequence_length, n_features)
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
Dropout: 30% (for uncertainty)
    ↓
LSTM Layer 2: 32 units, return_sequences=False
    ↓
Dropout: 30%
    ↓
Dense Layer 1: 32 units, ReLU activation
    ↓
Dropout: 30%
    ↓
Dense Layer 2: 16 units, ReLU activation
    ↓
Dropout: 30%
    ↓
Output Layer: 1 unit (log return prediction)
```

### **Input Features** *(Total: ~20-25 features)*

#### **Market Features** (6)
- Open, High, Low, Close, Volume, VIX

#### **Technical Indicators** (5)
- RSI, MACD_diff, BB_high, BB_low, OBV

#### **Macroeconomic Features** (3)
- CPI, Unemployment, Federal Funds Rate

#### **Lagged Returns** (4)
- return_lag_1, return_lag_2, return_lag_3, return_lag_5

#### **🎯 Regime Features** (3) - **KEY INNOVATION**
- regime_0_prob (Crisis probability)
- regime_1_prob (Normal probability)  
- regime_2_prob (Bull probability)

#### **Sentiment Features** (1) - *Optional*
- sentiment_score (if available)

---

## 🔬 Uncertainty Quantification

### **Monte Carlo Dropout Method**
```python
def predict_with_uncertainty(self, X, training=True):
    predictions = []
    
    # Multiple forward passes with dropout active
    for _ in range(self.monte_carlo_samples):
        y_pred = self.model(X, training=training)
        predictions.append(y_pred.numpy().flatten())
    
    predictions = np.array(predictions)
    
    # Statistical analysis
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # 95% Confidence intervals
    lower_ci = mean_pred - 1.96 * std_pred
    upper_ci = mean_pred + 1.96 * std_pred
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }
```

### **Why This Works**
- **Dropout as Bayesian Approximation**: Each forward pass samples a different network
- **Ensemble Effect**: 100 predictions provide distribution estimate
- **Calibrated Uncertainties**: Higher uncertainty during volatile periods

---

## 📊 Training Configuration

### **Hyperparameters**
```python
SEQUENCE_LENGTH = 60        # 60-day lookback window
LSTM_UNITS = [64, 32]      # Hierarchical LSTM layers  
DROPOUT_RATE = 0.3         # 30% dropout for uncertainty
LEARNING_RATE = 0.001      # Adam optimizer
BATCH_SIZE = 32            # Mini-batch size
MAX_EPOCHS = 100           # With early stopping
```

### **Loss Function & Optimization**
- **Loss**: Huber Loss (robust to outliers)
- **Optimizer**: Adam with learning rate scheduling
- **Callbacks**: Early stopping, learning rate reduction
- **Validation**: Time-series split (no data leakage)

### **Regime-Aware Training**
The model learns to:
1. **Recognize regime patterns** from probability inputs
2. **Adapt prediction behavior** based on market conditions
3. **Increase uncertainty** during regime transitions
4. **Maintain calibration** across different market states

---

## 🎯 Usage Examples

### **Basic Training**
```python
from models.bayesian_lstm import BayesianLSTM

# Load data with regimes
df = pd.read_csv("../data/data_with_regimes.csv")

# Initialize model
model = BayesianLSTM()

# Prepare data
X_train, y_train, X_test, y_test, dates = model.prepare_data(df)

# Train
history = model.train(X_train, y_train, X_test, y_test)

# Evaluate
results = model.evaluate_regime_performance(X_test, y_test, dates, df)
```

### **Custom Configuration**
```python
# High-uncertainty model (more conservative)
conservative_model = BayesianLSTM(
    dropout_rate=0.5,           # Higher dropout
    monte_carlo_samples=200     # More uncertainty samples
)

# Fast inference model (less uncertainty sampling)
fast_model = BayesianLSTM(
    monte_carlo_samples=50      # Fewer samples for speed
)
```

---

## 📈 Performance Characteristics

### **Training Time**
- **CPU**: ~10-15 minutes (depending on data size)
- **GPU**: ~2-3 minutes (recommended for larger datasets)

### **Memory Requirements**
- **Training**: ~2-4 GB RAM
- **Inference**: ~500 MB RAM
- **Model Size**: ~5-10 MB saved model

---

## 🔧 Troubleshooting

### **Common Issues**

#### **Memory Errors**
```python
# Reduce batch size or sequence length
model = BayesianLSTM(sequence_length=30, batch_size=16)
```

#### **Poor Convergence**
```python
# Lower learning rate or increase patience
# Add more regularization
model.compile(optimizer=Adam(learning_rate=0.0005))
```

#### **Overfitting**
```python
# Increase dropout rate
# Reduce model complexity
model = BayesianLSTM(dropout_rate=0.4, lstm_units=32)
```

#### **Miscalibrated Uncertainty**
```python
# Increase Monte Carlo samples
# Check for data leakage in time series split
model = BayesianLSTM(monte_carlo_samples=200)
```

---

## 🚀 Extensions & Modifications

### **Easy Customizations**

#### **Different Architectures**
```python
# Add more LSTM layers
# Experiment with GRU instead of LSTM
# Try attention mechanisms
```

#### **Alternative Uncertainty Methods**
```python
# Implement Variational Inference
# Try Deep Ensembles
# Use Concrete Dropout
```

#### **Enhanced Features**
```python
# Add news sentiment scores
# Include options data (VIX term structure)
# Incorporate cross-asset correlations
```

---

## 📚 References

- **Bayesian Deep Learning**: Gal, Y., & Ghahramani, Z. (2016)
- **Monte Carlo Dropout**: Gal, Y. (2016) - Uncertainty in Deep Learning
- **Financial Time Series**: Tsay, R. S. (2010) - Analysis of Financial Time Series
- **Regime Switching**: Hamilton, J. D. (1989) - Markov Switching Models

---

## 🤝 Contributing

To extend or modify the Bayesian LSTM implementation:

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement your changes** in `bayesian_lstm.py`
4. **Add tests** and documentation
5. **Submit a pull request**

Key areas for contribution:
- Alternative uncertainty quantification methods
- Enhanced architectures (attention, transformers)
- Multi-asset forecasting capabilities
- Real-time inference optimizations
