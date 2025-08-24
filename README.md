# 📊 Regime-Aware Probabilistic Forecasting of Stock Prices using Bayesian LSTM and Markov Switching Models

This project implements a sophisticated financial forecasting system that combines **Markov Switching Models (MSM)** for detecting market regimes with **Bayesian LSTM** to provide probabilistic forecasts with uncertainty quantification. The model adapts its predictions based on detected market conditions (Crisis, Normal, Bull) and provides confidence intervals for all forecasts.

---

## 📌 Problem Statement

Traditional stock price forecasting methods suffer from several limitations:
- **Lack of regime awareness** - Unable to adapt to changing market conditions
- **No uncertainty quantification** - Provide point estimates without confidence measures
- **Poor performance during market transitions** - Fail to capture regime shifts

This project addresses these issues by:
- Detecting different **market regimes** (Crisis/Normal/Bull) using custom MSM
- Forecasting future prices while **quantifying prediction uncertainty**
- Leveraging **Bayesian LSTM** with Monte Carlo Dropout for confidence intervals
- Creating **regime-aware predictions** that adapt to market conditions

---

## 🧠 Core Techniques & Innovations

### **Regime Detection**
- **Custom Multivariate MSM** - From-scratch implementation using Baum-Welch EM algorithm
- **Multivariate observations** - Uses log returns + VIX for robust regime identification
- **Three-regime classification** - Crisis, Normal, and Bull market periods

### **Bayesian Neural Networks**
- **Monte Carlo Dropout** - 100 forward passes for uncertainty estimation
- **Calibrated confidence intervals** - 95% prediction intervals with coverage assessment
- **Regime-aware architecture** - Uses regime probabilities as input features

### **Multi-Source Feature Engineering**
- **Technical indicators** - RSI, MACD, Bollinger Bands, OBV
- **Macroeconomic factors** - CPI, Unemployment, Federal Funds Rate
- **Market sentiment** - AAII investor sentiment data
- **Regime information** - Soft and hard regime classifications

---

## 📁 Project Structure

```
Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/
├── 📊 data/                    # Data processing pipeline
│   ├── dataset_creation.py     # Download & enrich market data
│   ├── eda.py                  # Data cleaning & exploration
│   ├── sentiment.py            # Sentiment data integration
│   ├── feature_regime_preprocessing.py  # Feature engineering & MSM
│   ├── data.csv               # Raw market data
│   ├── data_cleaned.csv       # Cleaned dataset
│   ├── data_with_sentiment.csv # With sentiment features
│   ├── data_with_regimes.csv  # Final dataset with regimes
│   └── README.md              # Data pipeline documentation
│
├── 🧠 models/                  # Bayesian LSTM implementation
│   ├── bayesian_lstm.py       # Main Bayesian LSTM model
│   ├── bayesian_lstm_model.h5 # Trained model (generated)
│   └── README.md              # Model documentation
│
├── 📈 evaluation/              # Comprehensive evaluation toolkit
│   ├── model_evaluation.py    # Evaluation & visualization
│   └── README.md              # Evaluation documentation
│
├── 📊 msm/                     # Custom MSM implementation
│   ├── msm.py                 # Multivariate Markov Switching Model
│   └── README.md              # MSM algorithm documentation
│
├── 📁 results/                 # Output files (generated)
│   └── bayesian_lstm_predictions.csv
│
├── 🚀 Pipeline Scripts
│   ├── setup_and_run.py       # Interactive component runner
│   ├── run_complete_pipeline.py # Automated full pipeline
│   └── requirements.txt       # Dependencies
│
└── 📖 README.md               # This file
```

---

## 🚀 Quick Start Guide

### **Prerequisites**
- Python 3.8+
- GPU recommended for faster training (optional)

### **Installation**
```bash
# Clone the repository
git clone https://github.com/dhars1n1/Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models.git
cd Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models

# Install dependencies
pip install -r requirements.txt

# Set up FRED API key (for macro data)
# Create .env file and add: FRED_API_KEY=your_api_key_here
```

### **Option 1: Interactive Setup & Execution (Recommended)**
```bash
python setup_and_run.py
```
This provides an interactive menu to:
- Check setup and dependencies
- Run individual components
- Execute the complete pipeline

### **Option 2: Automated Complete Pipeline**
```bash
python run_complete_pipeline.py
```

### **Option 3: Step-by-Step Manual Execution**
```bash
# Step 1: Data preparation (if needed)
python data/dataset_creation.py  # Download market data
python data/eda.py              # Clean and explore
python data/sentiment.py        # Add sentiment data

# Step 2: Regime detection & feature engineering
python data/feature_regime_preprocessing.py

# Step 3: Train Bayesian LSTM
python models/bayesian_lstm.py

# Step 4: Comprehensive evaluation
python evaluation/model_evaluation.py
```

---

## 📊 Implementation Highlights

### **🎯 Regime-Aware Architecture**
```python
# Regime probabilities become input features
regime_features = [
    'regime_0_prob',  # Crisis probability
    'regime_1_prob',  # Normal probability  
    'regime_2_prob'   # Bull probability
]

# Model adapts predictions based on current regime
model_input = market_features + technical_features + regime_features
```

### **🔬 Uncertainty Quantification**
```python
# Monte Carlo Dropout for uncertainty estimation
predictions = []
for _ in range(100):  # 100 forward passes
    y_pred = model(X, training=True)  # Keep dropout active
    predictions.append(y_pred)

# Calculate confidence intervals
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
confidence_interval = mean_pred ± 1.96 * std_pred
```

### **📈 Expected Outputs**

After running the pipeline, you'll receive:

1. **📋 Evaluation Report**
   - Overall performance metrics (MSE, MAE, RMSE, R2)
   - Uncertainty calibration analysis 
   - Regime-specific performance breakdown
   - Model insights and recommendations

2. **📊 Comprehensive Visualizations**
   - Time series predictions with uncertainty bands
   - Regime-colored actual vs predicted scatter plots
   - Uncertainty calibration analysis
   - Performance breakdown by market regime

3. **📁 Generated Files**
   - `models/bayesian_lstm_model.h5` - Trained model
   - `results/bayesian_lstm_predictions.csv` - Predictions with uncertainty
   - `data/data_with_regimes.csv` - Enhanced dataset

---

## 🎯 Key Features & Innovations

### **✅ Regime Awareness**
- **Automatic regime detection** using custom multivariate MSM
- **Adaptive predictions** that change behavior based on market conditions
- **Regime transition modeling** for better forecasting accuracy

### **✅ Uncertainty Quantification**
- **Calibrated confidence intervals** with coverage assessment
- **Monte Carlo Dropout** for robust uncertainty estimation
- **Prediction reliability scoring** for risk management

### **✅ Comprehensive Evaluation**
- **Multi-metric assessment** beyond simple accuracy
- **Regime-specific performance** analysis
- **Uncertainty calibration** validation
- **Interactive visualization** dashboard

### **✅ Production-Ready Design**
- **Modular architecture** for easy maintenance and extension
- **Robust error handling** with detailed logging
- **Flexible execution options** (interactive, automated, manual)
- **Comprehensive documentation** and examples

---

## 📚 Research & Technical Details

### **Mathematical Foundation**

**Markov Switching Model:**
```
P(s_t = j | s_{t-1} = i) = A_{ij}
y_t | s_t = k ~ N(μ_k, Σ_k)
```

**Bayesian LSTM with MC Dropout:**
```
p(y|x) ≈ 1/T ∑_{t=1}^T f(x, θ_t)
where θ_t ~ q(θ) (dropout distribution)
```

### **Model Architecture**
- **Input Layer**: Multi-dimensional features (market + regime + sentiment)
- **LSTM Layers**: 64 → 32 units with dropout (0.3)
- **Dense Layers**: 32 → 16 → 1 with dropout for uncertainty
- **Output**: Log return predictions with uncertainty bounds

---

## 🔬 Implementation Notes

### **Architecture Design**
- **Input Layer**: Multi-dimensional features (market + regime + sentiment)
- **LSTM Layers**: 64 → 32 units with dropout (0.3)
- **Dense Layers**: 32 → 16 → 1 with dropout for uncertainty
- **Output**: Log return predictions with uncertainty bounds

### **Training Strategy**
- **Loss Function**: Huber loss (robust to outliers)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Early stopping + dropout + L2
- **Validation**: Time-series split (80/20)

### **Key Findings**
1. **Regime awareness significantly improves** forecasting during market transitions
2. **Uncertainty estimates are well-calibrated** and useful for risk management
3. **Crisis periods show higher uncertainty** as expected
4. **Model performs best in Normal markets**, struggles most in Crisis periods

---

## 🚧 Future Enhancements

- [ ] **Multi-asset forecasting** with cross-asset regime dependencies
- [ ] **Real-time regime detection** for live trading applications
- [ ] **Alternative uncertainty methods** (Variational Inference, Ensemble methods)
- [ ] **Extended feature engineering** (News sentiment, Options data)
- [ ] **Portfolio optimization** integration with uncertainty-aware allocation

---

## 📖 References & Acknowledgments

This project builds upon research in:
- Hidden Markov Models for financial time series (Hamilton, 1989)
- Bayesian Deep Learning (Gal & Ghahramani, 2016)
- Regime-switching models in finance (Ang & Bekaert, 2002)

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📞 Contact

**Author**: dhars1n1  
**Repository**: [Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models](https://github.com/dhars1n1/Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.


