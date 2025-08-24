# 📈 Model Evaluation & Analysis

This folder contains comprehensive evaluation tools for the regime-aware Bayesian LSTM model, including performance metrics, uncertainty calibration analysis, and visualization dashboards.

---

## 📁 Files Overview

### **`model_evaluation.py`**
Complete evaluation framework with the following capabilities:

#### **ModelEvaluator Class**
```python
evaluator = ModelEvaluator("../results/bayesian_lstm_predictions.csv")
evaluator.generate_report()          # Comprehensive text report
evaluator.plot_comprehensive_analysis()  # 9-panel visualization
```

---

## 📊 Evaluation Components

### **1. Performance Metrics**
#### **Standard Regression Metrics**
- **MSE (Mean Squared Error)**: Overall prediction accuracy
- **MAE (Mean Absolute Error)**: Robust to outliers
- **RMSE (Root Mean Squared Error)**: Same units as target
- **R² (Coefficient of Determination)**: Variance explained
- **Correlation**: Linear relationship strength

#### **Financial-Specific Metrics**
- **Mean Bias**: Systematic prediction error
- **Standard Deviation of Bias**: Prediction consistency
- **Direction Accuracy**: Correct sign prediction rate

### **2. Uncertainty Calibration Analysis**
#### **Coverage Probability**
```python
# Should be ~0.95 for well-calibrated 95% CI
actual = predictions['Actual']
lower_ci = predictions['Lower_CI'] 
upper_ci = predictions['Upper_CI']

coverage = np.mean((actual >= lower_ci) & (actual <= upper_ci))
```

#### **Interval Width Analysis**
- **Average Width**: Confidence interval size
- **Width Variability**: Uncertainty adaptation
- **Sharp vs Calibrated Trade-off**: Narrow but accurate intervals

#### **Uncertainty-Error Correlation**
```python
uncertainty = predictions['Uncertainty']
errors = np.abs(actual - predicted)
correlation = np.corrcoef(uncertainty, errors)[0, 1]

# Good uncertainty: correlation > 0.3
# Perfect uncertainty: correlation = 1.0
```

### **3. Regime-Specific Performance**
#### **Breakdown by Market Regime**
```python
for regime in ['Crisis', 'Normal', 'Bull']:
    regime_data = df[df['Regime'] == regime]
    metrics = calculate_metrics(regime_data)
    
    # Typical expectations:
    # Crisis: Higher MSE, Higher Uncertainty
    # Normal: Best Performance, Moderate Uncertainty  
    # Bull: Good Performance, Lower Uncertainty
```

#### **Regime Transition Analysis**
- **Performance during transitions**: Model adaptation capability
- **Uncertainty spikes**: Detection of regime changes
- **Recovery time**: How quickly model adapts to new regime

---

## 📋 Evaluation Report Format

The evaluation framework generates detailed performance reports in the following structured format:

```
════════════════════════════════════════════════════════════
REGIME-AWARE BAYESIAN LSTM EVALUATION REPORT
════════════════════════════════════════════════════════════

📊 OVERALL PERFORMANCE METRICS:
----------------------------------------
MSE, MAE, RMSE, R2, Correlation
Mean_Bias, Std_Bias

🎯 UNCERTAINTY CALIBRATION:
----------------------------------------
coverage_probability, average_interval_width
uncertainty_error_correlation

📝 Calibration Assessment based on coverage

🏛️ REGIME-SPECIFIC PERFORMANCE:
----------------------------------------
Metrics broken down by Crisis/Normal/Bull regimes
Including MSE, MAE, R2, Count, Avg_Uncertainty

💡 KEY INSIGHTS section with performance summary
```

*Note: Actual metric values generated during evaluation*

---

## 📊 Visualization Dashboard

### **9-Panel Comprehensive Analysis**

#### **Panel 1: Time Series with Uncertainty**
- **Plot**: Actual vs Predicted with 95% confidence bands
- **Purpose**: Overall model performance over time
- **Key Insights**: Uncertainty adaptation to market volatility

#### **Panel 2: Actual vs Predicted Scatter**
- **Plot**: Perfect prediction line vs actual scatter
- **Purpose**: Overall accuracy assessment
- **Key Insights**: Systematic biases, outlier patterns

#### **Panel 3: Residuals Analysis**
- **Plot**: Predicted vs Residuals scatter
- **Purpose**: Detect heteroscedasticity, non-linear patterns
- **Key Insights**: Model assumptions validation

#### **Panel 4: Uncertainty vs Error**
- **Plot**: Predicted uncertainty vs absolute error
- **Purpose**: Uncertainty calibration quality
- **Key Insights**: Higher uncertainty should correlate with higher errors

#### **Panel 5: Performance by Regime**
- **Plot**: MSE bar chart by regime
- **Purpose**: Regime-specific model performance
- **Key Insights**: Which market conditions are most challenging

#### **Panel 6: Coverage Probability by Regime**
- **Plot**: Coverage rates vs expected 95%
- **Purpose**: Uncertainty calibration by regime
- **Key Insights**: Consistent calibration across market conditions

#### **Panel 7: Residuals Distribution**
- **Plot**: Histogram with normal overlay
- **Purpose**: Check normality assumption
- **Key Insights**: Fat tails, skewness in prediction errors

#### **Panel 8: Returns by Regime**
- **Plot**: Colored scatter by regime
- **Purpose**: Regime identification quality
- **Key Insights**: Clear regime separation, transition periods

#### **Panel 9: Uncertainty Over Time**
- **Plot**: Time series of model uncertainty
- **Purpose**: Uncertainty evolution patterns
- **Key Insights**: Uncertainty spikes during volatile periods

---

## 🎯 Interpretation Guidelines

### **Good Performance Indicators**
- **Overall RMSE < 0.015** (for log returns)
- **Coverage probability: 0.93-0.97** (well-calibrated)
- **Uncertainty-error correlation > 0.3** (good uncertainty)
- **R² > 0.4** (reasonable explanatory power)

### **Regime-Specific Expectations**
```python
# Typical performance hierarchy:
Normal > Bull > Crisis  # (MSE ranking)

# Uncertainty hierarchy:
Crisis > Bull > Normal  # (Uncertainty ranking)

# Coverage should be consistent across regimes (~0.95)
```

### **Warning Signs**
- **Coverage < 0.90 or > 0.98**: Miscalibrated uncertainty
- **Very low uncertainty-error correlation**: Poor uncertainty estimation
- **High systematic bias**: Model drift issues
- **Poor Crisis regime performance**: Inadequate regime awareness

---

## 🔧 Customizing Evaluation

### **Adding New Metrics**
```python
def custom_metric(actual, predicted):
    """Add your custom evaluation metric"""
    return your_calculation

# Integrate into evaluation
evaluator.custom_metrics = {'your_metric': custom_metric}
```

### **Regime-Specific Analysis**
```python
def analyze_regime_transitions(df):
    """Analyze performance during regime changes"""
    transitions = detect_regime_changes(df)
    return transition_performance

# Add to evaluation pipeline
evaluator.add_analysis(analyze_regime_transitions)
```

### **Financial Risk Metrics**
```python
def calculate_var(predictions, confidence=0.05):
    """Value at Risk calculation"""
    return np.percentile(predictions, confidence * 100)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Sharpe ratio for return predictions"""
    return (np.mean(returns) - risk_free_rate) / np.std(returns)
```

---

## 📈 Advanced Analysis Features

### **Backtesting Framework**
```python
def rolling_evaluation(df, window_size=252):
    """Rolling window evaluation for temporal stability"""
    results = []
    for i in range(window_size, len(df)):
        window_data = df.iloc[i-window_size:i]
        metrics = calculate_metrics(window_data)
        results.append(metrics)
    return results
```

### **Regime Transition Detection**
```python
def detect_regime_changes(df, threshold=0.7):
    """Identify when regimes change"""
    max_prob = df[['regime_0_prob', 'regime_1_prob', 'regime_2_prob']].max(axis=1)
    transitions = max_prob < threshold
    return transitions
```

### **Uncertainty Decomposition**
```python
def decompose_uncertainty(predictions):
    """Separate aleatory vs epistemic uncertainty"""
    # Aleatory: irreducible uncertainty (data noise)
    # Epistemic: model uncertainty (reducible with more data)
    return aleatory_component, epistemic_component
```

---

## 🚀 Usage Examples

### **Basic Evaluation**
```python
from evaluation.model_evaluation import ModelEvaluator

# Load predictions
evaluator = ModelEvaluator("../results/bayesian_lstm_predictions.csv")

# Generate complete analysis
evaluator.generate_report()
evaluator.plot_comprehensive_analysis()
```

### **Custom Analysis**
```python
# Focus on specific aspects
metrics = evaluator.calculate_metrics()
uncertainty_analysis = evaluator.uncertainty_calibration()
regime_performance = evaluator.regime_specific_analysis()

# Custom visualizations
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
evaluator.plot_time_series_with_regimes()
plt.title("Custom Analysis: Regime-Aware Predictions")
plt.show()
```

### **Comparative Analysis**
```python
# Compare multiple model versions
models = ['baseline', 'regime_aware', 'enhanced']
results = {}

for model in models:
    evaluator = ModelEvaluator(f"../results/{model}_predictions.csv")
    results[model] = evaluator.calculate_metrics()

# Performance comparison
comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

---

## 📚 Best Practices

### **Evaluation Workflow**
1. **Load predictions** with uncertainty estimates
2. **Generate standard report** for overview
3. **Examine visualizations** for patterns
4. **Focus on problem areas** (poor regimes, miscalibration)
5. **Compare with baselines** (non-regime-aware models)
6. **Document findings** for model improvement

### **Interpretation Tips**
- **Don't over-optimize** on single metrics
- **Consider regime context** when evaluating performance  
- **Validate uncertainty calibration** before deployment
- **Check temporal stability** with rolling evaluation
- **Compare against domain-specific benchmarks**

### **Red Flags**
- **Perfect performance**: Likely data leakage
- **Inconsistent calibration**: Regime-dependent issues
- **High bias**: Systematic model problems
- **Poor uncertainty**: Questionable confidence intervals

---

## 🤝 Contributing

To enhance the evaluation framework:

1. **Add new metrics** relevant to financial forecasting
2. **Implement advanced uncertainty analysis** methods
3. **Create regime-specific** evaluation techniques
4. **Develop interactive visualizations** for better insights
5. **Build automated reporting** systems

Areas for improvement:
- Real-time evaluation capabilities
- Model comparison frameworks  
- Risk-adjusted performance metrics
- Behavioral finance evaluation aspects
