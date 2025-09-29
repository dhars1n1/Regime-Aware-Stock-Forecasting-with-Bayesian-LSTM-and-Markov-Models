# Regime-Aware Stock Forecasting with Bayesian LSTM and Markov Models

A comprehensive implementation of **LSTM with Monte Carlo Dropout (MC Dropout) for stock forecasting** that incorporates market regime information from Markov Switching Models (MSM) to provide **one-step-ahead forecasts with predictive uncertainty** and comprehensive visualization.

## 🎯 Project Overview

This project implements a state-of-the-art Bayesian LSTM system for stock return forecasting with the following key features:

- **Monte Carlo Dropout**: Uncertainty quantification through MC sampling during inference
- **Regime Awareness**: Uses hard regime labels from MSM as input features  
- **Comprehensive Evaluation**: Advanced metrics for uncertainty calibration and model performance
- **Rich Visualizations**: Interactive plots and comprehensive analysis dashboards
- **End-to-End Pipeline**: Complete automated workflow from data to results

## 🏗️ Architecture

### Input Format
- **Shape**: `(batch_size, sequence_length=20, n_features=27+1)` *(27 features + 1 regime label)*
- **Features**: All 27 numerical features in your dataset (auto-detected)
- **Available Features**: Open, High, Low, Close, Volume, RSI, MACD_diff, BB_high, BB_low, OBV, VIX, CPI, Unemployment, FedFunds, sentiment, returns, return_lags, Bullish, Neutral, Bearish, sentiment_score, is_crisis, fed_meeting, earnings_season, etc.
- **Target**: Next-day log return distribution

### Model Components

1. **BayesianLSTM**: Core LSTM with Monte Carlo Dropout
2. **RegimeAwareDataProcessor**: Feature engineering and sequence creation
3. **BayesianLSTMTrainer**: Training pipeline with callbacks and monitoring
4. **BayesianLSTMPredictor**: Prediction engine with uncertainty quantification
5. **BayesianLSTMVisualizer**: Comprehensive visualization system
6. **BayesianLSTMEvaluator**: Advanced evaluation and calibration metrics

## 📁 Project Structure

```
Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_complete_pipeline.py          # Main pipeline script
├── setup_and_run.py                  # Setup and quick start
├── config.yaml                       # Configuration file (generated)
│
├── data/                              # Data processing
│   ├── data.csv                       # Raw stock data
│   ├── data_with_regimes.csv         # Data with MSM regime labels
│   ├── data_processor.py             # ✨ NEW: Comprehensive data processor
│   └── [other data files...]
│
├── models/                           # Model implementations
│   ├── bayesian_lstm.py             # ✨ ENHANCED: Core Bayesian LSTM
│   ├── train_model.py               # ✨ NEW: Training infrastructure
│   ├── prediction_engine.py         # ✨ NEW: Prediction system
│   └── visualization_system.py      # ✨ NEW: Visualization suite
│
├── evaluation/                       # Evaluation system
│   ├── evaluation_metrics.py        # ✨ NEW: Comprehensive metrics
│   └── model_evaluation.py          # Original evaluation
│
├── msm/                              # Markov Switching Model
│   └── msm.py                        # MSM implementation
│
├── results/                          # Generated results (created at runtime)
│   ├── models/                       # Trained model files
│   ├── predictions/                  # Prediction outputs
│   ├── visualizations/              # Generated plots
│   ├── evaluation/                   # Evaluation reports
│   └── reports/                      # Final summaries
│
└── visualization/                    # Architecture diagrams
    └── [diagram files...]
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Regime-Aware-Stock-Forecasting-with-Bayesian-LSTM-and-Markov-Models

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Configuration

```bash
# Create a sample configuration file
python run_complete_pipeline.py --create_config
```

### 3. Run the Complete Pipeline

```bash
# Run with default settings
python run_complete_pipeline.py

# Run with custom configuration
python run_complete_pipeline.py --config_path config.yaml --output_dir my_results

# Run with specific data file
python run_complete_pipeline.py --data_path data/data_with_regimes.csv --output_dir results
```

## ⚙️ Configuration

The system uses a YAML configuration file for easy customization:

```yaml
data_path: 'data/data_with_regimes.csv'
output_dir: 'results'

data:
  feature_columns: 'auto'  # Auto-detect all available features
  exclude_columns: ['Date', 'regime_viterbi', 'regime_0_prob', 'regime_1_prob', 'regime_2_prob']
  target_column: 'log_return'
  regime_column: 'regime_label'
  test_size: 0.2

model:
  sequence_length: 20
  lstm_units: 64
  dropout_rate: 0.3
  n_monte_carlo: 100

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

prediction:
  n_monte_carlo: 200
  multi_step_horizon: 5
```

## 🔧 Core Components

### 1. Data Processing (`data/data_processor.py`)

**RegimeAwareDataProcessor** handles:
- Feature engineering and scaling
- Sequence creation for LSTM input
- Regime label encoding
- Temporal train/test splitting

```python
processor = RegimeAwareDataProcessor(
    sequence_length=20,
    feature_columns=['log_return', 'volume', 'volatility_5d'],
    target_column='log_return',
    regime_column='regime_label'
)

processed_data = processor.prepare_data_for_training(data, test_size=0.2)
```

### 2. Model Architecture (`models/bayesian_lstm.py`)

**BayesianLSTM** features:
- LSTM layers with Monte Carlo Dropout
- Regime-aware feature integration
- Uncertainty quantification via MC sampling
- Comprehensive evaluation methods

```python
model = BayesianLSTM(
    input_shape=(20, 4),  # sequence_length, n_features
    lstm_units=64,
    dropout_rate=0.3,
    n_monte_carlo=100
)

# Get predictions with uncertainty
predictions = model.predict_with_uncertainty(X_test, n_monte_carlo=100)
```

### 3. Training System (`models/train_model.py`)

**BayesianLSTMTrainer** provides:
- Automated training pipeline
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Training history visualization
- Model checkpointing

```python
trainer = BayesianLSTMTrainer(model=model, save_dir='models')
history = trainer.run_training_pipeline(
    X_train, y_train, X_val, y_val,
    epochs=100, batch_size=32
)
```

### 4. Prediction Engine (`models/prediction_engine.py`)

**BayesianLSTMPredictor** offers:
- One-step-ahead predictions
- Multi-step forecasting
- Confidence interval analysis
- Prediction visualization

```python
predictor = BayesianLSTMPredictor(model)

# Single prediction
next_prediction = predictor.predict_next_return(last_sequence)

# Multi-step forecast
forecast = predictor.multi_step_prediction(last_sequence, n_steps=5)
```

### 5. Evaluation System (`evaluation/evaluation_metrics.py`)

**BayesianLSTMEvaluator** includes:
- Uncertainty calibration analysis
- Coverage and reliability metrics
- Regime-specific performance
- Statistical significance tests

```python
evaluator = BayesianLSTMEvaluator()
results = evaluator.comprehensive_evaluation(
    actual=y_test,
    predictions=predictions,
    regime_labels=regime_labels
)
```

### 6. Visualization Suite (`models/visualization_system.py`)

**BayesianLSTMVisualizer** creates:
- Prediction plots with uncertainty bands
- Regime-colored analysis
- Performance dashboards
- Interactive Plotly visualizations

```python
visualizer = BayesianLSTMVisualizer(save_dir='visualizations')
visualizer.generate_all_visualizations(
    dates, predictions, evaluation_results, actual_values, regime_labels
)
```

## 📊 Output and Results

After running the pipeline, you'll find:

### Generated Files

```
results/
├── models/
│   ├── bayesian_lstm_model.h5        # Trained model weights
│   ├── training_history.json         # Training metrics
│   └── model_architecture.json       # Model configuration
│
├── predictions/
│   └── predictions.json              # Predictions with uncertainty
│
├── visualizations/
│   ├── predictions_comprehensive.png # Main prediction plot
│   ├── regime_analysis.png          # Regime-specific analysis
│   ├── uncertainty_distribution.png  # Uncertainty analysis
│   ├── performance_dashboard.png     # Performance overview
│   └── interactive_predictions.html  # Interactive plot
│
├── evaluation/
│   └── comprehensive_report.txt      # Detailed evaluation
│
└── reports/
    ├── final_report.json            # Complete results summary
    └── executive_summary.txt        # Human-readable summary
```

### Key Metrics

The system provides comprehensive evaluation including:

- **Basic Performance**: MSE, MAE, RMSE, R², Correlation
- **Uncertainty Quality**: Calibration error, Coverage analysis, PIT uniformity
- **Regime Analysis**: Performance by market regime, Cross-regime comparisons
- **Advanced Diagnostics**: Residual analysis, Statistical tests

## 🎯 Key Features

### ✨ Uncertainty Quantification

- **Monte Carlo Dropout**: Proper implementation with `training=True` during inference
- **Calibration Analysis**: Coverage analysis across multiple confidence levels
- **Confidence Intervals**: 68%, 95% prediction intervals
- **Uncertainty Visualization**: Time-series uncertainty plots

### ⚡ Regime Awareness

- **Hard Regime Labels**: Uses MSM output as categorical features
- **Regime-Specific Analysis**: Performance metrics by market regime
- **Regime Visualization**: Color-coded plots by regime type

### 📈 Comprehensive Evaluation

- **Statistical Tests**: Ljung-Box, Breusch-Pagan, normality tests
- **Coverage Analysis**: Empirical vs nominal coverage rates
- **Sharpness Metrics**: Prediction interval width analysis
- **Model Diagnostics**: Residual analysis and adequacy tests

### 🎨 Rich Visualizations

- **Static Plots**: High-quality matplotlib/seaborn visualizations
- **Interactive Plots**: Plotly-based interactive explorations
- **Dashboard Views**: Multi-panel performance overviews
- **Uncertainty Bands**: Proper uncertainty visualization

## 🔬 Technical Details

### Monte Carlo Dropout Implementation

```python
def predict_with_uncertainty(self, X, n_monte_carlo=100):
    predictions = []
    
    for _ in range(n_monte_carlo):
        # Keep training=True to enable dropout during inference
        pred = self.model(X, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    return {
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0),
        'ci_95_lower': np.percentile(predictions, 2.5, axis=0),
        'ci_95_upper': np.percentile(predictions, 97.5, axis=0)
    }
```

### Regime Feature Engineering

```python
def engineer_features(self, data):
    # Add technical indicators
    data['volatility_5d'] = data['log_return'].rolling(5).std()
    data['volume_ma'] = data['volume'].rolling(5).mean()
    
    # Encode regime labels
    le = LabelEncoder()
    data['regime_encoded'] = le.fit_transform(data['regime_label'])
    
    return data, le
```

### Uncertainty Calibration

```python
def evaluate_uncertainty_calibration(self, actual, predicted_mean, predicted_std):
    confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
    calibration_results = {}
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower = predicted_mean - z_score * predicted_std
        upper = predicted_mean + z_score * predicted_std
        
        empirical_coverage = np.mean((actual >= lower) & (actual <= upper))
        calibration_results[f'coverage_{conf_level}'] = empirical_coverage
    
    return calibration_results
```

## 📚 Dependencies

### Core Requirements
- `tensorflow>=2.8.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities

### Visualization
- `matplotlib>=3.4.0` - Static plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `plotly>=5.0.0` - Interactive plots

### Additional
- `scipy>=1.7.0` - Statistical functions
- `pyyaml>=5.4.0` - Configuration files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Usage

If you use this code in academic work, please consider citing:

```bibtex
@software{regime_aware_bayesian_lstm,
  title={Regime-Aware Stock Forecasting with Bayesian LSTM and Markov Models},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Format**: Check CSV format matches expected column names
3. **Memory Issues**: Reduce `n_monte_carlo` or `batch_size` for large datasets
4. **Convergence**: Adjust learning rate or increase epochs

### Performance Tips

1. **GPU Acceleration**: Use TensorFlow-GPU for faster training
2. **Batch Size**: Optimize batch size for your hardware
3. **Sequence Length**: Balance between memory usage and performance
4. **Monte Carlo Samples**: More samples = better uncertainty but slower inference

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in individual module files
- Review the generated evaluation reports for model insights

---

## 🎉 **SYSTEM READY - FULLY AUTOMATIC FEATURE DETECTION**

Your Bayesian LSTM system now automatically detects and uses **ALL 27 numerical features** from your dataset:

**✨ Auto-Detected Features:**
- **Market Data**: Open, High, Low, Close, Volume, returns, log_return
- **Technical Indicators**: RSI, MACD_diff, BB_high, BB_low, OBV
- **Economic Indicators**: VIX, CPI, Unemployment, FedFunds
- **Sentiment Features**: sentiment, Bullish, Neutral, Bearish, sentiment_score
- **Market Context**: is_crisis, fed_meeting, earnings_season
- **Temporal Features**: return_lag_1, return_lag_2, return_lag_3, return_lag_5
- **Regime Information**: regime_label (encoded)

**🚀 Input Shape**: `(batch_size, 20, 28)` - 27 features + 1 regime label

**Just run:**
```bash
python run_complete_pipeline.py
```

The system will automatically detect, process, and use all available features for maximum predictive power!

---

**Built with ❤️ for quantitative finance and uncertainty-aware machine learning**