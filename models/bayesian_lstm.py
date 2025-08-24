"""
Regime-Aware Bayesian LSTM for Stock Price Forecasting with Uncertainty Quantification
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

class BayesianLSTM:
    def __init__(self, sequence_length=60, lstm_units=64, dropout_rate=0.3, 
                 n_features=None, monte_carlo_samples=100):
        """
        Initialize Bayesian LSTM model
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for Monte Carlo Dropout
            n_features: Number of input features
            monte_carlo_samples: Number of MC samples for uncertainty estimation
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.n_features = n_features
        self.monte_carlo_samples = monte_carlo_samples
        self.model = None
        self.scalers = {}
        self.feature_columns = None
        
    def build_model(self):
        """Build the Bayesian LSTM model with Monte Carlo Dropout"""
        model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(self.lstm_units, 
                       return_sequences=True,
                       input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(self.dropout_rate),
            
            # Second LSTM layer
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            
            # Dense layers with dropout for uncertainty
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer - predicting next day's return
            layers.Dense(1)
        ])
        
        # Compile with custom loss and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_regime_aware_features(self, df):
        """
        Prepare features including regime information
        """
        # Core market features
        market_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'VIX'
        ]
        
        # Technical indicators
        technical_features = [
            'RSI', 'MACD_diff', 'BB_high', 'BB_low', 'OBV'
        ]
        
        # Macro features
        macro_features = [
            'CPI', 'Unemployment', 'FedFunds'
        ]
        
        # Lagged returns
        lag_features = [
            'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5'
        ]
        
        # Regime features - THIS IS THE KEY ADDITION
        regime_features = [
            'regime_0_prob', 'regime_1_prob', 'regime_2_prob'
        ]
        
        # Sentiment features (if available)
        sentiment_features = []
        if 'sentiment_score' in df.columns:
            sentiment_features = ['sentiment_score']
        
        # Combine all features
        self.feature_columns = (market_features + technical_features + 
                               macro_features + lag_features + 
                               regime_features + sentiment_features)
        
        # Filter to available columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        print(f"Using {len(self.feature_columns)} features:")
        print(f"- Market: {len([f for f in market_features if f in available_features])}")
        print(f"- Technical: {len([f for f in technical_features if f in available_features])}")
        print(f"- Macro: {len([f for f in macro_features if f in available_features])}")
        print(f"- Lagged: {len([f for f in lag_features if f in available_features])}")
        print(f"- Regime: {len([f for f in regime_features if f in available_features])}")
        print(f"- Sentiment: {len([f for f in sentiment_features if f in available_features])}")
        
        return df[self.feature_columns].copy()
    
    def create_sequences(self, data, target_col='log_return'):
        """
        Create sequences for LSTM training
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Features for the sequence
            X.append(data.iloc[i-self.sequence_length:i].values)
            # Target (next day's log return)
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare and scale data for training
        """
        # Prepare features
        feature_data = self.prepare_regime_aware_features(df)
        
        # Add target column
        feature_data['log_return'] = df['log_return']
        
        # Remove any remaining NaN values
        feature_data = feature_data.dropna()
        
        # Split data
        split_idx = int(len(feature_data) * (1 - test_size))
        train_data = feature_data.iloc[:split_idx]
        test_data = feature_data.iloc[split_idx:]
        
        # Scale features (fit on training data only)
        self.scalers['features'] = MinMaxScaler()
        train_features_scaled = self.scalers['features'].fit_transform(
            train_data[self.feature_columns]
        )
        test_features_scaled = self.scalers['features'].transform(
            test_data[self.feature_columns]
        )
        
        # Scale target
        self.scalers['target'] = MinMaxScaler()
        train_target_scaled = self.scalers['target'].fit_transform(
            train_data[['log_return']]
        ).flatten()
        test_target_scaled = self.scalers['target'].transform(
            test_data[['log_return']]
        ).flatten()
        
        # Create scaled dataframes
        train_scaled = pd.DataFrame(train_features_scaled, 
                                   columns=self.feature_columns,
                                   index=train_data.index)
        train_scaled['log_return'] = train_target_scaled
        
        test_scaled = pd.DataFrame(test_features_scaled,
                                  columns=self.feature_columns,
                                  index=test_data.index)
        test_scaled['log_return'] = test_target_scaled
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)
        
        self.n_features = X_train.shape[2]
        
        return X_train, y_train, X_test, y_test, test_data.index[self.sequence_length:]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the Bayesian LSTM model
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict_with_uncertainty(self, X, training=True):
        """
        Make predictions with uncertainty quantification using Monte Carlo Dropout
        """
        predictions = []
        
        for _ in range(self.monte_carlo_samples):
            # Enable dropout during inference for uncertainty estimation
            y_pred = self.model(X, training=training)
            predictions.append(y_pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals (assuming normal distribution)
        lower_ci = mean_pred - 1.96 * std_pred  # 95% CI
        upper_ci = mean_pred + 1.96 * std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'all_predictions': predictions
        }
    
    def evaluate_regime_performance(self, X_test, y_test, test_dates, df):
        """
        Evaluate model performance by regime
        """
        # Get predictions
        predictions = self.predict_with_uncertainty(X_test)
        
        # Inverse transform predictions and targets
        y_pred_orig = self.scalers['target'].inverse_transform(
            predictions['mean'].reshape(-1, 1)
        ).flatten()
        y_test_orig = self.scalers['target'].inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # Get regime labels for test period
        test_regimes = df.loc[test_dates, 'regime_label'].values
        
        # Evaluate by regime
        results = {}
        for regime in ['Crisis', 'Normal', 'Bull']:
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                regime_pred = y_pred_orig[mask]
                regime_actual = y_test_orig[mask]
                
                results[regime] = {
                    'mse': mean_squared_error(regime_actual, regime_pred),
                    'mae': mean_absolute_error(regime_actual, regime_pred),
                    'count': np.sum(mask)
                }
        
        return results, predictions, y_test_orig, test_regimes
    
    def plot_predictions(self, test_dates, y_actual, predictions, test_regimes):
        """
        Plot predictions with uncertainty bands and regime information
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Predictions with uncertainty
        ax1.plot(test_dates, y_actual, label='Actual', color='black', alpha=0.7)
        ax1.plot(test_dates, predictions['mean'], label='Predicted', color='blue')
        
        # Uncertainty bands
        ax1.fill_between(test_dates, 
                        predictions['lower_ci'], 
                        predictions['upper_ci'],
                        alpha=0.3, color='blue', label='95% CI')
        
        ax1.set_title('Regime-Aware Bayesian LSTM Predictions with Uncertainty')
        ax1.set_ylabel('Log Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Regime coloring
        colors = {'Crisis': 'red', 'Normal': 'orange', 'Bull': 'green'}
        for regime in ['Crisis', 'Normal', 'Bull']:
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                ax2.scatter(test_dates[mask], y_actual[mask], 
                           c=colors[regime], label=f'{regime} (Actual)', 
                           alpha=0.6, s=20)
        
        ax2.set_title('Actual Returns by Market Regime')
        ax2.set_ylabel('Log Returns')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run regime-aware Bayesian LSTM training and evaluation
    """
    # Load data with regimes
    print("Loading data with regime information...")
    df = pd.read_csv("data/data_with_regimes.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Regime distribution:\n{df['regime_label'].value_counts()}")
    
    # Initialize Bayesian LSTM
    model = BayesianLSTM(
        sequence_length=60,
        lstm_units=64,
        dropout_rate=0.3,
        monte_carlo_samples=100
    )
    
    # Prepare data
    print("\nPreparing data...")
    X_train, y_train, X_test, y_test, test_dates = model.prepare_data(df, test_size=0.2)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train model
    print("\nTraining Bayesian LSTM...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # Evaluate model
    print("\nEvaluating model performance by regime...")
    regime_results, predictions, y_actual, test_regimes = model.evaluate_regime_performance(
        X_test, y_test, test_dates, df
    )
    
    # Print results
    print("\nPerformance by Regime:")
    for regime, metrics in regime_results.items():
        print(f"{regime:>8}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"Count={metrics['count']}")
    
    # Plot results
    model.plot_predictions(test_dates, y_actual, predictions, test_regimes)
    
    # Save model
    print("\nSaving model...")
    os.makedirs("models", exist_ok=True)
    model.model.save("models/bayesian_lstm_model.h5")
    
    # Save predictions for further analysis
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_actual,
        'Predicted': predictions['mean'],
        'Lower_CI': predictions['lower_ci'],
        'Upper_CI': predictions['upper_ci'],
        'Uncertainty': predictions['std'],
        'Regime': test_regimes
    })
    results_df.to_csv("results/bayesian_lstm_predictions.csv", index=False)
    
    print("✅ Training and evaluation complete!")
    
    return model, regime_results, predictions

if __name__ == "__main__":
    model, results, predictions = main()
