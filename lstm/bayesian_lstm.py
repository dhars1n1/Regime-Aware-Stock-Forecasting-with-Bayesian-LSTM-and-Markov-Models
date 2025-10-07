"""
Regime-Aware Bayesian LSTM for Stock Price Forecasting with Uncertainty Quantification
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
import os
from typing import Dict, Tuple, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

class BayesianLSTM:
    def __init__(self, sequence_length: int = 20, lstm_units: int = 64, 
                 dropout_rate: float = 0.3, n_features: Optional[int] = None, 
                 monte_carlo_samples: int = 100, use_regime_label: bool = True):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.n_features = n_features
        self.monte_carlo_samples = monte_carlo_samples
        self.use_regime_label = use_regime_label
        self.model = None
        self.scalers = {}
        self.feature_columns = None
        self.regime_encoder = None
        
    def build_model(self) -> keras.Model:
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")
        
        model = keras.Sequential([
            layers.LSTM(
                self.lstm_units, 
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                name='lstm_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            layers.LSTM(
                self.lstm_units // 2, 
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            layers.Dense(32, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate, name='dropout_3'),
            layers.Dense(16, activation='relu', name='dense_2'),
            layers.Dropout(self.dropout_rate, name='dropout_4'),
            layers.Dense(1, name='output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    


    def prepare_regime_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        market_features = ['log_return', 'Volume', 'VIX']
        technical_features = ['RSI', 'MACD_diff', 'BB_high', 'BB_low', 'OBV']
    
        macro_features = []
        for feature in ['CPI', 'Unemployment', 'FedFunds']:
            if feature in df.columns and df[feature].notna().mean() > 0.8:
                macro_features.append(feature)
    
        lag_features = []
        for lag in [1, 2, 3, 5]:
            lag_col = f'return_lag_{lag}'
            if lag_col in df.columns:
                lag_features.append(lag_col)
    
        if self.use_regime_label and 'regime_label' in df.columns:
            regime_features = ['regime_label']
        else:
            regime_features = []
            for i in range(3):
                prob_col = f'regime_{i}_prob'
                if prob_col in df.columns:
                    regime_features.append(prob_col)
    
        volatility_features = []
        if 'log_return' in df.columns:
            df['volatility_5d'] = df['log_return'].rolling(5).std()
            volatility_features = ['volatility_5d']
    
        # ✅ Updated here: using 'sentiment_news' instead of 'sentiment_score'
        sentiment_features = []
        if 'sentiment_news' in df.columns and df['sentiment_news'].notna().mean() > 0.5:
            sentiment_features = ['sentiment_news']
    
        all_feature_groups = [
            ('Market', market_features),
            ('Technical', technical_features),
            ('Macro', macro_features),
            ('Lagged', lag_features),
            ('Regime', regime_features),
            ('Volatility', volatility_features),
            ('Sentiment', sentiment_features)
        ]
    
        self.feature_columns = []
        feature_summary = {}
    
        for group_name, features in all_feature_groups:
            available = [f for f in features if f in df.columns]
            self.feature_columns.extend(available)
            feature_summary[group_name] = len(available)
            if available:
                print(f"- {group_name}: {available}")
    
        print(f"\nTotal features: {len(self.feature_columns)}")
        for group, count in feature_summary.items():
            if count > 0:
                print(f"  {group}: {count}")
    
        feature_df = df[self.feature_columns].copy()
    
        if self.use_regime_label and 'regime_label' in self.feature_columns:
            from sklearn.preprocessing import LabelEncoder
            if self.regime_encoder is None:
                self.regime_encoder = LabelEncoder()
                feature_df['regime_label'] = self.regime_encoder.fit_transform(feature_df['regime_label'])
            else:
                feature_df['regime_label'] = self.regime_encoder.transform(feature_df['regime_label'])
    
        return feature_df
    
    
    
    def create_sequences(self, feature_data: pd.DataFrame, target_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        if len(feature_data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data: need at least {self.sequence_length + 1} samples")
        
        for i in range(self.sequence_length, len(feature_data)):
            sequence = feature_data.iloc[i-self.sequence_length:i].values
            X.append(sequence)
            y.append(target_data.iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        print(f"Input format: (n_samples={X.shape[0]}, sequence_length={X.shape[1]}, n_features={X.shape[2]})")
        
        return X, y
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        print("Preparing regime-aware features...")
        
        # Ensure log_return exists
        if 'log_return' not in df.columns:
            print("Creating log_return column...")
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Generate features
        feature_data = self.prepare_regime_aware_features(df)
        
        print(f"Original data shape: {feature_data.shape}")
        feature_data = feature_data.dropna()
        
        # 🚨 safeguard: drop duplicate indices if present
        if not feature_data.index.is_unique:
            dup_count = feature_data.index.duplicated().sum()
            print(f"Warning: {dup_count} duplicate indices found in features. Dropping duplicates...")
            feature_data = feature_data.loc[~feature_data.index.duplicated(keep="first")]
        
        print(f"After removing NaN and duplicates: {feature_data.shape}")
        
        # Align target with features
        # Ensure df index is unique to avoid reindex error
        df_unique = df.loc[~df.index.duplicated(keep="first")]

        # Align target safely
        target_data = df_unique['log_return'].reindex(feature_data.index)

        # Sanity check
        if len(target_data) != len(feature_data):
            raise ValueError(f"Target length {len(target_data)} does not match feature length {len(feature_data)}")

        
        # Temporal split
        split_idx = int(len(feature_data) * (1 - test_size))
        train_idx = feature_data.index[:split_idx]
        test_idx = feature_data.index[split_idx:]
        
        train_features = feature_data.loc[train_idx].copy()
        test_features = feature_data.loc[test_idx].copy()
        train_target = target_data.loc[train_idx].copy()
        test_target = target_data.loc[test_idx].copy()
        
        print(f"Train period: {train_features.index[0]} to {train_features.index[-1]}")
        print(f"Test period: {test_features.index[0]} to {test_features.index[-1]}")
        print(f"Train samples: {len(train_features)}, Test samples: {len(test_features)}")
        
        # Scaling
        print("Scaling features...")
        self.scalers['features'] = StandardScaler()
        train_features_scaled = self.scalers['features'].fit_transform(train_features)
        test_features_scaled = self.scalers['features'].transform(test_features)
        
        self.scalers['target'] = StandardScaler()
        train_target_array = train_target.to_numpy().reshape(-1, 1)
        test_target_array = test_target.to_numpy().reshape(-1, 1)
        
        train_target_scaled = self.scalers['target'].fit_transform(train_target_array).flatten()
        test_target_scaled = self.scalers['target'].transform(test_target_array).flatten()
        
        print(f"Scaled lengths: train={len(train_target_scaled)}, test={len(test_target_scaled)}")
        
        # ✅ sanity check: lengths must match
        if len(train_target_scaled) != len(train_features):
            raise ValueError(f"Mismatch after scaling: train_target={len(train_target_scaled)} vs train_features={len(train_features)}")
        if len(test_target_scaled) != len(test_features):
            raise ValueError(f"Mismatch after scaling: test_target={len(test_target_scaled)} vs test_features={len(test_features)}")
        
        # Repack as DataFrames/Series
        train_features_df = pd.DataFrame(train_features_scaled, columns=self.feature_columns, index=train_features.index)
        test_features_df = pd.DataFrame(test_features_scaled, columns=self.feature_columns, index=test_features.index)
        
        train_target_series = pd.Series(train_target_scaled, index=train_features.index)
        test_target_series = pd.Series(test_target_scaled, index=test_features.index)
        
        # Create sequences
        print("Creating sequences...")
        X_train, y_train = self.create_sequences(train_features_df, train_target_series)
        X_test, y_test = self.create_sequences(test_features_df, test_target_series)
        
        self.n_features = X_train.shape[2]
        test_dates = test_features.index[self.sequence_length:]
        
        print(f"Final shapes: X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"Features per timestep: {self.n_features}")
        
        return X_train, y_train, X_test, y_test, test_dates

    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        if self.model is None:
            self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            shuffle=False
        )
        
        return history
    
    def predict_with_uncertainty(self, X: np.ndarray, return_raw: bool = False) -> Dict:
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        print(f"Generating {self.monte_carlo_samples} MC samples for uncertainty estimation...")
        
        predictions = []
        
        for i in range(self.monte_carlo_samples):
            if i % 20 == 0:
                print(f"  Sample {i+1}/{self.monte_carlo_samples}")
            
            y_pred = self.model(X, training=True)
            predictions.append(y_pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        lower_ci_95 = np.percentile(predictions, 2.5, axis=0)
        upper_ci_95 = np.percentile(predictions, 97.5, axis=0)
        lower_ci_68 = np.percentile(predictions, 16, axis=0)
        upper_ci_68 = np.percentile(predictions, 84, axis=0)
        
        median_pred = np.median(predictions, axis=0)
        iqr = np.percentile(predictions, 75, axis=0) - np.percentile(predictions, 25, axis=0)
        
        result = {
            'mean': mean_pred,
            'median': median_pred,
            'std': std_pred,
            'iqr': iqr,
            'lower_ci_95': lower_ci_95,
            'upper_ci_95': upper_ci_95,
            'lower_ci_68': lower_ci_68,
            'upper_ci_68': upper_ci_68
        }
        
        if return_raw:
            result['all_predictions'] = predictions
        
        return result
    
    def evaluate_regime_performance(self, X_test: np.ndarray, y_test: np.ndarray, 
                                    test_dates: pd.DatetimeIndex, df: pd.DataFrame) -> Dict:
        print("Generating predictions with uncertainty...")
        
        # Predict with uncertainty
        predictions = self.predict_with_uncertainty(X_test, return_raw=True)
        
        # Inverse scale predictions & targets
        y_pred_orig = self.scalers['target'].inverse_transform(
            predictions['mean'].reshape(-1, 1)
        ).flatten()
        
        y_test_orig = self.scalers['target'].inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        lower_95_orig = self.scalers['target'].inverse_transform(
            predictions['lower_ci_95'].reshape(-1, 1)
        ).flatten()
        
        upper_95_orig = self.scalers['target'].inverse_transform(
            predictions['upper_ci_95'].reshape(-1, 1)
        ).flatten()
        
        # Ensure df has unique index to avoid reindexing errors
        df_unique = df.loc[~df.index.duplicated(keep='first')]
        
        # Align regimes exactly with sequence-based test samples
        test_regimes = df_unique['regime_label'].reindex(test_dates).values
        
        # Sanity check
        assert len(test_regimes) == len(y_test_orig), (
            f"Length mismatch: regimes={len(test_regimes)} vs predictions={len(y_test_orig)}"
        )
        
        # Overall metrics
        overall_results = {
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'r2': 1 - np.sum((y_test_orig - y_pred_orig)**2) / np.sum((y_test_orig - np.mean(y_test_orig))**2)
        }
        
        # Coverage metrics
        coverage_95 = np.mean((y_test_orig >= lower_95_orig) & (y_test_orig <= upper_95_orig))
        interval_width_95 = np.mean(upper_95_orig - lower_95_orig)
        overall_results.update({
            'coverage_95': coverage_95,
            'interval_width_95': interval_width_95
        })
        
        # Regime-specific metrics
        regime_results = {}
        unique_regimes = np.unique(test_regimes)
        
        for regime in unique_regimes:
            mask = test_regimes == regime
            if np.sum(mask) > 5:  # skip regimes with too few samples
                regime_pred = y_pred_orig[mask]
                regime_actual = y_test_orig[mask]
                regime_lower = lower_95_orig[mask]
                regime_upper = upper_95_orig[mask]
                
                regime_coverage = np.mean((regime_actual >= regime_lower) & (regime_actual <= regime_upper))
                regime_width = np.mean(regime_upper - regime_lower)
                
                regime_results[regime] = {
                    'mse': mean_squared_error(regime_actual, regime_pred),
                    'mae': mean_absolute_error(regime_actual, regime_pred),
                    'rmse': np.sqrt(mean_squared_error(regime_actual, regime_pred)),
                    'coverage_95': regime_coverage,
                    'interval_width_95': regime_width,
                    'count': np.sum(mask),
                    'std_actual': np.std(regime_actual),
                    'std_pred': np.std(regime_pred)
                }
        
        # Return all results
        return {
            'overall': overall_results,
            'by_regime': regime_results,
            'predictions': {
                'mean': y_pred_orig,
                'lower_95': lower_95_orig,
                'upper_95': upper_95_orig,
                'actual': y_test_orig,
                'regimes': test_regimes,
                'uncertainty': predictions['std']
            }
        }



    
    def create_comprehensive_visualizations(self, test_dates: pd.DatetimeIndex, 
                                          evaluation_results: Dict, 
                                          save_path: str = "results") -> None:
        os.makedirs(save_path, exist_ok=True)
        
        predictions = evaluation_results['predictions']
        y_actual = predictions['actual']
        y_pred = predictions['mean']
        lower_95 = predictions['lower_95']
        upper_95 = predictions['upper_95']
        test_regimes = predictions['regimes']
        uncertainty = predictions['uncertainty']
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        ax1 = axes[0]
        ax1.plot(test_dates, y_actual, label='Actual Returns', color='black', alpha=0.8, linewidth=1.5)
        ax1.plot(test_dates, y_pred, label='Predicted Returns', color='blue', linewidth=1.5)
        ax1.fill_between(test_dates, lower_95, upper_95, 
                        alpha=0.3, color='blue', label='95% Confidence Interval')
        ax1.set_title('Regime-Aware Bayesian LSTM Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Log Returns', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        regime_colors = {'Crisis': '#e74c3c', 'Normal': '#f39c12', 'Bull': '#27ae60'}
        
        for regime in np.unique(test_regimes):
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                ax2.scatter(test_dates[mask], y_actual[mask], 
                           c=regime_colors.get(regime, 'gray'), 
                           label=f'{regime} Regime', 
                           alpha=0.7, s=15)
        
        ax2.set_title('Actual Returns Colored by Market Regime', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log Returns', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        ax3.plot(test_dates, uncertainty, color='purple', alpha=0.7, linewidth=1.5)
        ax3.fill_between(test_dates, 0, uncertainty, alpha=0.3, color='purple')
        ax3.set_title('Prediction Uncertainty Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Prediction Std Dev', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/predictions_with_uncertainty.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for regime in np.unique(test_regimes):
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                ax.scatter(y_actual[mask], y_pred[mask], 
                          c=regime_colors.get(regime, 'gray'), 
                          alpha=0.6, s=30, label=f'{regime} Regime')
        
        min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Log Returns', fontsize=12)
        ax.set_ylabel('Predicted Log Returns', fontsize=12)
        ax.set_title('Predicted vs Actual Returns by Regime', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_path}/prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for regime in np.unique(test_regimes):
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                axes[0].hist(uncertainty[mask], alpha=0.6, bins=30, 
                           label=f'{regime} Regime', 
                           color=regime_colors.get(regime, 'gray'))
        
        axes[0].set_xlabel('Prediction Uncertainty (Std Dev)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Prediction Uncertainty by Regime')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        regime_errors = []
        regime_labels = []
        for regime in np.unique(test_regimes):
            mask = test_regimes == regime
            if np.sum(mask) > 0:
                errors = np.abs(y_actual[mask] - y_pred[mask])
                regime_errors.extend(errors)
                regime_labels.extend([regime] * len(errors))
        
        error_df = pd.DataFrame({'Error': regime_errors, 'Regime': regime_labels})
        sns.boxplot(data=error_df, x='Regime', y='Error', ax=axes[1])
        axes[1].set_title('Absolute Errors by Regime')
        axes[1].set_ylabel('Absolute Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {save_path}/")
        
    def save_predictions_data(self, test_dates: pd.DatetimeIndex, 
                            evaluation_results: Dict, 
                            save_path: str = "results") -> None:
        os.makedirs(save_path, exist_ok=True)
        
        predictions = evaluation_results['predictions']
        
        results_df = pd.DataFrame({
            'Date': test_dates,
            'Actual_Return': predictions['actual'],
            'Predicted_Return': predictions['mean'], 
            'Lower_95_CI': predictions['lower_95'],
            'Upper_95_CI': predictions['upper_95'],
            'Prediction_Uncertainty': predictions['uncertainty'],
            'Regime': predictions['regimes']
        })
        
        results_df['Absolute_Error'] = np.abs(results_df['Actual_Return'] - results_df['Predicted_Return'])
        results_df['Squared_Error'] = (results_df['Actual_Return'] - results_df['Predicted_Return'])**2
        results_df['In_CI_95'] = ((results_df['Actual_Return'] >= results_df['Lower_95_CI']) & 
                                 (results_df['Actual_Return'] <= results_df['Upper_95_CI']))
        
        results_df.to_csv(f"{save_path}/detailed_predictions.csv", index=False)
        
        summary_stats = {
            'Overall_MSE': evaluation_results['overall']['mse'],
            'Overall_MAE': evaluation_results['overall']['mae'], 
            'Overall_RMSE': evaluation_results['overall']['rmse'],
            'Overall_R2': evaluation_results['overall']['r2'],
            'Coverage_95': evaluation_results['overall']['coverage_95'],
            'Average_Interval_Width': evaluation_results['overall']['interval_width_95']
        }
        
        for regime, stats in evaluation_results['by_regime'].items():
            for metric, value in stats.items():
                summary_stats[f'{regime}_{metric}'] = value
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f"{save_path}/performance_summary.csv", index=False)
        
        print(f"Prediction data saved to {save_path}/")
        print(f"- Detailed predictions: detailed_predictions.csv")  
        print(f"- Performance summary: performance_summary.csv")


def save_all_inference_artifacts(bayesian_lstm_model, save_path: str = "results"):
    """
    Save all model artifacts needed for inference
    
    This function saves:
    1. Feature and target scalers (needed to transform new data)
    2. Regime encoder (needed to encode regime labels)
    3. Model metadata (feature names, configuration)
    
    Args:
        bayesian_lstm_model: Trained BayesianLSTM instance
        save_path: Directory to save artifacts
    """
    import pickle
    import json
    
    print(f"💾 Saving inference artifacts to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save scalers (critical for data preprocessing)
    scalers_file = f"{save_path}/scalers.pkl"
    with open(scalers_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.scalers, f)
    print(f"✅ Saved scalers: {scalers_file}")
    
    # 2. Save regime encoder (needed for regime label transformation)
    encoder_file = f"{save_path}/regime_encoder.pkl"
    with open(encoder_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.regime_encoder, f)
    print(f"✅ Saved regime encoder: {encoder_file}")
    
    # 3. Save model metadata (configuration and feature info)
    metadata = {
        'feature_columns': bayesian_lstm_model.feature_columns,
        'n_features': bayesian_lstm_model.n_features,
        'sequence_length': bayesian_lstm_model.sequence_length,
        'dropout_rate': bayesian_lstm_model.dropout_rate,
        'lstm_units': bayesian_lstm_model.lstm_units,
        'monte_carlo_samples': bayesian_lstm_model.monte_carlo_samples,
        'use_regime_label': bayesian_lstm_model.use_regime_label
    }
    
    metadata_file = f"{save_path}/model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved metadata: {metadata_file}")
    
    print(f"📦 All inference artifacts saved!")


def main():
    print("Starting Regime-Aware Bayesian LSTM Training")
    print("=" * 60)
    
    print("Loading data with regime information...")
    df = pd.read_csv("../data/data_with_regimes.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Regime distribution:\n{df['regime_label'].value_counts()}")
    
    model = BayesianLSTM(
        sequence_length=20,
        lstm_units=64,
        dropout_rate=0.3,
        monte_carlo_samples=100,
        use_regime_label=True
    )
    
    print("\nPreparing data...")
    X_train, y_train, X_test, y_test, test_dates = model.prepare_data(df, test_size=0.2)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    print("\nBuilding Bayesian LSTM architecture...")
    model.build_model()
    print(model.model.summary())
    
    print("\nTraining Bayesian LSTM...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    print("\nEvaluating model performance...")
    evaluation_results = model.evaluate_regime_performance(X_test, y_test, test_dates, df)
    
    print("\nPERFORMANCE SUMMARY")
    print("-" * 40)
    
    overall = evaluation_results['overall']
    print(f"Overall Performance:")
    print(f"  MSE: {overall['mse']:.6f}")
    print(f"  MAE: {overall['mae']:.6f}") 
    print(f"  RMSE: {overall['rmse']:.6f}")
    print(f"  R²: {overall['r2']:.4f}")
    print(f"  95% Coverage: {overall['coverage_95']:.3f}")
    print(f"  Avg Interval Width: {overall['interval_width_95']:.6f}")
    
    print(f"\nRegime-Specific Performance:")
    for regime, metrics in evaluation_results['by_regime'].items():
        print(f"  {regime} Regime ({metrics['count']} samples):")
        print(f"    MSE: {metrics['mse']:.6f}")
        print(f"    MAE: {metrics['mae']:.6f}")
        print(f"    Coverage: {metrics['coverage_95']:.3f}")
    
    print("\nCreating visualizations...")
    model.create_comprehensive_visualizations(test_dates, evaluation_results)
    
    print("\nSaving results...")
    os.makedirs("results", exist_ok=True)
    
    # Save the trained model
    model.model.save("results/bayesian_lstm_model.h5")
    print("Model saved to results/bayesian_lstm_model.h5")
    
    # Save all artifacts needed for inference
    print("\nSaving all artifacts for inference...")
    save_all_inference_artifacts(model, save_path="results")
    
    model.save_predictions_data(test_dates, evaluation_results)
    
    print("\nTraining and evaluation complete!")
    print("All artifacts saved - ready for inference!")
    print("=" * 60)
    
    return model, evaluation_results

if __name__ == "__main__":
    model, results = main()