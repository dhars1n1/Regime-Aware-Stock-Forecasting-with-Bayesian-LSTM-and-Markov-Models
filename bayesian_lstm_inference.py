"""
🔮 Bayesian LSTM Inference Engine
===============================

This script demonstrates complete inference pipeline for regime-aware stock forecasting
using Bayesian LSTM with Monte Carlo Dropout for uncertainty quantification.

Key Features:
- Load trained model and all artifacts
- Perform Monte Carlo Dropout inference 
- Comprehensive evaluation with regime-specific analysis
- Detailed visualizations and uncertainty plots
- Production-ready inference functions

Author: AI Assistant
Date: October 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🔮 Bayesian LSTM Inference Engine")
print("=" * 60)


class BayesianLSTMInferenceEngine:
    """
    Complete inference engine for Bayesian LSTM with uncertainty quantification
    
    This class handles:
    1. Loading trained model and preprocessing artifacts
    2. Monte Carlo Dropout inference with uncertainty estimation
    3. Comprehensive evaluation with multiple metrics
    4. Advanced visualizations and regime-specific analysis
    """
    
    def _init_(self, model_path: str = "results"):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to directory containing model artifacts
            
        The directory should contain:
        - bayesian_lstm_model.h5 (trained model)
        - scalers.pkl (feature and target scalers) 
        - regime_encoder.pkl (regime label encoder)
        - model_metadata.json (configuration and feature info)
        """
        self.model_path = model_path
        self.model = None
        self.scalers = None
        self.regime_encoder = None
        self.metadata = None
        
        # Results storage
        self.predictions = None
        self.evaluation_metrics = None
        self.regime_analysis = None
        
        print(f"📁 Inference engine initialized with path: {model_path}")
    
    def load_artifacts(self) -> 'BayesianLSTMInferenceEngine':
        """
        Load all trained model artifacts required for inference
        
        Why this step is critical:
        - The .h5 file only contains model weights and architecture
        - We need scalers to transform raw data to model's expected scale
        - Regime encoder maps text labels (Crisis/Normal/Bull) to integers
        - Metadata ensures we use features in the same order as training
        
        Returns:
            Self (for method chaining)
            
        Raises:
            FileNotFoundError: If any required artifact is missing
        """
        print("\n🔄 Loading trained model artifacts...")
        
        # 1. Load the trained neural network (.h5 file)
        model_file = f"{self.model_path}/bayesian_lstm_model.h5"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        self.model = keras.models.load_model(model_file)
        print(f"✅ Loaded trained model: {model_file}")
        print(f"   Model input shape: {self.model.input.shape}")
        print(f"   Model output shape: {self.model.output.shape}")
        
        # 2. Load feature and target scalers
        scalers_file = f"{self.model_path}/scalers.pkl"
        if not os.path.exists(scalers_file):
            raise FileNotFoundError(f"Scalers file not found: {scalers_file}")
            
        with open(scalers_file, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"✅ Loaded scalers: {scalers_file}")
        print(f"   Feature scaler: {type(self.scalers['features'])._name_}")
        print(f"   Target scaler: {type(self.scalers['target'])._name_}")
        
        # 3. Load regime encoder  
        encoder_file = f"{self.model_path}/regime_encoder.pkl"
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Regime encoder not found: {encoder_file}")
            
        with open(encoder_file, 'rb') as f:
            self.regime_encoder = pickle.load(f)
        print(f"✅ Loaded regime encoder: {encoder_file}")
        print(f"   Regime classes: {self.regime_encoder.classes_}")
        
        # 4. Load model configuration and metadata
        metadata_file = f"{self.model_path}/model_metadata.json"
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded metadata: {metadata_file}")
        
        # Display model configuration
        print(f"\n📊 Model Configuration:")
        print(f"   Sequence Length: {self.metadata['sequence_length']}")
        print(f"   Features: {self.metadata['n_features']}")
        print(f"   LSTM Units: {self.metadata['lstm_units']}")
        print(f"   Dropout Rate: {self.metadata['dropout_rate']}")
        print(f"   MC Samples: {self.metadata['monte_carlo_samples']}")
        print(f"   Feature Columns: {len(self.metadata['feature_columns'])}")
        
        return self
    
    def prepare_inference_data(self, df: pd.DataFrame, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """
        Prepare raw data for inference by applying same preprocessing as training
        
        This step is crucial because:
        1. Features must be in exact same order as training
        2. Data must be scaled using training statistics (prevent data leakage)
        3. Regime labels must be encoded consistently
        4. Sequences must match training sequence length
        
        Args:
            df: DataFrame with same features as training data
            start_date: Optional start date for inference period
            end_date: Optional end date for inference period
            
        Returns:
            X_sequences: Scaled input sequences (n_samples, seq_len, n_features)
            dates: Corresponding dates for each sequence
            y_true: True target values (if available) for evaluation
        """
        print(f"\n🔄 Preparing data for inference...")
        
        # Filter date range if specified
        if start_date or end_date:
            if 'Date' in df.columns:
                df = df.set_index('Date')
            df = df.loc[start_date:end_date]
            print(f"   Filtered to date range: {start_date} to {end_date}")
        
        # Ensure required features exist
        required_features = self.metadata['feature_columns']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features in training order (CRITICAL!)
        feature_df = df[required_features].copy()
        print(f"   Using {len(required_features)} features in training order")
        
        # Handle regime encoding if present
        if 'regime_label' in feature_df.columns:
            # Map text labels to integers using training encoder
            feature_df['regime_label'] = self.regime_encoder.transform(
                feature_df['regime_label']
            )
            print(f"   Encoded regime labels: {self.regime_encoder.classes_}")
        
        # Remove NaN values (essential for sequence creation)
        initial_len = len(feature_df)
        feature_df = feature_df.dropna()
        print(f"   Removed {initial_len - len(feature_df)} rows with NaN values")
        
        # Scale features using training statistics (NO REFIT!)
        features_scaled = self.scalers['features'].transform(feature_df.values)
        print(f"   Applied feature scaling (mean={self.scalers['features'].mean_[0]:.4f})")
        
        # Extract true values if target column exists
        y_true = None
        if 'log_return' in df.columns:
            target_data = df['log_return'].reindex(feature_df.index).dropna()
            if len(target_data) > 0:
                y_true = target_data.values
                print(f"   Found {len(y_true)} true target values for evaluation")
        
        # Create sequences for LSTM input
        sequence_length = self.metadata['sequence_length']
        X_sequences = []
        sequence_dates = []
        y_sequence = []
        
        for i in range(sequence_length, len(features_scaled)):
            # Each sequence: [t-seq_len:t] -> predict t+1
            sequence = features_scaled[i-sequence_length:i]
            X_sequences.append(sequence)
            sequence_dates.append(feature_df.index[i])
            
            # Add corresponding target if available
            if y_true is not None and i < len(y_true):
                y_sequence.append(y_true[i])
        
        X_sequences = np.array(X_sequences)
        sequence_dates = pd.DatetimeIndex(sequence_dates)
        y_sequence = np.array(y_sequence) if y_sequence else None
        
        print(f"✅ Created {len(X_sequences)} sequences")
        print(f"   Input shape: {X_sequences.shape}")
        print(f"   Date range: {sequence_dates[0]} to {sequence_dates[-1]}")
        
        return X_sequences, sequence_dates, y_sequence
    
    def monte_carlo_dropout_inference(self, X: np.ndarray, 
                                    n_samples: Optional[int] = None,
                                    batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo Dropout inference for uncertainty quantification
        
        Why Monte Carlo Dropout?
        1. Traditional neural networks give point estimates without uncertainty
        2. MC Dropout approximates Bayesian inference by keeping dropout active
        3. Multiple forward passes with different dropout masks = posterior sampling
        4. Variance across samples = model uncertainty (epistemic uncertainty)
        5. Essential for financial applications where uncertainty matters!
        
        The Process:
        1. Keep dropout layers active during inference (training=True)
        2. Run multiple forward passes with different random dropout masks  
        3. Each pass gives slightly different prediction due to randomness
        4. Aggregate predictions to get mean (best estimate) and std (uncertainty)
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            n_samples: Number of MC samples (more = better uncertainty estimates)
            batch_size: Process in batches to manage memory
            
        Returns:
            Dictionary containing:
            - mean: Average prediction across all MC samples
            - std: Standard deviation (uncertainty measure)
            - lower_ci/upper_ci: Confidence interval bounds
            - raw_samples: All individual MC predictions
        """
        if n_samples is None:
            n_samples = self.metadata['monte_carlo_samples']
        
        print(f"\n🎲 Running Monte Carlo Dropout Inference...")
        print(f"   MC Samples: {n_samples}")
        print(f"   Input sequences: {X.shape[0]}")
        print(f"   Batch size: {batch_size}")
        
        # Store all MC predictions
        mc_predictions = []
        n_batches = int(np.ceil(len(X) / batch_size))
        
        print(f"   Processing {n_batches} batches...")
        
        # Run multiple forward passes with dropout enabled
        for sample_idx in range(n_samples):
            if sample_idx % 20 == 0:
                print(f"   Progress: {sample_idx}/{n_samples} samples")
            
            batch_predictions = []
            
            # Process in batches to manage memory
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                X_batch = X[start_idx:end_idx]
                
                # 🔥 KEY: training=True keeps dropout active!
                # This is what makes it "Bayesian" - we sample from posterior
                y_pred_batch = self.model(X_batch, training=True)
                batch_predictions.append(y_pred_batch.numpy())
            
            # Combine batch predictions
            sample_predictions = np.concatenate(batch_predictions, axis=0).flatten()
            mc_predictions.append(sample_predictions)
        
        mc_predictions = np.array(mc_predictions)  # Shape: (n_samples, n_sequences)
        
        print(f"✅ MC Dropout inference complete!")
        print(f"   Prediction tensor shape: {mc_predictions.shape}")
        
        # Calculate aggregate statistics
        mean_pred = np.mean(mc_predictions, axis=0)
        std_pred = np.std(mc_predictions, axis=0)
        
        # Calculate confidence intervals
        lower_ci_90 = np.percentile(mc_predictions, 5, axis=0)
        upper_ci_90 = np.percentile(mc_predictions, 95, axis=0)
        lower_ci_95 = np.percentile(mc_predictions, 2.5, axis=0)
        upper_ci_95 = np.percentile(mc_predictions, 97.5, axis=0)
        
        # Inverse transform to original scale
        # This is critical - model predicts scaled values!
        mean_original = self.scalers['target'].inverse_transform(
            mean_pred.reshape(-1, 1)
        ).flatten()
        
        std_original = std_pred * self.scalers['target'].scale_[0]
        
        lower_ci_90_orig = self.scalers['target'].inverse_transform(
            lower_ci_90.reshape(-1, 1)
        ).flatten()
        
        upper_ci_90_orig = self.scalers['target'].inverse_transform(
            upper_ci_90.reshape(-1, 1)
        ).flatten()
        
        lower_ci_95_orig = self.scalers['target'].inverse_transform(
            lower_ci_95.reshape(-1, 1)
        ).flatten()
        
        upper_ci_95_orig = self.scalers['target'].inverse_transform(
            upper_ci_95.reshape(-1, 1)
        ).flatten()
        
        print(f"   Mean prediction range: [{mean_original.min():.6f}, {mean_original.max():.6f}]")
        print(f"   Uncertainty range: [{std_original.min():.6f}, {std_original.max():.6f}]")
        
        return {
            'mean': mean_original,
            'std': std_original,
            'lower_ci_90': lower_ci_90_orig,
            'upper_ci_90': upper_ci_90_orig,
            'lower_ci_95': lower_ci_95_orig,
            'upper_ci_95': upper_ci_95_orig,
            'raw_samples': mc_predictions,
            'n_samples': n_samples
        }
    
    def evaluate_predictions(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                           dates: pd.DatetimeIndex, df: pd.DataFrame) -> Dict:
        """
        Comprehensive evaluation of Bayesian LSTM predictions
        
        Why these metrics?
        
        Point Prediction Metrics:
        - MAE: Mean Absolute Error - average magnitude of errors (interpretable)
        - RMSE: Root Mean Square Error - penalizes large errors more than MAE
        - MAPE: Mean Absolute Percentage - relative error (good for comparing assets)
        - R²: Coefficient of determination - fraction of variance explained
        
        Uncertainty Quantification Metrics:
        - Coverage: % of true values within confidence intervals
        - Interval Width: Average width of confidence intervals
        - Calibration: How well predicted uncertainty matches actual uncertainty
        - Sharpness: Preference for narrow intervals (good forecaster is sharp + calibrated)
        
        Financial Metrics:
        - Hit Rate: % of correct directional predictions (up/down)
        - Sharpe Ratio: Risk-adjusted returns if using predictions for trading
        
        Args:
            y_true: True target values
            predictions: Dictionary from monte_carlo_dropout_inference
            dates: Dates corresponding to predictions
            df: Original dataframe with regime information
            
        Returns:
            Comprehensive evaluation metrics dictionary
        """
        print(f"\n📊 Evaluating prediction performance...")
        
        y_pred = predictions['mean']
        y_std = predictions['std']
        lower_90 = predictions['lower_ci_90']
        upper_90 = predictions['upper_ci_90']
        lower_95 = predictions['lower_ci_95']
        upper_95 = predictions['upper_ci_95']
        
        print(f"   Evaluating {len(y_true)} predictions")
        
        # =================================================================
        # POINT PREDICTION METRICS
        # =================================================================
        
        # Mean Absolute Error - most interpretable for financial returns
        mae = mean_absolute_error(y_true, y_pred)
        
        # Root Mean Square Error - penalizes large errors
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error - scale-independent
        # Handle division by zero for returns near zero
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # R-squared - fraction of variance explained
        r2 = r2_score(y_true, y_pred)
        
        # Maximum error
        max_error = np.max(np.abs(y_true - y_pred))
        
        print(f"✅ Point prediction metrics calculated")
        print(f"   MAE: {mae:.6f} ({mae*100:.3f}%)")
        print(f"   RMSE: {rmse:.6f} ({rmse*100:.3f}%)")
        print(f"   R²: {r2:.4f}")
        
        # =================================================================
        # UNCERTAINTY QUANTIFICATION METRICS
        # =================================================================
        
        # Coverage: % of true values within confidence intervals
        coverage_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
        coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))
        
        # Interval width: average width of confidence intervals
        interval_width_90 = np.mean(upper_90 - lower_90)
        interval_width_95 = np.mean(upper_95 - lower_95)
        
        # Calibration error: deviation from expected coverage
        calibration_error_90 = np.abs(coverage_90 - 0.90)
        calibration_error_95 = np.abs(coverage_95 - 0.95)
        
        # Sharpness: preference for narrow intervals (conditional on calibration)
        sharpness_90 = interval_width_90
        sharpness_95 = interval_width_95
        
        # Prediction Interval Coverage Probability (PICP)
        picp_90 = coverage_90
        picp_95 = coverage_95
        
        # Mean Prediction Interval Width (MPIW) - normalized
        mpiw_90 = interval_width_90 / (np.max(y_true) - np.min(y_true))
        mpiw_95 = interval_width_95 / (np.max(y_true) - np.min(y_true))
        
        print(f"✅ Uncertainty metrics calculated")
        print(f"   90% Coverage: {coverage_90:.3f} (target: 0.900)")
        print(f"   95% Coverage: {coverage_95:.3f} (target: 0.950)")
        print(f"   Calibration Error (95%): {calibration_error_95:.3f}")
        
        # =================================================================
        # FINANCIAL METRICS
        # =================================================================
        
        # Hit rate: directional accuracy
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # Correlation between predictions and actuals
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Information Coefficient (IC) - rank correlation
        ic = stats.spearmanr(y_true, y_pred)[0]
        
        print(f"✅ Financial metrics calculated")
        print(f"   Hit Rate: {hit_rate:.3f}")
        print(f"   Correlation: {correlation:.4f}")
        print(f"   Information Coefficient: {ic:.4f}")
        
        # =================================================================
        # REGIME-SPECIFIC ANALYSIS
        # =================================================================
        
        # Align regimes with prediction dates
        regime_analysis = {}
        if 'regime_label' in df.columns:
            df_indexed = df.set_index('Date') if 'Date' in df.columns else df
            aligned_regimes = df_indexed['regime_label'].reindex(dates)
            
            for regime in ['Crisis', 'Normal', 'Bull']:
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    regime_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                    regime_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                    regime_r2 = r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 1 else np.nan
                    regime_coverage_95 = np.mean(
                        (y_true[mask] >= lower_95[mask]) & 
                        (y_true[mask] <= upper_95[mask])
                    )
                    regime_hit_rate = np.mean(
                        np.sign(y_true[mask]) == np.sign(y_pred[mask])
                    )
                    
                    regime_analysis[regime] = {
                        'count': mask.sum(),
                        'mae': regime_mae,
                        'rmse': regime_rmse,
                        'r2': regime_r2,
                        'coverage_95': regime_coverage_95,
                        'hit_rate': regime_hit_rate,
                        'avg_uncertainty': np.mean(y_std[mask])
                    }
            
            print(f"✅ Regime-specific analysis completed")
            for regime, metrics in regime_analysis.items():
                print(f"   {regime}: {metrics['count']} samples, MAE={metrics['mae']:.4f}")
        
        # =================================================================
        # COMPILE RESULTS
        # =================================================================
        
        evaluation_results = {
            # Point prediction metrics
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'max_error': max_error,
            
            # Uncertainty metrics
            'coverage_90': coverage_90,
            'coverage_95': coverage_95,
            'interval_width_90': interval_width_90,
            'interval_width_95': interval_width_95,
            'calibration_error_90': calibration_error_90,
            'calibration_error_95': calibration_error_95,
            'picp_90': picp_90,
            'picp_95': picp_95,
            'mpiw_90': mpiw_90,
            'mpiw_95': mpiw_95,
            
            # Financial metrics
            'hit_rate': hit_rate,
            'correlation': correlation,
            'information_coefficient': ic,
            
            # Meta information
            'n_predictions': len(y_true),
            'evaluation_date': datetime.now().isoformat(),
            'model_config': self.metadata
        }
        
        # Store results for later use
        self.evaluation_metrics = evaluation_results
        self.regime_analysis = regime_analysis
        
        return evaluation_results
    
    def create_comprehensive_visualizations(self, y_true: np.ndarray, 
                                          predictions: Dict[str, np.ndarray],
                                          dates: pd.DatetimeIndex,
                                          df: pd.DataFrame,
                                          save_path: str = "results/inference_plots"):
        """
        Create comprehensive visualizations for Bayesian LSTM inference results
        
        Why these visualizations?
        1. Time series plot: Shows predictions vs actuals over time
        2. Uncertainty bands: Visualizes model confidence 
        3. Regime coloring: Shows how performance varies by market regime
        4. Error analysis: Identifies patterns in prediction errors
        5. Uncertainty analysis: Understanding when model is uncertain
        6. Calibration plots: Evaluate uncertainty quality
        
        Args:
            y_true: True target values
            predictions: Predictions from monte_carlo_dropout_inference
            dates: Dates for each prediction
            df: Original dataframe with regime info
            save_path: Directory to save plots
        """
        print(f"\n📈 Creating comprehensive visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        y_pred = predictions['mean']
        y_std = predictions['std']
        lower_95 = predictions['lower_ci_95']
        upper_95 = predictions['upper_ci_95']
        lower_90 = predictions['lower_ci_90']
        upper_90 = predictions['upper_ci_90']
        
        # =================================================================
        # PLOT 1: TIME SERIES WITH UNCERTAINTY BANDS
        # =================================================================
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Main time series
        ax.plot(dates, y_true, label='Actual Returns', color='black', linewidth=2, alpha=0.8)
        ax.plot(dates, y_pred, label='Predicted Returns', color='blue', linewidth=2)
        
        # Uncertainty bands
        ax.fill_between(dates, lower_95, upper_95, alpha=0.2, color='blue', 
                       label='95% Confidence Interval')
        ax.fill_between(dates, lower_90, upper_90, alpha=0.3, color='blue',
                       label='90% Confidence Interval')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Log Returns', fontsize=12)
        ax.set_title('Bayesian LSTM: Predictions with Uncertainty Quantification', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        mae = self.evaluation_metrics['mae']
        r2 = self.evaluation_metrics['r2']
        coverage = self.evaluation_metrics['coverage_95']
        
        textstr = f'MAE: {mae:.4f}\\nR²: {r2:.3f}\\n95% Coverage: {coverage:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/time_series_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # =================================================================
        # PLOT 2: REGIME-COLORED PREDICTIONS
        # =================================================================
        
        if 'regime_label' in df.columns:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Get regime colors
            df_indexed = df.set_index('Date') if 'Date' in df.columns else df
            aligned_regimes = df_indexed['regime_label'].reindex(dates)
            
            regime_colors = {'Crisis': 'red', 'Normal': 'green', 'Bull': 'blue'}
            
            for regime, color in regime_colors.items():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    ax.scatter(dates[mask], y_true[mask], 
                             label=f'{regime} (Actual)', color=color, alpha=0.6, s=20)
                    ax.scatter(dates[mask], y_pred[mask],
                             label=f'{regime} (Predicted)', color=color, 
                             alpha=0.8, s=20, marker='x')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Log Returns', fontsize=12)
            ax.set_title('Predictions by Market Regime', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/predictions_by_regime.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # =================================================================
        # PLOT 3: PREDICTION vs ACTUAL SCATTER
        # =================================================================
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2,
                label='Perfect Prediction')
        
        # Scatter plot with color-coded regimes
        if 'regime_label' in df.columns:
            for regime, color in regime_colors.items():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    ax.scatter(y_true[mask], y_pred[mask], 
                             label=f'{regime}', color=color, alpha=0.6, s=30)
        else:
            ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        ax.set_xlabel('Actual Returns', fontsize=12)
        ax.set_ylabel('Predicted Returns', fontsize=12)
        ax.set_title('Predicted vs Actual Returns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² to plot
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/scatter_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # =================================================================
        # PLOT 4: UNCERTAINTY OVER TIME
        # =================================================================
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Top plot: Uncertainty over time
        ax1.plot(dates, y_std, color='red', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Prediction Uncertainty (Std)', fontsize=12)
        ax1.set_title('Model Uncertainty Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Color background by regime
        if 'regime_label' in df.columns:
            for regime, color in regime_colors.items():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    ax1.fill_between(dates[mask], 0, y_std.max(), 
                                   color=color, alpha=0.1, label=f'{regime} Periods')
        
        # Bottom plot: Absolute errors over time
        abs_errors = np.abs(y_true - y_pred)
        ax2.plot(dates, abs_errors, color='orange', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/uncertainty_and_errors_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # =================================================================
        # PLOT 5: ERROR DISTRIBUTION BY REGIME
        # =================================================================
        
        if 'regime_label' in df.columns:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Error distribution histogram
            for regime, color in regime_colors.items():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    errors = y_true[mask] - y_pred[mask]
                    ax1.hist(errors, bins=20, alpha=0.6, color=color, 
                           label=f'{regime} (n={mask.sum()})', density=True)
            
            ax1.set_xlabel('Prediction Error')
            ax1.set_ylabel('Density')
            ax1.set_title('Error Distribution by Regime')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Uncertainty distribution
            for regime, color in regime_colors.items():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    ax2.hist(y_std[mask], bins=20, alpha=0.6, color=color,
                           label=f'{regime}', density=True)
            
            ax2.set_xlabel('Prediction Uncertainty')
            ax2.set_ylabel('Density')
            ax2.set_title('Uncertainty Distribution by Regime')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Box plots of absolute errors
            regime_errors = []
            regime_labels = []
            for regime in regime_colors.keys():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    regime_errors.extend(abs_errors[mask])
                    regime_labels.extend([regime] * mask.sum())
            
            if regime_errors:
                import seaborn as sns
                error_df = pd.DataFrame({'Regime': regime_labels, 'Absolute_Error': regime_errors})
                sns.boxplot(data=error_df, x='Regime', y='Absolute_Error', ax=ax3)
                ax3.set_title('Absolute Errors by Regime')
                ax3.grid(True, alpha=0.3)
            
            # Coverage by regime
            regime_names = []
            coverage_values = []
            for regime in regime_colors.keys():
                mask = aligned_regimes == regime
                if mask.sum() > 0:
                    regime_coverage = np.mean(
                        (y_true[mask] >= lower_95[mask]) & (y_true[mask] <= upper_95[mask])
                    )
                    regime_names.append(regime)
                    coverage_values.append(regime_coverage)
            
            if regime_names:
                bars = ax4.bar(regime_names, coverage_values, 
                             color=[regime_colors[r] for r in regime_names], alpha=0.7)
                ax4.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)')
                ax4.set_ylabel('Coverage Rate')
                ax4.set_title('95% Confidence Interval Coverage by Regime')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, coverage_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/regime_analysis_detailed.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # =================================================================
        # PLOT 6: CALIBRATION PLOT
        # =================================================================
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate calibration curve
        confidence_levels = np.arange(0.1, 1.0, 0.1)
        empirical_coverage = []
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_bound = np.percentile(predictions['raw_samples'], 
                                      (alpha/2) * 100, axis=0)
            upper_bound = np.percentile(predictions['raw_samples'], 
                                      (1 - alpha/2) * 100, axis=0)
            
            # Convert to original scale
            lower_bound_orig = self.scalers['target'].inverse_transform(
                lower_bound.reshape(-1, 1)
            ).flatten()
            upper_bound_orig = self.scalers['target'].inverse_transform(
                upper_bound.reshape(-1, 1)
            ).flatten()
            
            coverage = np.mean((y_true >= lower_bound_orig) & (y_true <= upper_bound_orig))
            empirical_coverage.append(coverage)
        
        # Plot calibration
        ax.plot(confidence_levels, empirical_coverage, 'b-o', linewidth=2, markersize=6,
                label='Empirical Coverage')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        
        ax.set_xlabel('Theoretical Coverage', fontsize=12)
        ax.set_ylabel('Empirical Coverage', fontsize=12)
        ax.set_title('Calibration Plot: Predicted vs Actual Coverage', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ All visualizations saved to: {save_path}")
        
        return save_path
    
    def run_complete_inference(self, data_path: str, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             save_results: bool = True) -> Dict:
        """
        Run complete inference pipeline from start to finish
        
        This orchestrates the entire inference process:
        1. Load and prepare data
        2. Perform Monte Carlo Dropout inference  
        3. Evaluate predictions comprehensively
        4. Create visualizations
        5. Save results
        
        Args:
            data_path: Path to CSV file with same features as training
            start_date: Optional start date for inference period
            end_date: Optional end date for inference period
            save_results: Whether to save results to files
            
        Returns:
            Complete results dictionary
        """
        print(f"\n🚀 Running Complete Bayesian LSTM Inference Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        print(f"📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        print(f"   Data shape: {df.shape}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Step 2: Prepare inference data
        X, dates, y_true = self.prepare_inference_data(df, start_date, end_date)
        
        # Step 3: Run Monte Carlo Dropout inference
        predictions = self.monte_carlo_dropout_inference(X)
        
        # Step 4: Evaluate predictions
        if y_true is not None:
            evaluation_results = self.evaluate_predictions(y_true, predictions, dates, df)
        else:
            print("⚠ No target values available for evaluation")
            evaluation_results = None
        
        # Step 5: Create visualizations
        if y_true is not None:
            plot_path = self.create_comprehensive_visualizations(
                y_true, predictions, dates, df
            )
        else:
            plot_path = None
        
        # Step 6: Compile complete results
        complete_results = {
            'predictions': {
                'dates': dates.tolist(),
                'mean': predictions['mean'].tolist(),
                'std': predictions['std'].tolist(),
                'lower_ci_90': predictions['lower_ci_90'].tolist(),
                'upper_ci_90': predictions['upper_ci_90'].tolist(),
                'lower_ci_95': predictions['lower_ci_95'].tolist(),
                'upper_ci_95': predictions['upper_ci_95'].tolist(),
                'n_mc_samples': predictions['n_samples']
            },
            'evaluation': evaluation_results,
            'regime_analysis': self.regime_analysis,
            'metadata': {
                'data_path': data_path,
                'inference_date': datetime.now().isoformat(),
                'model_path': self.model_path,
                'start_date': start_date,
                'end_date': end_date,
                'n_predictions': len(predictions['mean'])
            }
        }
        
        # Step 7: Save results
        if save_results:
            results_path = f'{self.model_path}/inference_results.json'
            with open(results_path, 'w') as f:
                json.dump(complete_results, f, indent=4, default=str)
            print(f"✅ Complete results saved to: {results_path}")
            
            # Save predictions as CSV for easy analysis
            if y_true is not None:
                pred_df = pd.DataFrame({
                    'Date': dates,
                    'Actual': y_true,
                    'Predicted': predictions['mean'],
                    'Uncertainty': predictions['std'],
                    'Lower_CI_90': predictions['lower_ci_90'],
                    'Upper_CI_90': predictions['upper_ci_90'],
                    'Lower_CI_95': predictions['lower_ci_95'],
                    'Upper_CI_95': predictions['upper_ci_95'],
                    'Absolute_Error': np.abs(y_true - predictions['mean']),
                    'In_CI_95': ((y_true >= predictions['lower_ci_95']) & 
                               (y_true <= predictions['upper_ci_95']))
                })
                
                pred_csv_path = f'{self.model_path}/inference_predictions.csv'
                pred_df.to_csv(pred_csv_path, index=False)
                print(f"✅ Predictions CSV saved to: {pred_csv_path}")
        
        print(f"\n🎉 Inference pipeline completed successfully!")
        print("=" * 60)
        
        return complete_results


def save_model_artifacts(bayesian_lstm_model, save_path: str = "results"):
    """
    Save all model artifacts needed for inference
    
    This function should be called after training to ensure all components
    needed for inference are properly saved.
    """
    print(f"\n💾 Saving model artifacts to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save the trained model
    model_file = f"{save_path}/bayesian_lstm_model.h5"
    bayesian_lstm_model.model.save(model_file)
    print(f"✅ Saved model: {model_file}")
    
    # 2. Save scalers
    scalers_file = f"{save_path}/scalers.pkl"
    with open(scalers_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.scalers, f)
    print(f"✅ Saved scalers: {scalers_file}")
    
    # 3. Save regime encoder
    encoder_file = f"{save_path}/regime_encoder.pkl"
    with open(encoder_file, 'wb') as f:
        pickle.dump(bayesian_lstm_model.regime_encoder, f)
    print(f"✅ Saved regime encoder: {encoder_file}")
    
    # 4. Save metadata
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
    
    print(f"📦 All artifacts saved! Ready for inference.")


def main():
    """
    Main function demonstrating complete inference pipeline
    """
    print("🔮 Bayesian LSTM Inference Engine - Complete Demo")
    print("=" * 70)
    
    # Initialize inference engine
    inference_engine = BayesianLSTMInferenceEngine(model_path="results")
    
    try:
        # Load trained model and artifacts
        inference_engine.load_artifacts()
        
        # Run complete inference on test data
        data_path = "data/data_with_regimes.csv"
        
        # Option 1: Run inference on specific date range
        results = inference_engine.run_complete_inference(
            data_path=data_path,
            start_date="2023-01-01",  # Adjust based on your data
            end_date=None,  # Use all data from start_date onwards
            save_results=True
        )
        
        # Display summary
        if results['evaluation']:
            print(f"\n📊 FINAL RESULTS SUMMARY")
            print("=" * 40)
            eval_metrics = results['evaluation']
            print(f"MAE: {eval_metrics['mae']:.6f}")
            print(f"RMSE: {eval_metrics['rmse']:.6f}")
            print(f"R²: {eval_metrics['r2']:.4f}")
            print(f"95% Coverage: {eval_metrics['coverage_95']:.3f}")
            print(f"Hit Rate: {eval_metrics['hit_rate']:.3f}")
            
            # Regime breakdown
            if inference_engine.regime_analysis:
                print(f"\n📈 REGIME PERFORMANCE:")
                for regime, metrics in inference_engine.regime_analysis.items():
                    print(f"  {regime}: MAE={metrics['mae']:.4f}, "
                          f"Coverage={metrics['coverage_95']:.3f}, "
                          f"Samples={metrics['count']}")
        
        print(f"\n✅ Inference completed successfully!")
        print(f"Check the results/ folder for detailed outputs and visualizations.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure you have run training first and saved all artifacts.")
        print(f"Required files in results/ directory:")
        print(f"  - bayesian_lstm_model.h5")
        print(f"  - scalers.pkl") 
        print(f"  - regime_encoder.pkl")
        print(f"  - model_metadata.json")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if _name_ == "_main_":
    main()