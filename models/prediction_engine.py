"""
Prediction Engine for Regime-Aware Bayesian LSTM

This module provides:
1. One-step-ahead forecasting with uncertainty quantification
2. Multi-step forecasting capabilities  
3. Real-time prediction with regime awareness
4. Prediction confidence analysis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')


class BayesianLSTMPredictor:
    """
    Prediction engine for Bayesian LSTM with uncertainty quantification
    
    Features:
    - One-step-ahead forecasting
    - Monte Carlo uncertainty estimation
    - Regime-aware predictions
    - Confidence interval generation
    - Prediction visualization
    """
    
    def __init__(self, model_path: str, scalers: Dict, 
                 sequence_length: int = 20, 
                 monte_carlo_samples: int = 100):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved Keras model
            scalers: Dictionary with fitted scalers
            sequence_length: Input sequence length
            monte_carlo_samples: Number of MC samples for uncertainty
        """
        self.model_path = model_path
        self.scalers = scalers
        self.sequence_length = sequence_length
        self.monte_carlo_samples = monte_carlo_samples
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the trained Bayesian LSTM model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def predict_next_return(self, input_sequence: np.ndarray, 
                           return_distribution: bool = False) -> Dict:
        """
        Generate one-step-ahead return prediction with uncertainty
        
        Args:
            input_sequence: Input sequence (sequence_length, n_features)
            return_distribution: Whether to return full prediction distribution
            
        Returns:
            Dictionary with prediction statistics and uncertainty measures
        """
        if input_sequence.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {input_sequence.shape[0]}")
        
        # Reshape for model input (add batch dimension)
        X = input_sequence.reshape(1, self.sequence_length, -1)
        
        # Generate Monte Carlo predictions
        predictions = []
        for _ in range(self.monte_carlo_samples):
            # Enable dropout during inference for uncertainty estimation
            pred = self.model(X, training=True)
            predictions.append(pred.numpy().flatten()[0])
        
        predictions = np.array(predictions)
        
        # Calculate prediction statistics
        result = {
            'mean': np.mean(predictions),
            'median': np.median(predictions),
            'std': np.std(predictions),
            'var': np.var(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75),
            'iqr': np.percentile(predictions, 75) - np.percentile(predictions, 25),
            'ci_95_lower': np.percentile(predictions, 2.5),
            'ci_95_upper': np.percentile(predictions, 97.5),
            'ci_68_lower': np.percentile(predictions, 16),
            'ci_68_upper': np.percentile(predictions, 84)
        }
        
        # Add prediction distribution if requested
        if return_distribution:
            result['distribution'] = predictions
        
        # Transform back to original scale if scalers available
        if 'target' in self.scalers:
            scaled_values = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 
                           'ci_95_lower', 'ci_95_upper', 'ci_68_lower', 'ci_68_upper']
            
            for key in scaled_values:
                if key in result:
                    if key == 'std':
                        # For std, we need to handle scaling differently
                        result[f'{key}_original'] = result[key] * self.scalers['target'].scale_[0]
                    else:
                        result[f'{key}_original'] = self.scalers['target'].inverse_transform([[result[key]]])[0][0]
            
            if return_distribution:
                result['distribution_original'] = self.scalers['target'].inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()
        
        return result
    
    def predict_sequence(self, input_sequences: np.ndarray, 
                        return_raw: bool = False) -> Dict:
        """
        Generate predictions for multiple sequences
        
        Args:
            input_sequences: Input sequences (batch_size, sequence_length, n_features)
            return_raw: Whether to return raw MC samples
            
        Returns:
            Dictionary with batch prediction results
        """
        batch_size = input_sequences.shape[0]
        print(f"🔮 Generating predictions for {batch_size} sequences...")
        
        batch_predictions = []
        
        for i, sequence in enumerate(input_sequences):
            if i % 100 == 0:
                print(f"  Processing sequence {i+1}/{batch_size}")
            
            pred_result = self.predict_next_return(sequence, return_distribution=return_raw)
            batch_predictions.append(pred_result)
        
        # Aggregate results
        result = {
            'mean': np.array([p['mean'] for p in batch_predictions]),
            'std': np.array([p['std'] for p in batch_predictions]),
            'ci_95_lower': np.array([p['ci_95_lower'] for p in batch_predictions]),
            'ci_95_upper': np.array([p['ci_95_upper'] for p in batch_predictions]),
            'ci_68_lower': np.array([p['ci_68_lower'] for p in batch_predictions]),
            'ci_68_upper': np.array([p['ci_68_upper'] for p in batch_predictions])
        }
        
        # Add original scale if available
        if 'target' in self.scalers:
            result.update({
                'mean_original': np.array([p.get('mean_original', p['mean']) for p in batch_predictions]),
                'std_original': np.array([p.get('std_original', p['std']) for p in batch_predictions]),
                'ci_95_lower_original': np.array([p.get('ci_95_lower_original', p['ci_95_lower']) for p in batch_predictions]),
                'ci_95_upper_original': np.array([p.get('ci_95_upper_original', p['ci_95_upper']) for p in batch_predictions])
            })
        
        if return_raw:
            result['raw_predictions'] = [p.get('distribution', []) for p in batch_predictions]
        
        return result
    
    def multi_step_prediction(self, initial_sequence: np.ndarray, 
                             n_steps: int, regime_values: Optional[List] = None) -> Dict:
        """
        Generate multi-step-ahead predictions
        
        Args:
            initial_sequence: Initial sequence (sequence_length, n_features)
            n_steps: Number of steps to predict ahead
            regime_values: Regime values for future steps (if known)
            
        Returns:
            Dictionary with multi-step predictions
        """
        print(f"🔮 Generating {n_steps}-step-ahead predictions...")
        
        # Initialize with the input sequence
        current_sequence = initial_sequence.copy()
        predictions = []
        uncertainties = []
        
        for step in range(n_steps):
            print(f"  Step {step + 1}/{n_steps}")
            
            # Predict next value
            pred_result = self.predict_next_return(current_sequence)
            predictions.append(pred_result['mean'])
            uncertainties.append(pred_result['std'])
            
            # Update sequence for next prediction
            # This is a simplified approach - in practice, you'd need to update
            # other features (volume, regime info, etc.) appropriately
            new_row = current_sequence[-1].copy()  # Copy last row
            new_row[0] = pred_result['mean']  # Update return (assuming first feature is return)
            
            # Update regime if provided
            if regime_values and step < len(regime_values):
                # Assuming regime is the last feature (adjust index as needed)
                regime_feature_idx = -1  
                new_row[regime_feature_idx] = regime_values[step]
            
            # Roll the sequence forward
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
        
        return {
            'predictions': np.array(predictions),
            'uncertainties': np.array(uncertainties),
            'cumulative_return': np.cumsum(predictions),
            'cumulative_uncertainty': np.sqrt(np.cumsum(np.array(uncertainties)**2))
        }
    
    def analyze_prediction_confidence(self, predictions: np.ndarray, 
                                    actual_values: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze prediction confidence and calibration
        
        Args:
            predictions: Prediction results from predict_sequence
            actual_values: Actual values for calibration analysis
            
        Returns:
            Dictionary with confidence analysis
        """
        print("📊 Analyzing prediction confidence...")
        
        analysis = {
            'mean_uncertainty': np.mean(predictions['std']),
            'uncertainty_range': (np.min(predictions['std']), np.max(predictions['std'])),
            'high_uncertainty_ratio': np.mean(predictions['std'] > np.percentile(predictions['std'], 75)),
            'prediction_range': (np.min(predictions['mean']), np.max(predictions['mean']))
        }
        
        # Calibration analysis if actual values provided
        if actual_values is not None:
            # Check if actual values fall within confidence intervals
            in_95_ci = ((actual_values >= predictions['ci_95_lower']) & 
                       (actual_values <= predictions['ci_95_upper']))
            in_68_ci = ((actual_values >= predictions['ci_68_lower']) & 
                       (actual_values <= predictions['ci_68_upper']))
            
            analysis.update({
                'coverage_95': np.mean(in_95_ci),
                'coverage_68': np.mean(in_68_ci),
                'expected_coverage_95': 0.95,
                'expected_coverage_68': 0.68,
                'calibration_95': abs(np.mean(in_95_ci) - 0.95),
                'calibration_68': abs(np.mean(in_68_ci) - 0.68),
                'mean_absolute_error': np.mean(np.abs(actual_values - predictions['mean'])),
                'mean_squared_error': np.mean((actual_values - predictions['mean'])**2)
            })
        
        return analysis
    
    def visualize_predictions(self, dates: pd.DatetimeIndex, 
                            predictions: Dict, 
                            actual_values: Optional[np.ndarray] = None,
                            regime_labels: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize predictions with uncertainty bands
        
        Args:
            dates: Dates corresponding to predictions
            predictions: Prediction results
            actual_values: Actual values to compare against
            regime_labels: Market regime labels
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Main prediction plot
        ax1 = axes[0]
        
        # Plot predictions
        ax1.plot(dates, predictions['mean'], label='Predicted Returns', 
                color='blue', linewidth=2, alpha=0.8)
        
        # Plot actual values if available
        if actual_values is not None:
            ax1.plot(dates, actual_values, label='Actual Returns', 
                    color='black', linewidth=2, alpha=0.8)
        
        # Uncertainty bands
        ax1.fill_between(dates, 
                        predictions['ci_95_lower'], 
                        predictions['ci_95_upper'],
                        alpha=0.3, color='blue', label='95% Confidence Interval')
        ax1.fill_between(dates,
                        predictions['ci_68_lower'],
                        predictions['ci_68_upper'], 
                        alpha=0.5, color='blue', label='68% Confidence Interval')
        
        ax1.set_title('Bayesian LSTM Predictions with Uncertainty', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Log Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty over time
        ax2 = axes[1]
        ax2.plot(dates, predictions['std'], color='red', alpha=0.8, linewidth=2)
        ax2.fill_between(dates, 0, predictions['std'], alpha=0.3, color='red')
        
        # Color by regime if available
        if regime_labels is not None:
            regime_colors = {'Crisis': '#e74c3c', 'Normal': '#f39c12', 'Bull': '#27ae60'}
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                ax2.scatter(dates[mask], predictions['std'][mask], 
                           c=regime_colors.get(regime, 'gray'), 
                           alpha=0.7, s=10, label=f'{regime} Regime')
            ax2.legend()
        
        ax2.set_title('Prediction Uncertainty Over Time', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Prediction Standard Deviation')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Prediction plot saved to {save_path}")
        
        plt.show()
    
    def generate_prediction_report(self, dates: pd.DatetimeIndex,
                                 predictions: Dict,
                                 actual_values: Optional[np.ndarray] = None,
                                 save_dir: str = "prediction_results") -> Dict:
        """
        Generate comprehensive prediction report
        
        Args:
            dates: Prediction dates
            predictions: Prediction results
            actual_values: Actual values for evaluation
            save_dir: Directory to save results
            
        Returns:
            Dictionary with report metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("📋 Generating prediction report...")
        
        # Basic statistics
        report = {
            'prediction_summary': {
                'total_predictions': len(predictions['mean']),
                'date_range': (str(dates[0]), str(dates[-1])),
                'mean_prediction': float(np.mean(predictions['mean'])),
                'prediction_volatility': float(np.std(predictions['mean'])),
                'mean_uncertainty': float(np.mean(predictions['std'])),
                'max_uncertainty': float(np.max(predictions['std'])),
                'min_uncertainty': float(np.min(predictions['std']))
            }
        }
        
        # Performance metrics if actual values available
        if actual_values is not None:
            confidence_analysis = self.analyze_prediction_confidence(predictions, actual_values)
            report['performance'] = confidence_analysis
        
        # Create detailed predictions DataFrame
        pred_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Return': predictions['mean'],
            'Prediction_Std': predictions['std'],
            'CI_95_Lower': predictions['ci_95_lower'],
            'CI_95_Upper': predictions['ci_95_upper'],
            'CI_68_Lower': predictions['ci_68_lower'],
            'CI_68_Upper': predictions['ci_68_upper']
        })
        
        if actual_values is not None:
            pred_df['Actual_Return'] = actual_values
            pred_df['Absolute_Error'] = np.abs(actual_values - predictions['mean'])
            pred_df['In_95_CI'] = ((actual_values >= predictions['ci_95_lower']) & 
                                  (actual_values <= predictions['ci_95_upper']))
        
        # Save detailed results
        pred_df.to_csv(os.path.join(save_dir, 'detailed_predictions.csv'), index=False)
        
        # Save report
        import json
        report_path = os.path.join(save_dir, 'prediction_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Prediction report saved to {save_dir}")
        
        return report


def demo_prediction_pipeline():
    """
    Demonstration of the prediction pipeline
    (This would typically use a real trained model and data)
    """
    print("🧪 BAYESIAN LSTM PREDICTION DEMO")
    print("=" * 50)
    
    # Note: This is a demo - in practice you'd load real model and data
    print("⚠️  This is a demonstration. Replace with actual trained model and data.")
    
    # Example usage pattern:
    example_usage = """
    # Load trained model and scalers
    predictor = BayesianLSTMPredictor(
        model_path='results/bayesian_lstm_model.h5',
        scalers=saved_scalers,
        sequence_length=20,
        monte_carlo_samples=100
    )
    
    # Generate predictions for test data
    predictions = predictor.predict_sequence(X_test)
    
    # Analyze confidence
    analysis = predictor.analyze_prediction_confidence(predictions, y_actual)
    
    # Visualize results
    predictor.visualize_predictions(test_dates, predictions, y_actual, regimes)
    
    # Generate report
    report = predictor.generate_prediction_report(test_dates, predictions, y_actual)
    """
    
    print("Example usage:")
    print(example_usage)
    
    return example_usage


if __name__ == "__main__":
    demo_prediction_pipeline()