"""
Training Script for Regime-Aware Bayesian LSTM

This script handles:
1. Model training with proper callbacks
2. Training monitoring and visualization
3. Model checkpointing
4. Early stopping based on validation performance
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

from data_processor import RegimeAwareDataProcessor
from bayesian_lstm import BayesianLSTM


class BayesianLSTMTrainer:
    """
    Training manager for Bayesian LSTM with regime awareness
    
    Handles complete training pipeline including:
    - Model setup and compilation
    - Training with callbacks
    - Performance monitoring
    - Model checkpointing
    """
    
    def __init__(self, model: BayesianLSTM, save_dir: str = "results"):
        """
        Initialize trainer
        
        Args:
            model: BayesianLSTM instance
            save_dir: Directory to save training artifacts
        """
        self.model = model
        self.save_dir = save_dir
        self.history = None
        self.training_config = {}
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
    def setup_callbacks(self, patience_early: int = 15, patience_lr: int = 10,
                       monitor: str = 'val_loss') -> list:
        """
        Setup training callbacks for better training control
        
        Args:
            patience_early: Patience for early stopping
            patience_lr: Patience for learning rate reduction
            monitor: Metric to monitor for callbacks
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Model checkpointing
        checkpoint_path = os.path.join(self.save_dir, 'best_model.h5')
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(model_checkpoint)
        
        # Custom callback to log training progress
        training_logger = TrainingLogger(self.save_dir)
        callbacks.append(training_logger)
        
        print(f"✅ Setup {len(callbacks)} training callbacks")
        return callbacks
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 32,
                   validation_split: float = 0.0,
                   **kwargs) -> keras.callbacks.History:
        """
        Train the Bayesian LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences  
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation (if X_val not provided)
            **kwargs: Additional training parameters
            
        Returns:
            Training history object
        """
        print("🎯 Starting Bayesian LSTM Training")
        print("=" * 50)
        
        # Store training configuration
        self.training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else None,
            'features': X_train.shape[2],
            'sequence_length': X_train.shape[1],
            'model_params': {
                'lstm_units': self.model.lstm_units,
                'dropout_rate': self.model.dropout_rate,
                'monte_carlo_samples': self.model.monte_carlo_samples
            },
            'training_start': datetime.now().isoformat()
        }
        
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val) if X_val is not None else 'Using split'}")
        print(f"  Input shape: {X_train.shape}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
            print(f"  Using validation split: {validation_split}")
        
        # Train the model
        print(f"\n🚀 Training started...")
        
        try:
            self.history = self.model.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                **kwargs
            )
            
            self.training_config['training_end'] = datetime.now().isoformat()
            self.training_config['epochs_completed'] = len(self.history.history['loss'])
            
            print(f"\n✅ Training completed!")
            print(f"  Epochs completed: {self.training_config['epochs_completed']}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            self.training_config['training_end'] = datetime.now().isoformat()
            self.training_config['interrupted'] = True
        
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            self.training_config['training_end'] = datetime.now().isoformat()
            self.training_config['error'] = str(e)
            raise
        
        return self.history
    
    def plot_training_history(self) -> None:
        """Plot training and validation metrics"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        print("📊 Plotting training history...")
        
        history_dict = self.history.history
        
        # Setup the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bayesian LSTM Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(history_dict['loss'], label='Training Loss', alpha=0.8)
        if 'val_loss' in history_dict:
            ax.plot(history_dict['val_loss'], label='Validation Loss', alpha=0.8)
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE plot
        ax = axes[0, 1]
        ax.plot(history_dict['mae'], label='Training MAE', alpha=0.8)
        if 'val_mae' in history_dict:
            ax.plot(history_dict['val_mae'], label='Validation MAE', alpha=0.8)
        ax.set_title('Mean Absolute Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MSE plot
        ax = axes[1, 0]
        ax.plot(history_dict['mse'], label='Training MSE', alpha=0.8)
        if 'val_mse' in history_dict:
            ax.plot(history_dict['val_mse'], label='Validation MSE', alpha=0.8)
        ax.set_title('Mean Squared Error')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        ax = axes[1, 1]
        if 'lr' in history_dict:
            ax.plot(history_dict['lr'], label='Learning Rate', alpha=0.8)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Show epoch vs final loss if lr not available
            epochs = range(1, len(history_dict['loss']) + 1)
            ax.plot(epochs, history_dict['loss'], 'b-', alpha=0.8, label='Training Loss')
            if 'val_loss' in history_dict:
                ax.plot(epochs, history_dict['val_loss'], 'r-', alpha=0.8, label='Validation Loss')
            ax.set_title('Loss Over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training history saved to {plot_path}")
    
    def save_training_config(self) -> None:
        """Save training configuration and results"""
        config_path = os.path.join(self.save_dir, 'training_config.json')
        
        # Add final metrics if available
        if self.history is not None:
            history_dict = self.history.history
            self.training_config['final_metrics'] = {
                'final_loss': float(history_dict['loss'][-1]),
                'final_mae': float(history_dict['mae'][-1]),
                'final_mse': float(history_dict['mse'][-1])
            }
            
            if 'val_loss' in history_dict:
                self.training_config['final_metrics'].update({
                    'final_val_loss': float(history_dict['val_loss'][-1]),
                    'final_val_mae': float(history_dict['val_mae'][-1]),
                    'final_val_mse': float(history_dict['val_mse'][-1])
                })
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(self.training_config, f, indent=2)
        
        print(f"✅ Training configuration saved to {config_path}")
    
    def save_model_artifacts(self) -> None:
        """Save all model artifacts"""
        print("💾 Saving model artifacts...")
        
        # Save the full model
        model_path = os.path.join(self.save_dir, 'bayesian_lstm_model.h5')
        self.model.model.save(model_path)
        print(f"  Model saved to {model_path}")
        
        # Save training configuration
        self.save_training_config()
        
        # Save model summary
        summary_path = os.path.join(self.save_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            self.model.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"  Model summary saved to {summary_path}")
        
        print("✅ All artifacts saved successfully!")


class TrainingLogger(keras.callbacks.Callback):
    """Custom callback to log detailed training progress"""
    
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.epoch_logs = []
        
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Log metrics at the end of each epoch"""
        if logs is not None:
            epoch_info = {
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat(),
                **logs
            }
            self.epoch_logs.append(epoch_info)
            
            # Save epoch logs
            logs_path = os.path.join(self.save_dir, 'epoch_logs.json')
            with open(logs_path, 'w') as f:
                json.dump(self.epoch_logs, f, indent=2)


def run_training_pipeline(sequence_length: int = 20, 
                         lstm_units: int = 64,
                         dropout_rate: float = 0.3,
                         monte_carlo_samples: int = 100,
                         epochs: int = 100,
                         batch_size: int = 32,
                         test_size: float = 0.2,
                         save_dir: str = "results") -> Tuple[BayesianLSTM, Dict]:
    """
    Complete training pipeline for regime-aware Bayesian LSTM
    
    Args:
        sequence_length: LSTM sequence length
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for MC Dropout
        monte_carlo_samples: Number of MC samples for uncertainty
        epochs: Training epochs
        batch_size: Training batch size
        test_size: Test set fraction
        save_dir: Directory to save results
        
    Returns:
        Trained model and evaluation results
    """
    print("🚀 REGIME-AWARE BAYESIAN LSTM TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Data Preparation
    print("\n1️⃣ DATA PREPARATION")
    processor = RegimeAwareDataProcessor(
        sequence_length=sequence_length,
        use_regime_label=True
    )
    
    df = processor.load_data()
    X_train, y_train, X_test, y_test, test_dates = processor.prepare_data_for_training(
        df, test_size=test_size
    )
    
    # 2. Model Setup
    print("\n2️⃣ MODEL SETUP")
    model = BayesianLSTM(
        sequence_length=sequence_length,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        monte_carlo_samples=monte_carlo_samples,
        use_regime_label=True
    )
    
    # Set number of features and build model
    model.n_features = X_train.shape[2]
    model.build_model()
    
    print("Model Architecture:")
    model.model.summary()
    
    # 3. Training
    print("\n3️⃣ MODEL TRAINING")
    trainer = BayesianLSTMTrainer(model, save_dir)
    
    history = trainer.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 4. Training Visualization
    print("\n4️⃣ TRAINING ANALYSIS")
    trainer.plot_training_history()
    trainer.save_model_artifacts()
    
    # 5. Model Evaluation
    print("\n5️⃣ MODEL EVALUATION")
    evaluation_results = model.evaluate_regime_performance(
        X_test, y_test, test_dates, df
    )
    
    # Print results summary
    overall = evaluation_results['overall']
    print(f"\n📊 TRAINING RESULTS SUMMARY")
    print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    print(f"  Test MSE: {overall['mse']:.6f}")
    print(f"  Test MAE: {overall['mae']:.6f}")
    print(f"  Test R²: {overall['r2']:.4f}")
    print(f"  95% Coverage: {overall['coverage_95']:.3f}")
    
    # 6. Save Results
    print("\n6️⃣ SAVING RESULTS")
    model.create_comprehensive_visualizations(test_dates, evaluation_results, save_dir)
    model.save_predictions_data(test_dates, evaluation_results, save_dir)
    
    print(f"\n✅ Training pipeline completed successfully!")
    print(f"All results saved to: {save_dir}")
    print("=" * 60)
    
    return model, evaluation_results


if __name__ == "__main__":
    # Run the complete training pipeline
    model, results = run_training_pipeline(
        sequence_length=20,
        lstm_units=64, 
        dropout_rate=0.3,
        monte_carlo_samples=100,
        epochs=100,
        batch_size=32,
        test_size=0.2,
        save_dir="results"
    )