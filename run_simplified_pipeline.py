"""
Simplified Bayesian LSTM Pipeline - TensorFlow Import Issue Workaround
====================================================================

This script runs the Bayesian LSTM pipeline while avoiding complex import issues.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

def main():
    print("🚀 SIMPLIFIED BAYESIAN LSTM PIPELINE")
    print("=" * 50)
    
    # Test TensorFlow availability
    print("\n1️⃣ Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__} loaded successfully")
        
        # Set memory growth to avoid GPU issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   🖥️ GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"   ⚠️ GPU setup issue: {e}")
        
    except ImportError as e:
        print(f"   ❌ TensorFlow import failed: {e}")
        return
    
    # Load configuration
    print("\n2️⃣ Loading configuration...")
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ Configuration loaded from config.yaml")
    else:
        print("   ⚠️ No config.yaml found, using defaults")
        config = {
            'data_path': 'data/data_with_regimes.csv',
            'output_dir': 'results',
            'data': {
                'feature_columns': 'auto',
                'target_column': 'log_return',
                'regime_column': 'regime_label',
                'test_size': 0.2
            },
            'model': {
                'sequence_length': 20,
                'lstm_units': 64,
                'dropout_rate': 0.3,
                'n_monte_carlo': 100
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
    
    # Load and examine data
    print("\n3️⃣ Loading and examining data...")
    try:
        data_path = config['data_path']
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"   ✅ Data loaded: {df.shape}")
            print(f"   📊 Columns: {list(df.columns)}")
            
            # Auto-detect features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = config['data'].get('exclude_columns', [])
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            print(f"   🎯 Auto-detected features: {len(feature_cols)}")
            print(f"   📋 Features: {feature_cols[:10]}...")  # Show first 10
            
        else:
            print(f"   ❌ Data file not found: {data_path}")
            return
            
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return
    
    # Create minimal Bayesian LSTM model inline
    print("\n4️⃣ Creating Bayesian LSTM model...")
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        sequence_length = config['model']['sequence_length']
        n_features = len(feature_cols)
        lstm_units = config['model']['lstm_units']
        dropout_rate = config['model']['dropout_rate']
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, 
                       input_shape=(sequence_length, n_features)),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(50, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   ✅ Model created with {n_features} features")
        print(f"   📊 Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return
    
    # Prepare data for training (simplified)
    print("\n5️⃣ Preparing training data...")
    try:
        # Get features and target
        X_data = df[feature_cols].values
        y_data = df[config['data']['target_column']].values
        
        # Simple train/test split
        split_idx = int(len(X_data) * (1 - config['data']['test_size']))
        
        # Create sequences
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_length):
                X_seq.append(X[i:i+seq_length])
                y_seq.append(y[i+seq_length])
            return np.array(X_seq), np.array(y_seq)
        
        X_train, y_train = create_sequences(X_data[:split_idx], y_data[:split_idx], sequence_length)
        X_test, y_test = create_sequences(X_data[split_idx:], y_data[split_idx:], sequence_length)
        
        print(f"   ✅ Training data: {X_train.shape}, {y_train.shape}")
        print(f"   ✅ Test data: {X_test.shape}, {y_test.shape}")
        
    except Exception as e:
        print(f"   ❌ Data preparation failed: {e}")
        return
    
    # Train model
    print("\n6️⃣ Training model...")
    try:
        # Create results directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                f"{config['output_dir']}/best_model.keras",
                save_best_only=True
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✅ Training completed!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return
    
    # Make predictions with Monte Carlo Dropout
    print("\n7️⃣ Making Monte Carlo predictions...")
    try:
        n_monte_carlo = config['model']['n_monte_carlo']
        
        # Enable dropout during inference for MC Dropout
        mc_predictions = []
        for _ in range(n_monte_carlo):
            # Make prediction with dropout enabled
            pred = model(X_test, training=True)
            mc_predictions.append(pred.numpy())
        
        mc_predictions = np.array(mc_predictions)
        
        # Calculate mean and uncertainty
        pred_mean = np.mean(mc_predictions, axis=0)
        pred_std = np.std(mc_predictions, axis=0)
        
        print(f"   ✅ Monte Carlo predictions completed ({n_monte_carlo} samples)")
        print(f"   📊 Prediction shape: {pred_mean.shape}")
        print(f"   📊 Uncertainty range: {pred_std.min():.4f} - {pred_std.max():.4f}")
        
    except Exception as e:
        print(f"   ❌ Monte Carlo prediction failed: {e}")
        return
    
    # Calculate evaluation metrics
    print("\n8️⃣ Calculating evaluation metrics...")
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(y_test, pred_mean)
        mae = mean_absolute_error(y_test, pred_mean)
        rmse = np.sqrt(mse)
        
        print(f"   ✅ MSE: {mse:.6f}")
        print(f"   ✅ MAE: {mae:.6f}")
        print(f"   ✅ RMSE: {rmse:.6f}")
        
        # Save results
        results = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_monte_carlo': n_monte_carlo,
            'model_config': config['model'],
            'training_config': config['training']
        }
        
        with open(f"{config['output_dir']}/results.json", 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'actual': y_test.flatten(),
            'predicted': pred_mean.flatten(),
            'uncertainty': pred_std.flatten()
        })
        pred_df.to_csv(f"{config['output_dir']}/predictions.csv", index=False)
        
        print(f"   ✅ Results saved to {config['output_dir']}/")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📂 Results saved in: {config['output_dir']}/")
    print("📋 Files created:")
    print("   - best_model.keras (trained model)")
    print("   - results.json (evaluation metrics)")
    print("   - predictions.csv (predictions with uncertainty)")

if __name__ == "__main__":
    main()