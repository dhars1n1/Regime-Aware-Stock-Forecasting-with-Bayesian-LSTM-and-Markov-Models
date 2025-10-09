import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    'csv_file': 'data_final_with_regime.csv',
    'target_column': 'log_return', 
    'date_column': 'Date',  
    'exclude_columns': [], 
    # Data split ratios
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Sequence parameters
    'lookback_window': 30,  # Number of time steps to look back
    'forecast_horizon': 1,  # Number of steps ahead to predict
    
    # Model parameters
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dropout_rate': 0.4,
    'recurrent_dropout_rate': 0.4,
    
    # Training parameters
    'epochs': 40,
    'batch_size': 32,
    'learning_rate': 0.001,
    'patience': 10,
    
    # MC Dropout parameters
    'mc_samples': 100,
    
    # Output
    'save_model': True,
    'model_path': 'models/bayesian_lstm_model.h5'
}

# ========================================
# 1. Load and Prepare Data
# ========================================
def load_and_prepare_data(config):
    """
    Load CSV data and prepare features/target
    """
    print("📂 Loading data from CSV...")
    df = pd.read_csv(config['csv_file'])
    
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle date column
    if config['date_column'] and config['date_column'] in df.columns:
        df[config['date_column']] = pd.to_datetime(df[config['date_column']])
        df = df.sort_values(config['date_column'])
        dates = df[config['date_column']].values
        df = df.drop(columns=[config['date_column']])
    else:
        dates = None
    
    # Check for target column
    if config['target_column'] not in df.columns:
        raise ValueError(f"Target column '{config['target_column']}' not found in CSV!")
    
    # Separate target and features
    y = df[config['target_column']].values
    
    # Remove target and excluded columns from features
    exclude_cols = [config['target_column']] + config['exclude_columns']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    feature_names = X.columns.tolist()
    X = X.values
    
    print(f"\n📊 Data Summary:")
    print(f"   Features: {len(feature_names)} columns")
    print(f"   Feature names: {feature_names}")
    print(f"   Target: {config['target_column']}")
    print(f"   Total samples: {len(X)}")
    print(f"\n   Missing values in features: {np.isnan(X).sum()}")
    print(f"   Missing values in target: {np.isnan(y).sum()}")
    
    # Handle missing values
    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        print("\n⚠️  Missing values detected. Filling with forward fill then backward fill...")
        df_temp = pd.DataFrame(X, columns=feature_names)
        df_temp[config['target_column']] = y
        df_temp = df_temp.fillna(method='ffill').fillna(method='bfill')
        X = df_temp[feature_names].values
        y = df_temp[config['target_column']].values
    
    return X, y, feature_names, dates

# ========================================
# 2. Create Sequences for Time Series
# ========================================
def create_sequences(X, y, lookback, forecast_horizon=1):
    """
    Create sequences for LSTM training
    
    Args:
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        lookback: Number of time steps to look back
        forecast_horizon: Number of steps ahead to predict
    
    Returns:
        X_seq: Sequences (n_sequences, lookback, n_features)
        y_seq: Targets (n_sequences,)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - lookback - forecast_horizon + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback + forecast_horizon - 1])
    
    return np.array(X_seq), np.array(y_seq)

# ========================================
# 3. Split Data (Train/Val/Test)
# ========================================
def split_data(X, y, config):
    """
    Split data into train, validation, and test sets
    Time series: chronological split (no shuffling)
    """
    n = len(X)
    train_size = int(n * config['train_ratio'])
    val_size = int(n * config['val_ratio'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} samples ({config['train_ratio']*100:.0f}%)")
    print(f"   Val:   {len(X_val)} samples ({config['val_ratio']*100:.0f}%)")
    print(f"   Test:  {len(X_test)} samples ({config['test_ratio']*100:.0f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ========================================
# 4. Normalize Data
# ========================================
def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Normalize features and target using StandardScaler
    Fit only on training data to prevent data leakage
    """
    # Reshape for scaling
    n_train, lookback, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    # Scale features
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train.reshape(-1, n_features))
    X_val_scaled = X_scaler.transform(X_val.reshape(-1, n_features))
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, n_features))
    
    X_train_scaled = X_train_scaled.reshape(n_train, lookback, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, lookback, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, lookback, n_features)
    
    # Scale target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"\n✅ Data normalized")
    print(f"   Feature scaler: mean={X_scaler.mean_[:3]}, std={X_scaler.scale_[:3]}")
    print(f"   Target scaler: mean={y_scaler.mean_[0]:.4f}, std={y_scaler.scale_[0]:.4f}")
    
    return (X_train_scaled, y_train_scaled), \
           (X_val_scaled, y_val_scaled), \
           (X_test_scaled, y_test_scaled), \
           X_scaler, y_scaler

# ========================================
# 5. Build Bayesian LSTM Model
# ========================================
def build_model(input_shape, config):
    """
    Build Bayesian LSTM with MC Dropout
    """
    model = Sequential([
        LSTM(config['lstm_units_1'], 
             return_sequences=True, 
             input_shape=input_shape,
             recurrent_dropout=config['recurrent_dropout_rate']),
        Dropout(config['dropout_rate']),
        
        LSTM(config['lstm_units_2'], 
             return_sequences=False),
        Dropout(config['dropout_rate']),
        
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    print("\n🏗️  Model Architecture:")
    model.summary()
    
    return model

# ========================================
# 6. Monte Carlo Dropout Inference
# ========================================
def mc_dropout_predict(model, X, T=300):
    """
    Perform MC Dropout inference with T stochastic forward passes
    """
    preds = []
    for _ in range(T):
        y_pred = model(X, training=True)
        preds.append(y_pred.numpy())
    
    preds = np.array(preds).squeeze(axis=-1)
    
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)
    
    return mean_preds, std_preds, preds

# ========================================
# 7. Main Execution Pipeline
# ========================================
def main():
    # Load data
    X, y, feature_names, dates = load_and_prepare_data(CONFIG)
    
    # Create sequences
    print(f"\n🔄 Creating sequences with lookback={CONFIG['lookback_window']}...")
    X_seq, y_seq = create_sequences(X, y, CONFIG['lookback_window'], CONFIG['forecast_horizon'])
    print(f"   Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_seq, y_seq, CONFIG)
    
    # Normalize data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), X_scaler, y_scaler = \
        normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, CONFIG)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # MC Dropout prediction on test set
    print(f"\n🎲 Running MC Dropout inference with {CONFIG['mc_samples']} samples...")
    mean_preds_scaled, std_preds_scaled, all_preds_scaled = \
        mc_dropout_predict(model, X_test, T=CONFIG['mc_samples'])
    
    # Inverse transform predictions
    mean_preds = y_scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    std_preds = std_preds_scaled * y_scaler.scale_[0]
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Compute credible intervals (empirical percentiles)
    all_preds_original = y_scaler.inverse_transform(
        all_preds_scaled.T.reshape(-1, 1)
    ).reshape(all_preds_scaled.T.shape).T
    
    lower = np.percentile(all_preds_original, 2.5, axis=0)
    upper = np.percentile(all_preds_original, 97.5, axis=0)
    
    # Calculate metrics
    mse = np.mean((mean_preds - y_test_original) ** 2)
    mae = np.mean(np.abs(mean_preds - y_test_original))
    
    # Coverage: percentage of actuals within credible interval
    coverage = np.mean((y_test_original >= lower) & (y_test_original <= upper)) * 100
    
    print(f"\n📈 Test Set Performance:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   Mean Uncertainty (Std): {std_preds.mean():.6f}")
    print(f"   95% Credible Interval Coverage: {coverage:.2f}%")
    
    # Plot results
    plot_results(mean_preds, lower, upper, y_test_original, std_preds, history)
    
    # Export results
    export_results(mean_preds, lower, upper, std_preds, y_test_original)
    
    # Save model
    if CONFIG['save_model']:
        model.save(CONFIG['model_path'])
        print(f"\n💾 Model saved to {CONFIG['model_path']}")
    
    return model, X_scaler, y_scaler

# ========================================
# 8. Visualization
# ========================================
def plot_results(mean_preds, lower, upper, y_test, std_preds, history):
    """
    Create comprehensive plots
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Predictions with uncertainty bands
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(mean_preds, label='Predicted', color='blue', linewidth=2, alpha=0.8)
    ax1.plot(y_test, label='Actual', color='red', linewidth=1.5, alpha=0.7)
    ax1.fill_between(range(len(mean_preds)), lower, upper, 
                     color='blue', alpha=0.2, label='95% Credible Interval')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_title('Bayesian LSTM: Predictions with Uncertainty Bands', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Log Return')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: Prediction uncertainty over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(std_preds, color='orange', linewidth=2)
    ax2.fill_between(range(len(std_preds)), 0, std_preds, 
                     color='orange', alpha=0.3)
    ax2.set_title('Prediction Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 3: Prediction errors
    ax3 = fig.add_subplot(gs[1, 1])
    errors = mean_preds - y_test
    ax3.scatter(range(len(errors)), errors, alpha=0.5, s=10)
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_title('Prediction Errors', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error (Predicted - Actual)')
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 4: Training history
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax4.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax4.set_title('Training History', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (MSE)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 5: Prediction vs Actual scatter
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(y_test, mean_preds, alpha=0.5, s=10)
    min_val = min(y_test.min(), mean_preds.min())
    max_val = max(y_test.max(), mean_preds.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax5.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Actual Log Return')
    ax5.set_ylabel('Predicted Log Return')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.4)
    
    plt.savefig('bayesian_lstm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Plots saved to 'bayesian_lstm_analysis.png'")

# ========================================
# 9. Export Results
# ========================================
def export_results(mean_preds, lower, upper, std_preds, y_test):
    """
    Export predictions and uncertainty to CSV
    """
    df_results = pd.DataFrame({
        'time_step': np.arange(len(mean_preds)),
        'actual_log_return': y_test,
        'predicted_mean': mean_preds,
        'lower_bound_95': lower,
        'upper_bound_95': upper,
        'uncertainty_std': std_preds,
        'prediction_error': mean_preds - y_test,
        'within_ci': (y_test >= lower) & (y_test <= upper)
    })
    
    filename = 'bayesian_lstm_predictions.csv'
    df_results.to_csv(filename, index=False)
    
    print(f"\n✅ Results exported to '{filename}'")
    print("\n📋 Sample predictions:")
    print(df_results.head(10).to_string(index=False))
    
    return df_results

# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    model, X_scaler, y_scaler = main()