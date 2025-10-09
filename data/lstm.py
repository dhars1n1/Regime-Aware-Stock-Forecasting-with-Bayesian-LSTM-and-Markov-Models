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
# CONFIGURATION - AGGRESSIVE REGULARIZATION
# ========================================
CONFIG = {
    'csv_file': 'data_final_with_regime.csv',
    'target_column': 'log_return',
    'date_column': 'Date',
    'exclude_columns': [],

    # Data split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Sequence settings - REDUCED LOOKBACK
    'lookback_window': 20,          # Reduced from 60 to prevent memorization
    'forecast_horizon': 1,

    # Model parameters - SIGNIFICANTLY REDUCED CAPACITY
    'lstm_units_1': 32,             # Reduced from 96
    'lstm_units_2': 16,             # Reduced from 48 (or remove this layer)
    'dropout_rate': 0.5,            # Increased from 0.3
    'recurrent_dropout_rate': 0.4,  # Increased from 0.2
    'dense_units': 8,               # Reduced from 32

    # Training parameters
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0005,        # Reduced from 0.001 (slower learning)
    'patience': 20,                 # Increased patience

    # MC Dropout for uncertainty estimation
    'mc_samples': 100,

    # Output
    'save_model': True,
    'model_path': 'models/bayesian_lstm_regularized.h5'
}

# ========================================
# 1. Load and Prepare Data
# ========================================
def load_and_prepare_data(config):
    """Load CSV data and prepare features/target"""
    print("📂 Loading data from CSV...")
    df = pd.read_csv(config['csv_file'])
    
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    
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
    print(f"   Target: {config['target_column']}")
    print(f"   Total samples: {len(X)}")
    print(f"   Target std: {np.std(y):.6f} (baseline noise level)")
    
    # Handle missing values
    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        print("\n⚠️  Missing values detected. Filling...")
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
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - lookback - forecast_horizon + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback + forecast_horizon - 1])
    
    return np.array(X_seq), np.array(y_seq)

# ========================================
# 3. Split Data with Indices for Later Use
# ========================================
def split_data(X, y, config):
    """Split data into train, validation, and test sets"""
    n = len(X)
    train_size = int(n * config['train_ratio'])
    val_size = int(n * config['val_ratio'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Store indices for plotting
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, n)
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} samples ({config['train_ratio']*100:.0f}%)")
    print(f"   Val:   {len(X_val)} samples ({config['val_ratio']*100:.0f}%)")
    print(f"   Test:  {len(X_test)} samples ({config['test_ratio']*100:.0f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), \
           (train_indices, val_indices, test_indices)

# ========================================
# 4. Normalize Data
# ========================================
def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Normalize features and target using StandardScaler"""
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
    print(f"   Target scaler mean: {y_scaler.mean_[0]:.6f}, std: {y_scaler.scale_[0]:.6f}")
    
    return (X_train_scaled, y_train_scaled), \
           (X_val_scaled, y_val_scaled), \
           (X_test_scaled, y_test_scaled), \
           X_scaler, y_scaler

# ========================================
# 5. Build Regularized LSTM Model
# ========================================
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, config):
    """Build LSTM with aggressive regularization"""
    model = Sequential([
        LSTM(config['lstm_units_1'],
             return_sequences=True,
             input_shape=input_shape,
             dropout=config['dropout_rate'],
             recurrent_dropout=config['recurrent_dropout_rate'],
             kernel_regularizer=regularizers.l2(1e-3),
             recurrent_regularizer=regularizers.l2(1e-3)),
        
        LSTM(config['lstm_units_2'],
             return_sequences=False,
             dropout=config['dropout_rate'],
             recurrent_dropout=config['recurrent_dropout_rate'],
             kernel_regularizer=regularizers.l2(1e-3),
             recurrent_regularizer=regularizers.l2(1e-3)),
        
        Dense(config['dense_units'], 
              activation='relu',
              kernel_regularizer=regularizers.l2(1e-3)),
        Dropout(config['dropout_rate']),
        
        Dense(1)
    ])

    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print("\n🏗️ Regularized Model Architecture:")
    model.summary()
    return model

# ========================================
# 6. Monte Carlo Dropout Inference
# ========================================
def mc_dropout_predict(model, X, T=300):
    """Perform MC Dropout inference with T stochastic forward passes"""
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
    print(f"   Target distribution - mean: {np.mean(y_seq):.6f}, std: {np.std(y_seq):.6f}")
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
    (train_idx, val_idx, test_idx) = split_data(X_seq, y_seq, CONFIG)
    
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
        'best_model_regularized.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\n🚀 Training regularized model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Predictions on all splits
    print(f"\n🎲 Running MC Dropout inference...")
    
    # Training predictions
    mean_train_scaled, std_train_scaled, _ = mc_dropout_predict(model, X_train, T=CONFIG['mc_samples'])
    mean_train = y_scaler.inverse_transform(mean_train_scaled.reshape(-1, 1)).flatten()
    y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Validation predictions
    mean_val_scaled, std_val_scaled, _ = mc_dropout_predict(model, X_val, T=CONFIG['mc_samples'])
    mean_val = y_scaler.inverse_transform(mean_val_scaled.reshape(-1, 1)).flatten()
    y_val_original = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    # Test predictions
    mean_test_scaled, std_test_scaled, all_preds_scaled = \
        mc_dropout_predict(model, X_test, T=CONFIG['mc_samples'])
    mean_test = y_scaler.inverse_transform(mean_test_scaled.reshape(-1, 1)).flatten()
    std_test = std_test_scaled * y_scaler.scale_[0]
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Credible intervals
    all_preds_original = y_scaler.inverse_transform(
        all_preds_scaled.T.reshape(-1, 1)
    ).reshape(all_preds_scaled.T.shape).T
    
    lower = np.percentile(all_preds_original, 2.5, axis=0)
    upper = np.percentile(all_preds_original, 97.5, axis=0)
    
    # Calculate metrics
    mse_train = np.mean((mean_train - y_train_original) ** 2)
    mae_train = np.mean(np.abs(mean_train - y_train_original))
    
    mse_val = np.mean((mean_val - y_val_original) ** 2)
    mae_val = np.mean(np.abs(mean_val - y_val_original))
    
    mse_test = np.mean((mean_test - y_test_original) ** 2)
    mae_test = np.mean(np.abs(mean_test - y_test_original))
    
    coverage = np.mean((y_test_original >= lower) & (y_test_original <= upper)) * 100
    
    # Baseline: predict mean return
    baseline_mae = np.mean(np.abs(y_test_original - np.mean(y_train_original)))
    
    print(f"\n📈 Performance Metrics:")
    print(f"\n   Training:")
    print(f"      MSE: {mse_train:.6f}, MAE: {mae_train:.6f}")
    print(f"\n   Validation:")
    print(f"      MSE: {mse_val:.6f}, MAE: {mae_val:.6f}")
    print(f"\n   Test:")
    print(f"      MSE: {mse_test:.6f}, MAE: {mae_test:.6f}")
    print(f"      Mean Uncertainty (Std): {std_test.mean():.6f}")
    print(f"      95% Credible Interval Coverage: {coverage:.2f}%")
    print(f"\n   Baseline (predict mean): MAE = {baseline_mae:.6f}")
    print(f"   Model vs Baseline: {('BETTER ✓' if mae_test < baseline_mae else 'WORSE ✗')}")
    
    # Plot results
    plot_results_with_splits(
        (mean_train, y_train_original, train_idx),
        (mean_val, y_val_original, val_idx),
        (mean_test, y_test_original, test_idx, lower, upper),
        std_test, history
    )
    
    # Export results
    export_results(mean_test, lower, upper, std_test, y_test_original)
    
    # Save model
    if CONFIG['save_model']:
        model.save(CONFIG['model_path'])
        print(f"\n💾 Model saved to {CONFIG['model_path']}")
    
    return model, X_scaler, y_scaler

# ========================================
# 8. Visualization with Train/Val/Test Splits
# ========================================
def plot_results_with_splits(train_data, val_data, test_data, std_test, history):
    """Create comprehensive plots with different colors for splits"""
    mean_train, y_train, train_idx = train_data
    mean_val, y_val, val_idx = val_data
    mean_test, y_test, test_idx, lower, upper = test_data
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # ===== Plot 1: Full predictions with splits colored differently
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.plot(train_idx, mean_train, label='Train Predicted', 
             color='blue', linewidth=0.5, alpha=0.8)
    ax1.plot(val_idx, mean_val, label='Val Predicted', 
             color='green', linewidth=0.5, alpha=0.8)
    ax1.plot(test_idx, mean_test, label='Test Predicted', 
             color='orange', linewidth=0.5, alpha=0.8)
    
    ax1.plot(train_idx, y_train, label='Train Actual', 
             color='darkblue', linewidth=1, alpha=0.5, linestyle='--')
    ax1.plot(val_idx, y_val, label='Val Actual', 
             color='darkgreen', linewidth=1, alpha=0.5, linestyle='--')
    ax1.plot(test_idx, y_test, label='Test Actual', 
             color='red', linewidth=1, alpha=0.5, linestyle='--')
    
    ax1.axvline(train_idx[-1], color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax1.axvline(val_idx[-1], color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    
    ax1.set_title('Full Predictions: Train vs Val vs Test (Predicted vs Actual)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step (Global Index)')
    ax1.set_ylabel('Log Return')
    ax1.legend(loc='best', ncol=3, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
# ===== Plot 2: Test set only with credible intervals
    ax2 = fig.add_subplot(gs[1, :])

# Predicted line (blue)
    ax2.plot(test_idx, mean_test, label='Predicted', color='blue', linewidth=2, alpha=0.8)

# Actual line (red)
    ax2.plot(test_idx, y_test, label='Actual', color='red', linewidth=1.5, alpha=0.8)

# Credible interval (gray)
    ax2.fill_between(test_idx, lower, upper, 
                 color='gray', alpha=0.3, label='95% Credible Interval')

# Horizontal zero line
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

# Titles and labels
    ax2.set_title('Test Set: Predictions with Uncertainty Bands', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time Step (Global Index)')
    ax2.set_ylabel('Log Return')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.4)

    
    # ===== Plot 3: Training history
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax3.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='green')
    ax3.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    # ===== Plot 4: MAE by split
    ax4 = fig.add_subplot(gs[2, 1])
    mae_train = np.mean(np.abs(mean_train - y_train))
    mae_val = np.mean(np.abs(mean_val - y_val))
    mae_test = np.mean(np.abs(mean_test - y_test))
    
    splits = ['Train', 'Val', 'Test']
    maes = [mae_train, mae_val, mae_test]
    colors = ['blue', 'green', 'orange']
    
    bars = ax4.bar(splits, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_title('Mean Absolute Error by Split', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MAE')
    ax4.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.6f}', ha='center', va='bottom', fontsize=10)
    
    # ===== Plot 5: Test prediction errors
    ax5 = fig.add_subplot(gs[3, 0])
    errors_test = mean_test - y_test
    ax5.scatter(test_idx, errors_test, alpha=0.5, s=15, color='orange', edgecolors='darkorange')
    ax5.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax5.axhline(np.mean(errors_test), color='blue', linestyle='--', linewidth=1.5, label=f'Mean error: {np.mean(errors_test):.6f}')
    ax5.set_title('Test Prediction Errors', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Step (Global Index)')
    ax5.set_ylabel('Error (Predicted - Actual)')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.4)
    
    # ===== Plot 6: Prediction uncertainty over test set
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(test_idx, std_test, color='orange', linewidth=2, label='Std Dev')
    ax6.fill_between(test_idx, 0, std_test, color='orange', alpha=0.3)
    ax6.axhline(np.std(y_test), color='red', linestyle='--', linewidth=2, label=f'Actual return std: {np.std(y_test):.6f}')
    ax6.set_title('Prediction Uncertainty (Test Set)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time Step (Global Index)')
    ax6.set_ylabel('Standard Deviation')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.4)
    
    plt.savefig('bayesian_lstm_regularized.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Plots saved to 'bayesian_lstm_regularized.png'")

# ========================================
# 9. Export Results
# ========================================
def export_results(mean_preds, lower, upper, std_preds, y_test):
    """Export predictions and uncertainty to CSV"""
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
    
    filename = 'bayesian_lstm_predictions_regularized.csv'
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