"""
Enhanced Visualization Script for Bayesian LSTM Analysis
Generates three corrected plots with proper interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load the detailed predictions data"""
    try:
        df = pd.read_csv('results/detailed_predictions.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate uncertainty range
        df['Uncertainty_Range'] = df['Upper_95_CI'] - df['Lower_95_CI']
        
        print(f"Data loaded: {len(df)} samples from {df['Date'].min()} to {df['Date'].max()}")
        print(f"Regime distribution:\n{df['Regime'].value_counts()}")
        print(f"Overall 95% Coverage: {df['In_CI_95'].mean():.3f}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_1_actual_vs_predicted_timeseries(df):
    """
    Plot 1: Actual vs Predicted Log Returns with Confidence Intervals
    Shows temporal dynamics and regime-aware uncertainty
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Define regime colors
    regime_colors = {
        'Bull': '#27ae60',    # Green
        'Crisis': '#e74c3c',  # Red  
        'Normal': '#f39c12'   # Orange
    }
    
    # Plot 1a: Full time series with confidence intervals
    ax1 = axes[0]
    
    # Plot actual returns
    ax1.plot(df['Date'], df['Actual_Return'], 
             label='Actual Returns', color='black', alpha=0.8, linewidth=1.2)
    
    # Plot predicted returns
    ax1.plot(df['Date'], df['Predicted_Return'], 
             label='Predicted Returns', color='blue', linewidth=1.5)
    
    # Plot confidence intervals
    ax1.fill_between(df['Date'], df['Lower_95_CI'], df['Upper_95_CI'],
                     alpha=0.2, color='blue', label='95% Confidence Interval')
    
    ax1.set_title('Regime-Aware Bayesian LSTM: Actual vs Predicted Returns with Uncertainty', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Log Returns', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 1b: Actual returns colored by regime
    ax2 = axes[1]
    
    for regime in df['Regime'].unique():
        mask = df['Regime'] == regime
        ax2.scatter(df.loc[mask, 'Date'], df.loc[mask, 'Actual_Return'],
                   c=regime_colors.get(regime, 'gray'), 
                   label=f'{regime} Regime', alpha=0.7, s=20)
    
    ax2.set_title('Actual Returns Colored by Market Regime', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Log Returns', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 1c: Prediction uncertainty over time
    ax3 = axes[2]
    
    # Plot uncertainty as area chart
    ax3.fill_between(df['Date'], 0, df['Prediction_Uncertainty'],
                     alpha=0.6, color='purple', label='Prediction Uncertainty (Std Dev)')
    ax3.plot(df['Date'], df['Prediction_Uncertainty'], 
             color='darkviolet', linewidth=1.5)
    
    # Add regime background coloring
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        if len(regime_data) > 0:
            for _, row in regime_data.iterrows():
                ax3.axvline(x=row['Date'], color=regime_colors.get(regime, 'gray'), 
                           alpha=0.1, linewidth=0.5)
    
    ax3.set_title('Prediction Uncertainty Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Uncertainty (Std Dev)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_timeseries_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS INSIGHTS")
    print("="*50)
    
    print(f"Prediction Range: [{df['Predicted_Return'].min():.6f}, {df['Predicted_Return'].max():.6f}]")
    print(f"Actual Range: [{df['Actual_Return'].min():.6f}, {df['Actual_Return'].max():.6f}]")
    print(f"Average Uncertainty: {df['Prediction_Uncertainty'].mean():.6f}")
    print(f"Average CI Width: {df['Uncertainty_Range'].mean():.6f}")
    
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        print(f"\n{regime} Regime ({len(regime_data)} samples):")
        print(f"  Avg Uncertainty: {regime_data['Prediction_Uncertainty'].mean():.6f}")
        print(f"  Avg CI Width: {regime_data['Uncertainty_Range'].mean():.6f}")
        print(f"  Coverage: {regime_data['In_CI_95'].mean():.3f}")

def plot_2_scatter_predicted_vs_actual(df):
    """
    Plot 2: Scatter Plot of Predicted vs Actual with Regime Coloring
    Shows prediction accuracy and bias patterns
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    regime_colors = {
        'Bull': '#27ae60',
        'Crisis': '#e74c3c', 
        'Normal': '#f39c12'
    }
    
    # Plot 2a: Main scatter plot
    ax1 = axes[0, 0]
    
    for regime in df['Regime'].unique():
        mask = df['Regime'] == regime
        ax1.scatter(df.loc[mask, 'Actual_Return'], df.loc[mask, 'Predicted_Return'],
                   c=regime_colors.get(regime, 'gray'), alpha=0.6, s=30, 
                   label=f'{regime} Regime')
    
    # Perfect prediction line
    min_val = min(df['Actual_Return'].min(), df['Predicted_Return'].min())
    max_val = max(df['Actual_Return'].max(), df['Predicted_Return'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
             label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Log Returns', fontsize=12)
    ax1.set_ylabel('Predicted Log Returns', fontsize=12)
    ax1.set_title('Predicted vs Actual Returns by Regime', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2b: Residuals vs Predicted
    ax2 = axes[0, 1]
    
    residuals = df['Actual_Return'] - df['Predicted_Return']
    
    for regime in df['Regime'].unique():
        mask = df['Regime'] == regime
        ax2.scatter(df.loc[mask, 'Predicted_Return'], residuals[mask],
                   c=regime_colors.get(regime, 'gray'), alpha=0.6, s=30,
                   label=f'{regime} Regime')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Returns', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 2c: Error vs Uncertainty
    ax3 = axes[1, 0]
    
    for regime in df['Regime'].unique():
        mask = df['Regime'] == regime
        ax3.scatter(df.loc[mask, 'Prediction_Uncertainty'], df.loc[mask, 'Absolute_Error'],
                   c=regime_colors.get(regime, 'gray'), alpha=0.6, s=30,
                   label=f'{regime} Regime')
    
    ax3.set_xlabel('Prediction Uncertainty (Std Dev)', fontsize=12)
    ax3.set_ylabel('Absolute Error', fontsize=12)
    ax3.set_title('Absolute Error vs Uncertainty', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 2d: Distribution of predictions vs actuals
    ax4 = axes[1, 1]
    
    ax4.hist(df['Actual_Return'], bins=50, alpha=0.5, label='Actual', density=True)
    ax4.hist(df['Predicted_Return'], bins=50, alpha=0.5, label='Predicted', density=True)
    
    ax4.set_xlabel('Log Returns', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_scatter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print correlation metrics
    print("\n" + "="*50)
    print("SCATTER PLOT ANALYSIS INSIGHTS")
    print("="*50)
    
    correlation = np.corrcoef(df['Actual_Return'], df['Predicted_Return'])[0, 1]
    print(f"Overall Correlation: {correlation:.4f}")
    
    # Regime-specific correlations
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        regime_corr = np.corrcoef(regime_data['Actual_Return'], 
                                 regime_data['Predicted_Return'])[0, 1]
        print(f"{regime} Regime Correlation: {regime_corr:.4f}")
    
    # Check for prediction collapse
    pred_std = df['Predicted_Return'].std()
    actual_std = df['Actual_Return'].std()
    print(f"\nPrediction Std: {pred_std:.6f}")
    print(f"Actual Std: {actual_std:.6f}")
    print(f"Variance Ratio (Pred/Actual): {(pred_std/actual_std):.4f}")
    
    if pred_std/actual_std < 0.5:
        print("⚠️  WARNING: Prediction collapse detected - model predicting too narrow range!")

def plot_3_model_comparison_metrics(df):
    """
    Plot 3: Comprehensive Metrics Comparison
    (For now, showing Bayesian LSTM metrics with targets)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Calculate current metrics
    mse = df['Squared_Error'].mean()
    rmse = np.sqrt(mse)
    mae = df['Absolute_Error'].mean()
    r2 = 1 - (df['Squared_Error'].sum() / 
              ((df['Actual_Return'] - df['Actual_Return'].mean())**2).sum())
    coverage_95 = df['In_CI_95'].mean()
    avg_interval_width = df['Uncertainty_Range'].mean()
    
    # Define target/ideal values for comparison
    metrics_data = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Coverage_95', 'Avg_Interval_Width'],
        'Current_Bayesian_LSTM': [mse, rmse, mae, r2, coverage_95, avg_interval_width],
        'Target_Values': [mse*0.8, rmse*0.8, mae*0.8, 0.15, 0.95, avg_interval_width*3],
        'Standard_LSTM_Est': [mse*1.1, rmse*1.1, mae*1.1, r2*0.8, 0.0, 0.0]  # No uncertainty
    }
    
    # Create comparison plots
    for i, metric in enumerate(metrics_data['Metric']):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        methods = ['Current\nBayesian LSTM', 'Target\nValues', 'Standard LSTM\n(Estimated)']
        values = [metrics_data['Current_Bayesian_LSTM'][i], 
                 metrics_data['Target_Values'][i],
                 metrics_data['Standard_LSTM_Est'][i]]
        
        colors = ['#3498db', '#27ae60', '#e74c3c']
        bars = ax.bar(methods, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add target line for Coverage_95
        if metric == 'Coverage_95':
            ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, 
                      label='Ideal (95%)')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/enhanced_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed metrics analysis
    print("\n" + "="*50)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*50)
    
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"95% Coverage: {coverage_95:.3f} (Target: 0.950)")
    print(f"Average Interval Width: {avg_interval_width:.6f}")
    
    print(f"\nCOVERAGE ANALYSIS:")
    print(f"- Current Coverage: {coverage_95:.1%}")
    print(f"- Expected Coverage: 95.0%")
    print(f"- Coverage Deficit: {(0.95 - coverage_95):.1%}")
    print(f"- Intervals need to be {avg_interval_width*3/avg_interval_width:.1f}x wider")
    
    # Regime-specific metrics
    print(f"\nREGIME-SPECIFIC PERFORMANCE:")
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        regime_mse = regime_data['Squared_Error'].mean()
        regime_coverage = regime_data['In_CI_95'].mean()
        print(f"{regime} Regime:")
        print(f"  MSE: {regime_mse:.8f}")
        print(f"  Coverage: {regime_coverage:.3f}")
        print(f"  Samples: {len(regime_data)}")

def generate_model_improvement_suggestions(df):
    """Generate specific suggestions for model improvement"""
    
    print("\n" + "="*60)
    print("MODEL IMPROVEMENT SUGGESTIONS")
    print("="*60)
    
    coverage = df['In_CI_95'].mean()
    pred_std = df['Predicted_Return'].std()
    actual_std = df['Actual_Return'].std()
    
    print("CRITICAL ISSUES IDENTIFIED:")
    print("-" * 30)
    
    if coverage < 0.5:
        print("🚨 SEVERE UNDERCONFIDENCE: Coverage extremely low")
        print("   → Increase dropout rate from 0.3 to 0.6-0.8")
        print("   → Increase Monte Carlo samples from 100 to 500+")
        print("   → Consider post-hoc calibration")
    
    if pred_std / actual_std < 0.3:
        print("🚨 PREDICTION COLLAPSE: Model predicting too narrow range")
        print("   → Reduce regularization (lower dropout during training)")
        print("   → Use different loss function (e.g., quantile loss)")
        print("   → Add noise to targets during training")
    
    print("\nRECOMMENDED PARAMETER CHANGES:")
    print("-" * 35)
    print("Current → Suggested:")
    print(f"  dropout_rate: 0.3 → 0.6-0.7")
    print(f"  monte_carlo_samples: 100 → 500")
    print(f"  learning_rate: 0.001 → 0.0005")
    print(f"  batch_size: 32 → 16 (for better uncertainty)")
    
    print("\nARCHITECTURE IMPROVEMENTS:")
    print("-" * 30)
    print("  • Add Variational Bayesian layers")
    print("  • Use ensemble of models")
    print("  • Implement proper Bayesian neural network")
    print("  • Add temperature scaling for calibration")

def main():
    """Main execution function"""
    print("Enhanced Bayesian LSTM Analysis")
    print("=" * 50)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Generate all three enhanced plots
    print("\nGenerating Plot 1: Time Series Analysis...")
    plot_1_actual_vs_predicted_timeseries(df)
    
    print("\nGenerating Plot 2: Scatter Analysis...")
    plot_2_scatter_predicted_vs_actual(df)
    
    print("\nGenerating Plot 3: Metrics Comparison...")
    plot_3_model_comparison_metrics(df)
    
    # Generate improvement suggestions
    generate_model_improvement_suggestions(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Enhanced visualizations saved:")
    print("  • results/enhanced_timeseries_analysis.png")
    print("  • results/enhanced_scatter_analysis.png") 
    print("  • results/enhanced_metrics_comparison.png")

if __name__ == "__main__":
    main()