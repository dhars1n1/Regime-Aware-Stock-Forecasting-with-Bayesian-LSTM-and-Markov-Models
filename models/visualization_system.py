"""
Comprehensive Visualization System for Regime-Aware Bayesian LSTM

This module provides:
1. Prediction visualizations with uncertainty bands
2. Regime-aware analysis plots
3. Model performance dashboards  
4. Interactive uncertainty exploration
5. Distribution analysis plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")


class BayesianLSTMVisualizer:
    """
    Comprehensive visualization system for Bayesian LSTM results
    
    Features:
    - Prediction plots with uncertainty bands
    - Regime-aware analysis
    - Performance dashboards
    - Distribution visualizations
    - Interactive plots with Plotly
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        """Initialize visualizer"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color schemes
        self.regime_colors = {
            'Crisis': '#e74c3c',    # Red
            'Normal': '#f39c12',    # Orange  
            'Bull': '#27ae60'       # Green
        }
        
        self.uncertainty_colors = {
            'low': '#3498db',       # Blue
            'medium': '#f39c12',    # Orange
            'high': '#e74c3c'       # Red
        }
    
    def plot_predictions_with_uncertainty(self, 
                                        dates: pd.DatetimeIndex,
                                        predictions: Dict,
                                        actual_values: Optional[np.ndarray] = None,
                                        regime_labels: Optional[np.ndarray] = None,
                                        title: str = "Bayesian LSTM Predictions") -> None:
        """
        Create comprehensive prediction plot with uncertainty visualization
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Main prediction plot
        ax1 = axes[0]
        
        # Plot actual values first (behind predictions)
        if actual_values is not None:
            ax1.plot(dates, actual_values, label='Actual Returns', 
                    color='black', linewidth=2, alpha=0.8, zorder=3)
        
        # Plot predictions
        ax1.plot(dates, predictions['mean'], label='Predicted Returns',
                color='blue', linewidth=2.5, alpha=0.9, zorder=4)
        
        # Uncertainty bands (multiple levels)
        ax1.fill_between(dates, 
                        predictions['ci_95_lower'], 
                        predictions['ci_95_upper'],
                        alpha=0.2, color='blue', label='95% Confidence Interval', zorder=1)
        ax1.fill_between(dates,
                        predictions['ci_68_lower'],
                        predictions['ci_68_upper'],
                        alpha=0.4, color='blue', label='68% Confidence Interval', zorder=2)
        
        ax1.set_ylabel('Log Returns', fontsize=12)
        ax1.set_title('Predictions with Uncertainty Bands', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Regime-colored scatter plot (if regime labels available)
        if regime_labels is not None:
            ax2 = axes[1] 
            
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                if np.sum(mask) > 0:
                    values_to_plot = actual_values if actual_values is not None else predictions['mean']
                    ax2.scatter(dates[mask], values_to_plot[mask],
                               c=self.regime_colors.get(regime, 'gray'),
                               alpha=0.7, s=15, label=f'{regime} Regime')
            
            ax2.set_ylabel('Returns', fontsize=12)
            ax2.set_title('Returns Colored by Market Regime', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Alternative: prediction errors if no regime info
            if actual_values is not None:
                ax2 = axes[1]
                errors = actual_values - predictions['mean']
                ax2.plot(dates, errors, color='red', alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.fill_between(dates, 0, errors, alpha=0.3, color='red')
                ax2.set_ylabel('Prediction Error', fontsize=12)
                ax2.set_title('Prediction Errors', fontsize=14)
                ax2.grid(True, alpha=0.3)
        
        # Uncertainty over time
        ax3 = axes[2]
        ax3.plot(dates, predictions['std'], color='purple', linewidth=2, alpha=0.8)
        ax3.fill_between(dates, 0, predictions['std'], alpha=0.4, color='purple')
        
        # Add uncertainty thresholds
        uncertainty_thresholds = [
            np.percentile(predictions['std'], 33),
            np.percentile(predictions['std'], 67)
        ]
        
        for i, threshold in enumerate(uncertainty_thresholds):
            ax3.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5,
                       label=f'{33*(i+1)}rd percentile')
        
        ax3.set_ylabel('Prediction Uncertainty (σ)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('Prediction Uncertainty Over Time', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'predictions_comprehensive.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive prediction plot saved to {save_path}")
    
    def plot_regime_analysis(self, 
                           predictions: Dict,
                           actual_values: Optional[np.ndarray] = None,
                           regime_labels: Optional[np.ndarray] = None) -> None:
        """
        Create regime-specific analysis plots
        """
        if regime_labels is None:
            print("⚠️ No regime labels provided for regime analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Regime-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Uncertainty by regime (box plot)
        ax1 = axes[0, 0]
        uncertainty_by_regime = []
        regime_names = []
        
        for regime in np.unique(regime_labels):
            mask = regime_labels == regime
            if np.sum(mask) > 0:
                uncertainty_by_regime.extend(predictions['std'][mask])
                regime_names.extend([regime] * np.sum(mask))
        
        df_uncertainty = pd.DataFrame({
            'Uncertainty': uncertainty_by_regime,
            'Regime': regime_names
        })
        
        sns.boxplot(data=df_uncertainty, x='Regime', y='Uncertainty', ax=ax1)
        ax1.set_title('Prediction Uncertainty by Regime')
        ax1.set_ylabel('Prediction Std Dev')
        
        # 2. Prediction accuracy by regime (if actual values available)
        if actual_values is not None:
            ax2 = axes[0, 1]
            errors_by_regime = []
            regime_names_errors = []
            
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                if np.sum(mask) > 0:
                    errors = np.abs(actual_values[mask] - predictions['mean'][mask])
                    errors_by_regime.extend(errors)
                    regime_names_errors.extend([regime] * len(errors))
            
            df_errors = pd.DataFrame({
                'Absolute_Error': errors_by_regime,
                'Regime': regime_names_errors
            })
            
            sns.boxplot(data=df_errors, x='Regime', y='Absolute_Error', ax=ax2)
            ax2.set_title('Prediction Errors by Regime')
            ax2.set_ylabel('Absolute Error')
        
        # 3. Regime distribution over time
        ax3 = axes[1, 0]
        regime_counts = pd.Series(regime_labels).value_counts()
        colors = [self.regime_colors.get(regime, 'gray') for regime in regime_counts.index]
        
        ax3.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax3.set_title('Regime Distribution')
        
        # 4. Prediction vs actual scatter by regime (if actual values available)
        if actual_values is not None:
            ax4 = axes[1, 1]
            
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                if np.sum(mask) > 0:
                    ax4.scatter(actual_values[mask], predictions['mean'][mask],
                               c=self.regime_colors.get(regime, 'gray'),
                               alpha=0.6, s=30, label=f'{regime} Regime')
            
            # Perfect prediction line
            min_val = min(actual_values.min(), predictions['mean'].min())
            max_val = max(actual_values.max(), predictions['mean'].max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax4.set_xlabel('Actual Returns')
            ax4.set_ylabel('Predicted Returns')
            ax4.set_title('Predicted vs Actual by Regime')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'regime_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Regime analysis plot saved to {save_path}")
    
    def plot_uncertainty_distribution(self, predictions: Dict,
                                    actual_values: Optional[np.ndarray] = None) -> None:
        """
        Analyze and visualize uncertainty distributions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Uncertainty Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram of prediction uncertainties
        ax1 = axes[0, 0]
        ax1.hist(predictions['std'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(np.mean(predictions['std']), color='red', linestyle='--',
                   label=f'Mean: {np.mean(predictions["std"]):.4f}')
        ax1.axvline(np.median(predictions['std']), color='orange', linestyle='--',
                   label=f'Median: {np.median(predictions["std"]):.4f}')
        ax1.set_xlabel('Prediction Uncertainty (σ)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Uncertainties')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality check
        ax2 = axes[0, 1]
        stats.probplot(predictions['std'], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Uncertainty Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Uncertainty vs prediction magnitude
        ax3 = axes[1, 0]
        ax3.scatter(np.abs(predictions['mean']), predictions['std'], 
                   alpha=0.6, s=20, color='blue')
        
        # Add trend line
        z = np.polyfit(np.abs(predictions['mean']), predictions['std'], 1)
        p = np.poly1d(z)
        ax3.plot(np.sort(np.abs(predictions['mean'])), 
                p(np.sort(np.abs(predictions['mean']))), "r--", alpha=0.8)
        
        ax3.set_xlabel('|Predicted Return|')
        ax3.set_ylabel('Prediction Uncertainty')
        ax3.set_title('Uncertainty vs Prediction Magnitude')
        ax3.grid(True, alpha=0.3)
        
        # 4. Calibration plot (if actual values available)
        if actual_values is not None:
            ax4 = axes[1, 1]
            
            # Calculate empirical coverage for different confidence levels
            confidence_levels = np.linspace(0.1, 0.99, 20)
            empirical_coverage = []
            
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower = np.percentile(predictions['std'], (alpha/2) * 100)
                upper = np.percentile(predictions['std'], (1 - alpha/2) * 100)
                
                # This is simplified - in practice you'd need the actual prediction intervals
                coverage = np.mean((actual_values >= predictions['mean'] - 1.96 * predictions['std']) & 
                                 (actual_values <= predictions['mean'] + 1.96 * predictions['std']))
                empirical_coverage.append(coverage)
            
            ax4.plot(confidence_levels, empirical_coverage, 'bo-', alpha=0.7, label='Empirical')
            ax4.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')
            ax4.set_xlabel('Nominal Coverage')
            ax4.set_ylabel('Empirical Coverage') 
            ax4.set_title('Calibration Plot')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Show correlation between consecutive uncertainties
            ax4 = axes[1, 1]
            if len(predictions['std']) > 1:
                ax4.scatter(predictions['std'][:-1], predictions['std'][1:], alpha=0.6)
                ax4.set_xlabel('Uncertainty at t')
                ax4.set_ylabel('Uncertainty at t+1')
                ax4.set_title('Uncertainty Temporal Correlation')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'uncertainty_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Uncertainty distribution analysis saved to {save_path}")
    
    def create_performance_dashboard(self, 
                                   evaluation_results: Dict,
                                   predictions: Dict,
                                   actual_values: np.ndarray,
                                   dates: pd.DatetimeIndex) -> None:
        """
        Create a comprehensive performance dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Bayesian LSTM Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Key metrics summary (text)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        overall = evaluation_results['overall']
        metrics_text = f"""
        OVERALL PERFORMANCE METRICS
        
        Mean Squared Error: {overall['mse']:.6f}
        Mean Absolute Error: {overall['mae']:.6f}
        Root Mean Squared Error: {overall['rmse']:.6f}
        R² Score: {overall['r2']:.4f}
        
        UNCERTAINTY QUANTIFICATION
        
        95% Coverage: {overall['coverage_95']:.3f} (Target: 0.950)
        Average Interval Width: {overall['interval_width_95']:.6f}
        Mean Uncertainty: {np.mean(predictions['std']):.6f}
        """
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Regime performance comparison (if available)
        if 'by_regime' in evaluation_results:
            ax2 = fig.add_subplot(gs[0, 2:])
            
            regimes = list(evaluation_results['by_regime'].keys())
            mse_values = [evaluation_results['by_regime'][r]['mse'] for r in regimes]
            mae_values = [evaluation_results['by_regime'][r]['mae'] for r in regimes]
            
            x = np.arange(len(regimes))
            width = 0.35
            
            ax2.bar(x - width/2, mse_values, width, label='MSE', alpha=0.7)
            ax2.bar(x + width/2, mae_values, width, label='MAE', alpha=0.7)
            
            ax2.set_xlabel('Market Regime')
            ax2.set_ylabel('Error')
            ax2.set_title('Performance by Market Regime')
            ax2.set_xticks(x)
            ax2.set_xticklabels(regimes)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Time series of predictions
        ax3 = fig.add_subplot(gs[1, :])
        
        # Sample recent data for clarity
        recent_mask = slice(-200, None) if len(dates) > 200 else slice(None)
        recent_dates = dates[recent_mask]
        recent_actual = actual_values[recent_mask]
        recent_pred = predictions['mean'][recent_mask]
        recent_lower = predictions['ci_95_lower'][recent_mask]
        recent_upper = predictions['ci_95_upper'][recent_mask]
        
        ax3.plot(recent_dates, recent_actual, label='Actual', color='black', linewidth=2)
        ax3.plot(recent_dates, recent_pred, label='Predicted', color='blue', linewidth=2)
        ax3.fill_between(recent_dates, recent_lower, recent_upper, 
                        alpha=0.3, color='blue', label='95% CI')
        
        ax3.set_title('Recent Predictions (Last 200 observations)')
        ax3.set_ylabel('Log Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals histogram
        ax4 = fig.add_subplot(gs[2, 0])
        residuals = actual_values - predictions['mean']
        ax4.hist(residuals, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Residuals Distribution')
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Prediction vs Actual scatter
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(actual_values, predictions['mean'], alpha=0.5, s=10)
        
        min_val = min(actual_values.min(), predictions['mean'].min())
        max_val = max(actual_values.max(), predictions['mean'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax5.set_xlabel('Actual Returns')
        ax5.set_ylabel('Predicted Returns')
        ax5.set_title('Predicted vs Actual')
        ax5.grid(True, alpha=0.3)
        
        # 6. Uncertainty over time
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.plot(dates, predictions['std'], color='purple', alpha=0.7)
        ax6.set_title('Uncertainty Evolution')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Prediction Std')
        ax6.grid(True, alpha=0.3)
        
        # 7. Coverage analysis
        ax7 = fig.add_subplot(gs[2, 3])
        in_ci = ((actual_values >= predictions['ci_95_lower']) & 
                (actual_values <= predictions['ci_95_upper']))
        
        # Rolling coverage
        window = 50
        rolling_coverage = pd.Series(in_ci.astype(float)).rolling(window).mean()
        
        ax7.plot(dates, rolling_coverage, color='green', alpha=0.8)
        ax7.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Target (95%)')
        ax7.set_title(f'Rolling Coverage ({window} periods)')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Coverage Ratio')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Save dashboard
        save_path = os.path.join(self.save_dir, 'performance_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance dashboard saved to {save_path}")
    
    def create_interactive_plots(self, 
                               dates: pd.DatetimeIndex,
                               predictions: Dict,
                               actual_values: Optional[np.ndarray] = None,
                               regime_labels: Optional[np.ndarray] = None) -> None:
        """
        Create interactive Plotly visualizations
        """
        print("🌐 Creating interactive visualizations...")
        
        # Main interactive plot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Predictions with Uncertainty', 'Prediction Errors', 'Uncertainty Over Time'),
            vertical_spacing=0.08
        )
        
        # Predictions plot
        fig.add_trace(
            go.Scatter(x=dates, y=predictions['mean'], mode='lines',
                      name='Predicted Returns', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if actual_values is not None:
            fig.add_trace(
                go.Scatter(x=dates, y=actual_values, mode='lines',
                          name='Actual Returns', line=dict(color='black', width=2)),
                row=1, col=1
            )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(x=dates, y=predictions['ci_95_upper'], mode='lines',
                      line=dict(width=0), showlegend=False, name='Upper 95% CI'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=predictions['ci_95_lower'], mode='lines',
                      fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                      line=dict(width=0), name='95% Confidence Interval'),
            row=1, col=1
        )
        
        # Prediction errors
        if actual_values is not None:
            errors = actual_values - predictions['mean']
            fig.add_trace(
                go.Scatter(x=dates, y=errors, mode='lines',
                          name='Prediction Errors', line=dict(color='red')),
                row=2, col=1
            )
        
        # Uncertainty
        fig.add_trace(
            go.Scatter(x=dates, y=predictions['std'], mode='lines',
                      fill='tozeroy', fillcolor='rgba(128,0,128,0.3)',
                      name='Prediction Uncertainty', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Interactive Bayesian LSTM Results',
            height=800,
            showlegend=True
        )
        
        # Save interactive plot
        save_path = os.path.join(self.save_dir, 'interactive_predictions.html')
        fig.write_html(save_path)
        print(f"✅ Interactive plot saved to {save_path}")
    
    def generate_all_visualizations(self,
                                  dates: pd.DatetimeIndex,
                                  predictions: Dict,
                                  evaluation_results: Dict,
                                  actual_values: Optional[np.ndarray] = None,
                                  regime_labels: Optional[np.ndarray] = None) -> None:
        """
        Generate all visualizations in sequence
        """
        print("🎨 GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 60)
        
        # 1. Main predictions plot
        print("1️⃣ Creating prediction plots...")
        self.plot_predictions_with_uncertainty(
            dates, predictions, actual_values, regime_labels
        )
        
        # 2. Regime analysis
        if regime_labels is not None:
            print("2️⃣ Creating regime analysis...")
            self.plot_regime_analysis(predictions, actual_values, regime_labels)
        
        # 3. Uncertainty analysis
        print("3️⃣ Creating uncertainty analysis...")
        self.plot_uncertainty_distribution(predictions, actual_values)
        
        # 4. Performance dashboard
        if actual_values is not None:
            print("4️⃣ Creating performance dashboard...")
            self.create_performance_dashboard(
                evaluation_results, predictions, actual_values, dates
            )
        
        # 5. Interactive plots
        print("5️⃣ Creating interactive plots...")
        self.create_interactive_plots(dates, predictions, actual_values, regime_labels)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Saved to directory: {self.save_dir}")
        print("=" * 60)


def demo_visualization_system():
    """Demonstrate the visualization system"""
    print("🧪 VISUALIZATION SYSTEM DEMO")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Sample predictions
    actual_returns = np.random.normal(0, 0.02, n_samples)
    predicted_returns = actual_returns + np.random.normal(0, 0.005, n_samples)
    uncertainty = np.abs(np.random.normal(0.01, 0.005, n_samples))
    
    predictions = {
        'mean': predicted_returns,
        'std': uncertainty,
        'ci_95_lower': predicted_returns - 1.96 * uncertainty,
        'ci_95_upper': predicted_returns + 1.96 * uncertainty,
        'ci_68_lower': predicted_returns - uncertainty,
        'ci_68_upper': predicted_returns + uncertainty
    }
    
    # Sample regime labels
    regime_labels = np.random.choice(['Bull', 'Normal', 'Crisis'], n_samples, 
                                   p=[0.3, 0.5, 0.2])
    
    # Sample evaluation results
    evaluation_results = {
        'overall': {
            'mse': 0.0001,
            'mae': 0.008,
            'rmse': 0.01,
            'r2': 0.75,
            'coverage_95': 0.946,
            'interval_width_95': 0.04
        },
        'by_regime': {
            'Bull': {'mse': 0.00008, 'mae': 0.007},
            'Normal': {'mse': 0.0001, 'mae': 0.008}, 
            'Crisis': {'mse': 0.00015, 'mae': 0.012}
        }
    }
    
    # Create visualizer and generate plots
    visualizer = BayesianLSTMVisualizer("demo_visualizations")
    
    print("Generating sample visualizations...")
    visualizer.generate_all_visualizations(
        dates, predictions, evaluation_results, actual_returns, regime_labels
    )
    
    print("✅ Demo completed! Check 'demo_visualizations' directory.")


if __name__ == "__main__":
    demo_visualization_system()