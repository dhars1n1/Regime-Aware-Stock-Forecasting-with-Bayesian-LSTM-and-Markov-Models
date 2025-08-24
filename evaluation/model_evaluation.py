"""
Comprehensive evaluation of the Regime-Aware Bayesian LSTM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import os

class ModelEvaluator:
    def __init__(self, predictions_file):
        """Load predictions and prepare for evaluation"""
        self.df = pd.read_csv(predictions_file, parse_dates=['Date'])
        self.df.set_index('Date', inplace=True)
        
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        actual = self.df['Actual'].values
        predicted = self.df['Predicted'].values
        
        metrics = {
            'MSE': mean_squared_error(actual, predicted),
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'R2': r2_score(actual, predicted),
            'Correlation': np.corrcoef(actual, predicted)[0, 1],
            'Mean_Bias': np.mean(predicted - actual),
            'Std_Bias': np.std(predicted - actual)
        }
        
        return metrics
    
    def uncertainty_calibration(self):
        """Evaluate uncertainty calibration"""
        actual = self.df['Actual'].values
        lower_ci = self.df['Lower_CI'].values
        upper_ci = self.df['Upper_CI'].values
        
        # Coverage probability (should be ~0.95 for 95% CI)
        coverage = np.mean((actual >= lower_ci) & (actual <= upper_ci))
        
        # Average interval width
        avg_width = np.mean(upper_ci - lower_ci)
        
        # Uncertainty vs error correlation
        uncertainty = self.df['Uncertainty'].values
        errors = np.abs(actual - self.df['Predicted'].values)
        uncertainty_correlation = np.corrcoef(uncertainty, errors)[0, 1]
        
        return {
            'coverage_probability': coverage,
            'average_interval_width': avg_width,
            'uncertainty_error_correlation': uncertainty_correlation
        }
    
    def regime_specific_analysis(self):
        """Analyze performance by regime"""
        regime_metrics = {}
        
        for regime in self.df['Regime'].unique():
            regime_data = self.df[self.df['Regime'] == regime]
            actual = regime_data['Actual'].values
            predicted = regime_data['Predicted'].values
            
            if len(actual) > 0:
                regime_metrics[regime] = {
                    'MSE': mean_squared_error(actual, predicted),
                    'MAE': mean_absolute_error(actual, predicted),
                    'R2': r2_score(actual, predicted),
                    'Count': len(actual),
                    'Avg_Uncertainty': regime_data['Uncertainty'].mean()
                }
        
        return regime_metrics
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Time series plot with uncertainty
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(self.df.index, self.df['Actual'], label='Actual', alpha=0.7, color='black')
        plt.plot(self.df.index, self.df['Predicted'], label='Predicted', color='blue')
        plt.fill_between(self.df.index, self.df['Lower_CI'], self.df['Upper_CI'], 
                        alpha=0.3, color='blue', label='95% CI')
        plt.title('Time Series Predictions with Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Scatter plot: Actual vs Predicted
        ax2 = plt.subplot(3, 3, 2)
        plt.scatter(self.df['Actual'], self.df['Predicted'], alpha=0.6)
        min_val = min(self.df['Actual'].min(), self.df['Predicted'].min())
        max_val = max(self.df['Actual'].max(), self.df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        ax3 = plt.subplot(3, 3, 3)
        residuals = self.df['Actual'] - self.df['Predicted']
        plt.scatter(self.df['Predicted'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 4. Uncertainty vs Error
        ax4 = plt.subplot(3, 3, 4)
        errors = np.abs(self.df['Actual'] - self.df['Predicted'])
        plt.scatter(self.df['Uncertainty'], errors, alpha=0.6)
        plt.xlabel('Predicted Uncertainty')
        plt.ylabel('Absolute Error')
        plt.title('Uncertainty vs Absolute Error')
        plt.grid(True, alpha=0.3)
        
        # 5. Regime-specific performance
        ax5 = plt.subplot(3, 3, 5)
        regime_metrics = self.regime_specific_analysis()
        regimes = list(regime_metrics.keys())
        mse_values = [regime_metrics[r]['MSE'] for r in regimes]
        colors = ['red', 'orange', 'green'][:len(regimes)]
        plt.bar(regimes, mse_values, color=colors, alpha=0.7)
        plt.title('MSE by Regime')
        plt.ylabel('MSE')
        plt.xticks(rotation=45)
        
        # 6. Coverage probability visualization
        ax6 = plt.subplot(3, 3, 6)
        coverage_by_regime = {}
        for regime in self.df['Regime'].unique():
            regime_data = self.df[self.df['Regime'] == regime]
            actual = regime_data['Actual'].values
            lower_ci = regime_data['Lower_CI'].values
            upper_ci = regime_data['Upper_CI'].values
            coverage = np.mean((actual >= lower_ci) & (actual <= upper_ci))
            coverage_by_regime[regime] = coverage
        
        regimes = list(coverage_by_regime.keys())
        coverage_values = list(coverage_by_regime.values())
        plt.bar(regimes, coverage_values, alpha=0.7)
        plt.axhline(y=0.95, color='r', linestyle='--', label='Expected (95%)')
        plt.title('Coverage Probability by Regime')
        plt.ylabel('Coverage Probability')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 7. Distribution of residuals
        ax7 = plt.subplot(3, 3, 7)
        plt.hist(residuals, bins=50, alpha=0.7, density=True, color='skyblue')
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Regime transition analysis
        ax8 = plt.subplot(3, 3, 8)
        colors_map = {'Crisis': 'red', 'Normal': 'orange', 'Bull': 'green'}
        for regime in self.df['Regime'].unique():
            regime_data = self.df[self.df['Regime'] == regime]
            plt.scatter(regime_data.index, regime_data['Actual'], 
                       c=colors_map.get(regime, 'blue'), label=f'{regime}', alpha=0.6, s=20)
        plt.title('Actual Returns by Regime')
        plt.ylabel('Actual Returns')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 9. Uncertainty over time
        ax9 = plt.subplot(3, 3, 9)
        plt.plot(self.df.index, self.df['Uncertainty'], alpha=0.7, color='purple')
        plt.title('Model Uncertainty Over Time')
        plt.ylabel('Uncertainty (Std Dev)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("=" * 60)
        print("REGIME-AWARE BAYESIAN LSTM EVALUATION REPORT")
        print("=" * 60)
        
        # Overall metrics
        overall_metrics = self.calculate_metrics()
        print("\n📊 OVERALL PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in overall_metrics.items():
            print(f"{metric:>25}: {value:.6f}")
        
        # Uncertainty calibration
        uncertainty_metrics = self.uncertainty_calibration()
        print("\n🎯 UNCERTAINTY CALIBRATION:")
        print("-" * 40)
        for metric, value in uncertainty_metrics.items():
            print(f"{metric:>25}: {value:.6f}")
        
        print(f"\n📝 Calibration Assessment:")
        coverage = uncertainty_metrics['coverage_probability']
        if 0.93 <= coverage <= 0.97:
            print(f"   ✅ Well-calibrated (Coverage: {coverage:.3f})")
        else:
            print(f"   ⚠️  Miscalibrated (Coverage: {coverage:.3f}, Expected: ~0.95)")
        
        # Regime-specific analysis
        regime_metrics = self.regime_specific_analysis()
        print("\n🏛️ REGIME-SPECIFIC PERFORMANCE:")
        print("-" * 40)
        for regime, metrics in regime_metrics.items():
            print(f"\n{regime} Market:")
            for metric, value in metrics.items():
                if metric != 'Count':
                    print(f"  {metric:>20}: {value:.6f}")
                else:
                    print(f"  {metric:>20}: {value}")
        
        # Model insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 40)
        
        # Best performing regime
        best_regime = min(regime_metrics.keys(), 
                         key=lambda x: regime_metrics[x]['MSE'])
        worst_regime = max(regime_metrics.keys(), 
                          key=lambda x: regime_metrics[x]['MSE'])
        
        print(f"   • Best performance in {best_regime} markets")
        print(f"   • Most challenging regime: {worst_regime}")
        
        # Uncertainty-error correlation
        unc_corr = uncertainty_metrics['uncertainty_error_correlation']
        if unc_corr > 0.3:
            print(f"   • Good uncertainty estimation (correlation: {unc_corr:.3f})")
        else:
            print(f"   • Uncertainty estimation needs improvement (correlation: {unc_corr:.3f})")
        
        print("\n" + "=" * 60)

def main():
    """Run comprehensive model evaluation"""
    predictions_file = "results/bayesian_lstm_predictions.csv"
    
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        print("Please run the Bayesian LSTM model first to generate predictions.")
        return
    
    evaluator = ModelEvaluator(predictions_file)
    
    # Generate report
    evaluator.generate_report()
    
    # Create visualizations
    evaluator.plot_comprehensive_analysis()

if __name__ == "__main__":
    main()
