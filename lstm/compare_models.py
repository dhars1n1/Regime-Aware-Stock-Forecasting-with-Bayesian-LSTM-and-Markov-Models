"""
Comprehensive Comparison: Bayesian LSTM vs Standard LSTM
Analyzes prediction accuracy, uncertainty quantification, and regime-specific performance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, bayesian_path="results", standard_path="results_standard"):
        self.bayesian_path = bayesian_path
        self.standard_path = standard_path
        self.comparison_results = {}
        
    def load_results(self):
        """Load prediction results from both models"""
        print("Loading results...")
        
        # Load Bayesian LSTM results
        self.bayesian_predictions = pd.read_csv(
            f"{self.bayesian_path}/detailed_predictions.csv"
        )
        self.bayesian_summary = pd.read_csv(
            f"{self.bayesian_path}/performance_summary.csv"
        )
        
        # Load Standard LSTM results
        self.standard_predictions = pd.read_csv(
            f"{self.standard_path}/detailed_predictions_standard.csv"
        )
        self.standard_summary = pd.read_csv(
            f"{self.standard_path}/performance_summary_standard.csv"
        )
        
        # Convert date columns
        self.bayesian_predictions['Date'] = pd.to_datetime(self.bayesian_predictions['Date'])
        self.standard_predictions['Date'] = pd.to_datetime(self.standard_predictions['Date'])
        
        print(f"✓ Bayesian predictions: {len(self.bayesian_predictions)} samples")
        print(f"✓ Standard predictions: {len(self.standard_predictions)} samples")
        
    def compare_overall_performance(self):
        """Compare overall prediction accuracy metrics"""
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE COMPARISON")
        print("="*80)
        
        metrics = ['Overall_MSE', 'Overall_MAE', 'Overall_RMSE', 'Overall_R2']
        
        comparison_df = pd.DataFrame({
            'Metric': metrics,
            'Bayesian_LSTM': [self.bayesian_summary[m].values[0] for m in metrics],
            'Standard_LSTM': [self.standard_summary[m].values[0] for m in metrics]
        })
        
        comparison_df['Difference'] = comparison_df['Bayesian_LSTM'] - comparison_df['Standard_LSTM']
        comparison_df['Improvement_%'] = (
            (comparison_df['Standard_LSTM'] - comparison_df['Bayesian_LSTM']) / 
            comparison_df['Standard_LSTM'] * 100
        )
        
        # For R2, positive difference is better
        comparison_df.loc[comparison_df['Metric'] == 'Overall_R2', 'Improvement_%'] *= -1
        
        print("\n" + "-"*80)
        print(f"{'Metric':<20} {'Bayesian':<15} {'Standard':<15} {'Difference':<15} {'Improvement %':<15}")
        print("-"*80)
        
        for _, row in comparison_df.iterrows():
            metric_name = row['Metric'].replace('Overall_', '')
            print(f"{metric_name:<20} {row['Bayesian_LSTM']:<15.6f} {row['Standard_LSTM']:<15.6f} "
                  f"{row['Difference']:<15.6f} {row['Improvement_%']:<15.2f}%")
        
        print("-"*80)
        
        # Determine winner
        better_count = (comparison_df['Improvement_%'] > 0).sum()
        if better_count > len(metrics) / 2:
            winner = "Bayesian LSTM"
        elif better_count < len(metrics) / 2:
            winner = "Standard LSTM"
        else:
            winner = "Tie"
        
        print(f"\n🏆 Winner (Prediction Accuracy): {winner}")
        
        self.comparison_results['overall'] = comparison_df
        return comparison_df
    
    def compare_regime_performance(self):
        """Compare performance across different market regimes"""
        print("\n" + "="*80)
        print("REGIME-SPECIFIC PERFORMANCE COMPARISON")
        print("="*80)
        
        regimes = self.bayesian_predictions['Regime'].unique()
        
        regime_comparison = []
        
        for regime in regimes:
            print(f"\n{regime} Regime:")
            print("-" * 60)
            
            # Bayesian metrics
            bayesian_regime = self.bayesian_predictions[
                self.bayesian_predictions['Regime'] == regime
            ]
            bayesian_mae = bayesian_regime['Absolute_Error'].mean()
            bayesian_mse = bayesian_regime['Squared_Error'].mean()
            bayesian_rmse = np.sqrt(bayesian_mse)
            
            # Standard metrics
            standard_regime = self.standard_predictions[
                self.standard_predictions['Regime'] == regime
            ]
            standard_mae = standard_regime['Absolute_Error'].mean()
            standard_mse = standard_regime['Squared_Error'].mean()
            standard_rmse = np.sqrt(standard_mse)
            
            # Calculate improvements
            mae_improvement = (standard_mae - bayesian_mae) / standard_mae * 100
            mse_improvement = (standard_mse - bayesian_mse) / standard_mse * 100
            rmse_improvement = (standard_rmse - bayesian_rmse) / standard_rmse * 100
            
            print(f"  MAE:  Bayesian={bayesian_mae:.6f}, Standard={standard_mae:.6f}, "
                  f"Improvement={mae_improvement:+.2f}%")
            print(f"  MSE:  Bayesian={bayesian_mse:.6f}, Standard={standard_mse:.6f}, "
                  f"Improvement={mse_improvement:+.2f}%")
            print(f"  RMSE: Bayesian={bayesian_rmse:.6f}, Standard={standard_rmse:.6f}, "
                  f"Improvement={rmse_improvement:+.2f}%")
            
            regime_comparison.append({
                'Regime': regime,
                'Count': len(bayesian_regime),
                'Bayesian_MAE': bayesian_mae,
                'Standard_MAE': standard_mae,
                'MAE_Improvement_%': mae_improvement,
                'Bayesian_RMSE': bayesian_rmse,
                'Standard_RMSE': standard_rmse,
                'RMSE_Improvement_%': rmse_improvement
            })
        
        regime_df = pd.DataFrame(regime_comparison)
        self.comparison_results['regime'] = regime_df
        
        return regime_df
    
    def analyze_uncertainty_quality(self):
        """Analyze the quality of Bayesian LSTM's uncertainty estimates"""
        print("\n" + "="*80)
        print("UNCERTAINTY QUANTIFICATION ANALYSIS (Bayesian LSTM Only)")
        print("="*80)
        
        # Coverage analysis
        coverage_95 = self.bayesian_predictions['In_CI_95'].mean()
        print(f"\n95% Confidence Interval Coverage: {coverage_95:.3f}")
        print(f"  Expected: 0.950")
        print(f"  Calibration Error: {abs(coverage_95 - 0.95):.3f}")
        
        if abs(coverage_95 - 0.95) < 0.05:
            print(f"  ✓ Well-calibrated uncertainty estimates")
        else:
            if coverage_95 > 0.95:
                print(f"  ⚠ Overconfident (intervals too wide)")
            else:
                print(f"  ⚠ Underconfident (intervals too narrow)")
        
        # Uncertainty by regime
        print("\nUncertainty Statistics by Regime:")
        print("-" * 60)
        
        for regime in self.bayesian_predictions['Regime'].unique():
            regime_data = self.bayesian_predictions[
                self.bayesian_predictions['Regime'] == regime
            ]
            
            avg_uncertainty = regime_data['Prediction_Uncertainty'].mean()
            avg_width = (regime_data['Upper_95_CI'] - regime_data['Lower_95_CI']).mean()
            regime_coverage = regime_data['In_CI_95'].mean()
            
            print(f"\n  {regime} Regime ({len(regime_data)} samples):")
            print(f"    Avg Uncertainty: {avg_uncertainty:.6f}")
            print(f"    Avg Interval Width: {avg_width:.6f}")
            print(f"    Coverage: {regime_coverage:.3f}")
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(
            self.bayesian_predictions['Prediction_Uncertainty'],
            self.bayesian_predictions['Absolute_Error']
        )[0, 1]
        
        print(f"\nCorrelation (Uncertainty vs Absolute Error): {correlation:.3f}")
        if correlation > 0.3:
            print(f"  ✓ Uncertainty is predictive of errors")
        else:
            print(f"  ⚠ Weak relationship between uncertainty and errors")
        
        # Advantage over standard LSTM
        print("\n" + "="*80)
        print("KEY ADVANTAGE: RISK-AWARE DECISION MAKING")
        print("="*80)
        
        # High uncertainty predictions
        high_uncertainty_threshold = self.bayesian_predictions['Prediction_Uncertainty'].quantile(0.75)
        high_uncertainty_mask = self.bayesian_predictions['Prediction_Uncertainty'] > high_uncertainty_threshold
        
        bayesian_high_unc_mae = self.bayesian_predictions[high_uncertainty_mask]['Absolute_Error'].mean()
        standard_high_unc_mae = self.standard_predictions[high_uncertainty_mask]['Absolute_Error'].mean()
        
        print(f"\nHigh Uncertainty Predictions (top 25%):")
        print(f"  Bayesian LSTM can FLAG these as risky")
        print(f"  MAE in high uncertainty: {bayesian_high_unc_mae:.6f}")
        print(f"  Standard LSTM has NO WARNING system")
        print(f"  MAE in same period: {standard_high_unc_mae:.6f}")
        print(f"\n  💡 Bayesian LSTM allows selective trading:")
        print(f"     - Trade only when uncertainty < threshold")
        print(f"     - Adjust position size based on confidence")
        print(f"     - Avoid high-risk periods identified by model")
        
    def compare_error_distributions(self):
        """Compare error distributions between models"""
        print("\n" + "="*80)
        print("ERROR DISTRIBUTION ANALYSIS")
        print("="*80)
        
        bayesian_errors = self.bayesian_predictions['Absolute_Error']
        standard_errors = self.standard_predictions['Absolute_Error']
        
        print(f"\nAbsolute Error Statistics:")
        print(f"{'Metric':<20} {'Bayesian':<15} {'Standard':<15} {'Difference':<15}")
        print("-" * 65)
        
        stats_comparison = {
            'Mean': (bayesian_errors.mean(), standard_errors.mean()),
            'Median': (bayesian_errors.median(), standard_errors.median()),
            'Std Dev': (bayesian_errors.std(), standard_errors.std()),
            'Max': (bayesian_errors.max(), standard_errors.max()),
            '95th Percentile': (bayesian_errors.quantile(0.95), standard_errors.quantile(0.95))
        }
        
        for metric, (bayesian_val, standard_val) in stats_comparison.items():
            diff = bayesian_val - standard_val
            print(f"{metric:<20} {bayesian_val:<15.6f} {standard_val:<15.6f} {diff:<15.6f}")
        
        # Statistical test
        statistic, p_value = stats.wilcoxon(bayesian_errors, standard_errors)
        print(f"\nWilcoxon Signed-Rank Test:")
        print(f"  Statistic: {statistic:.2f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ Significant difference in error distributions (p < 0.05)")
        else:
            print(f"  ○ No significant difference in error distributions (p >= 0.05)")
    
    def create_comprehensive_visualizations(self, save_path="comparison_results"):
        """Create comprehensive comparison visualizations"""
        os.makedirs(save_path, exist_ok=True)
        print(f"\nCreating comparison visualizations...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Side-by-side predictions
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Bayesian predictions
        ax1 = axes[0]
        ax1.plot(self.bayesian_predictions['Date'], 
                self.bayesian_predictions['Actual_Return'], 
                label='Actual', color='black', alpha=0.7, linewidth=1.5)
        ax1.plot(self.bayesian_predictions['Date'], 
                self.bayesian_predictions['Predicted_Return'], 
                label='Bayesian Prediction', color='blue', linewidth=1.5)
        ax1.fill_between(self.bayesian_predictions['Date'],
                        self.bayesian_predictions['Lower_95_CI'],
                        self.bayesian_predictions['Upper_95_CI'],
                        alpha=0.3, color='blue', label='95% CI')
        ax1.set_title('Bayesian LSTM with Uncertainty Quantification', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Log Returns', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Standard predictions
        ax2 = axes[1]
        ax2.plot(self.standard_predictions['Date'], 
                self.standard_predictions['Actual_Return'], 
                label='Actual', color='black', alpha=0.7, linewidth=1.5)
        ax2.plot(self.standard_predictions['Date'], 
                self.standard_predictions['Predicted_Return'], 
                label='Standard Prediction', color='red', linewidth=1.5)
        ax2.set_title('Standard LSTM (No Uncertainty)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Log Returns', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/side_by_side_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error comparison by regime
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        error_data = []
        labels = []
        for regime in self.bayesian_predictions['Regime'].unique():
            bayesian_regime_errors = self.bayesian_predictions[
                self.bayesian_predictions['Regime'] == regime
            ]['Absolute_Error']
            standard_regime_errors = self.standard_predictions[
                self.standard_predictions['Regime'] == regime
            ]['Absolute_Error']
            
            error_data.extend([bayesian_regime_errors, standard_regime_errors])
            labels.extend([f'{regime}\nBayesian', f'{regime}\nStandard'])
        
        bp = axes[0].boxplot(error_data, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            if i % 2 == 0:
                patch.set_facecolor('lightblue')
            else:
                patch.set_facecolor('lightcoral')
        
        axes[0].set_title('Absolute Errors by Regime and Model', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Absolute Error', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Bar plot - Average errors
        regime_df = self.comparison_results['regime']
        x = np.arange(len(regime_df))
        width = 0.35
        
        axes[1].bar(x - width/2, regime_df['Bayesian_MAE'], width, 
                   label='Bayesian LSTM', color='lightblue')
        axes[1].bar(x + width/2, regime_df['Standard_MAE'], width, 
                   label='Standard LSTM', color='lightcoral')
        
        axes[1].set_xlabel('Regime', fontsize=12)
        axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[1].set_title('Average MAE by Regime', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regime_df['Regime'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/regime_error_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Uncertainty vs Error (Bayesian only)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        regime_colors = {'Crisis': '#e74c3c', 'Normal': '#f39c12', 'Bull': '#27ae60'}
        
        for regime in self.bayesian_predictions['Regime'].unique():
            regime_data = self.bayesian_predictions[
                self.bayesian_predictions['Regime'] == regime
            ]
            axes[0].scatter(regime_data['Prediction_Uncertainty'],
                          regime_data['Absolute_Error'],
                          c=regime_colors.get(regime, 'gray'),
                          alpha=0.6, s=20, label=regime)
        
        axes[0].set_xlabel('Prediction Uncertainty', fontsize=12)
        axes[0].set_ylabel('Absolute Error', fontsize=12)
        axes[0].set_title('Uncertainty vs Error (Bayesian LSTM)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(self.bayesian_predictions['Prediction_Uncertainty'], 
                      self.bayesian_predictions['Absolute_Error'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.bayesian_predictions['Prediction_Uncertainty'].min(),
                            self.bayesian_predictions['Prediction_Uncertainty'].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Error distribution comparison
        axes[1].hist(self.bayesian_predictions['Absolute_Error'], 
                    bins=50, alpha=0.6, label='Bayesian LSTM', color='blue')
        axes[1].hist(self.standard_predictions['Absolute_Error'], 
                    bins=50, alpha=0.6, label='Standard LSTM', color='red')
        axes[1].set_xlabel('Absolute Error', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/uncertainty_and_error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance metrics comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        overall_df = self.comparison_results['overall']
        metrics = ['MSE', 'MAE', 'RMSE']
        bayesian_values = []
        standard_values = []
        
        for metric in metrics:
            row = overall_df[overall_df['Metric'] == f'Overall_{metric}']
            bayesian_values.append(row['Bayesian_LSTM'].values[0])
            standard_values.append(row['Standard_LSTM'].values[0])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, bayesian_values, width, label='Bayesian LSTM', color='lightblue')
        ax.bar(x + width/2, standard_values, width, label='Standard LSTM', color='lightcoral')
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Overall Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement percentages on top
        for i, (b, s) in enumerate(zip(bayesian_values, standard_values)):
            improvement = (s - b) / s * 100
            y_pos = max(b, s) * 1.05
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, y_pos, f'{improvement:+.1f}%', 
                   ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/overall_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {save_path}/")
    
    def generate_comparison_report(self, save_path="comparison_results"):
        """Generate a comprehensive comparison report"""
        os.makedirs(save_path, exist_ok=True)
        
        report_path = f"{save_path}/comparison_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BAYESIAN LSTM vs STANDARD LSTM - COMPREHENSIVE COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall performance
            f.write("1. OVERALL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            overall_df = self.comparison_results['overall']
            f.write(overall_df.to_string(index=False))
            f.write("\n\n")
            
            # Regime performance
            f.write("2. REGIME-SPECIFIC PERFORMANCE\n")
            f.write("-"*80 + "\n")
            regime_df = self.comparison_results['regime']
            f.write(regime_df.to_string(index=False))
            f.write("\n\n")
            
            # Uncertainty analysis
            f.write("3. UNCERTAINTY QUANTIFICATION (Bayesian LSTM Only)\n")
            f.write("-"*80 + "\n")
            coverage = self.bayesian_predictions['In_CI_95'].mean()
            f.write(f"95% CI Coverage: {coverage:.3f}\n")
            f.write(f"Calibration Error: {abs(coverage - 0.95):.3f}\n")
            
            correlation = np.corrcoef(
                self.bayesian_predictions['Prediction_Uncertainty'],
                self.bayesian_predictions['Absolute_Error']
            )[0, 1]
            f.write(f"Uncertainty-Error Correlation: {correlation:.3f}\n\n")
            
            # Key findings
            f.write("4. KEY FINDINGS\n")
            f.write("-"*80 + "\n")
            
            avg_improvement = overall_df['Improvement_%'].mean()
            f.write(f"• Average improvement: {avg_improvement:+.2f}%\n")
            
            if avg_improvement > 0:
                f.write("• Bayesian LSTM shows better prediction accuracy\n")
            elif avg_improvement < 0:
                f.write("• Standard LSTM shows better prediction accuracy\n")
            else:
                f.write("• Similar prediction accuracy between models\n")
            
            f.write("\n• Bayesian LSTM KEY ADVANTAGES:\n")
            f.write("  - Provides uncertainty quantification\n")
            f.write("  - Enables risk-aware trading decisions\n")
            f.write("  - Can identify high-risk predictions\n")
            f.write("  - Allows position sizing based on confidence\n")
            
            f.write("\n• Standard LSTM characteristics:\n")
            f.write("  - Faster inference (no MC sampling)\n")
            f.write("  - Deterministic predictions\n")
            f.write("  - No uncertainty awareness\n")
            f.write("  - Suitable for baseline comparisons\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Comparison report saved to {report_path}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        print("\n" + "🔬 STARTING COMPREHENSIVE MODEL COMPARISON" + "\n")
        
        # Load data
        self.load_results()
        
        # Compare metrics
        self.compare_overall_performance()
        self.compare_regime_performance()
        self.compare_error_distributions()
        self.analyze_uncertainty_quality()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Generate report
        self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  📊 comparison_results/side_by_side_predictions.png")
        print("  📊 comparison_results/regime_error_comparison.png")
        print("  📊 comparison_results/uncertainty_and_error_analysis.png")
        print("  📊 comparison_results/overall_metrics_comparison.png")
        print("  📄 comparison_results/comparison_report.txt")
        print("\n" + "="*80)


def main():
    """Main comparison execution"""
    
    # Check if result directories exist
    if not os.path.exists("results") or not os.path.exists("results_standard"):
        print("❌ Error: Result directories not found!")
        print("Please run both models first:")
        print("  1. python bayesian_lstm.py")
        print("  2. python lstm.py")
        return
    
    # Create comparison object
    comparator = ModelComparison(
        bayesian_path="results",
        standard_path="results_standard"
    )
    
    # Run full comparison
    comparator.run_full_comparison()
    
    # Summary recommendation
    print("\n💡 RECOMMENDATION:")
    print("-"*80)
    
    overall_df = comparator.comparison_results['overall']
    avg_improvement = overall_df['Improvement_%'].mean()
    
    if abs(avg_improvement) < 2:
        print("Prediction accuracy is similar between models.")
        print("✅ Choose BAYESIAN LSTM for:")
        print("   - Risk management and uncertainty quantification")
        print("   - Confidence-based position sizing")
        print("   - Identifying high-risk predictions")
        print("\n✅ Choose STANDARD LSTM for:")
        print("   - Faster inference requirements")
        print("   - Baseline comparisons")
        print("   - Simple point predictions")
    elif avg_improvement > 0:
        print(f"Bayesian LSTM shows {avg_improvement:.2f}% better accuracy on average.")
        print("✅ RECOMMENDED: Bayesian LSTM")
        print("   - Better predictions + uncertainty quantification")
    else:
        print(f"Standard LSTM shows {abs(avg_improvement):.2f}% better accuracy on average.")
        print("⚖️ Trade-off decision:")
        print("   - Standard LSTM: Better accuracy, no uncertainty")
        print("   - Bayesian LSTM: Slightly lower accuracy, but risk-aware")
    
    print("="*80)


if __name__ == "__main__":
    main()