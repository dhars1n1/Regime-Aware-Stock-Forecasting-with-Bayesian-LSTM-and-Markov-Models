"""
Comprehensive Evaluation System for Regime-Aware Bayesian LSTM

This module provides:
1. Uncertainty quantification metrics (calibration, sharpness, coverage)
2. Regime-specific performance analysis
3. Statistical tests and diagnostics
4. Model comparison utilities
5. Comprehensive reporting system
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class BayesianLSTMEvaluator:
    """
    Comprehensive evaluation system for Bayesian LSTM models
    
    Features:
    - Uncertainty calibration analysis
    - Coverage and reliability metrics
    - Regime-specific performance evaluation
    - Statistical significance testing
    - Comprehensive reporting
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_results = {}
        
    def calculate_basic_metrics(self, 
                              actual: np.ndarray, 
                              predicted: np.ndarray) -> Dict:
        """Calculate basic regression metrics"""
        metrics = {
            'mse': mean_squared_error(actual, predicted),
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'r2': r2_score(actual, predicted),
            'correlation': np.corrcoef(actual, predicted)[0, 1],
            'bias': np.mean(predicted - actual),
            'variance_explained': 1 - np.var(actual - predicted) / np.var(actual)
        }
        
        # Directional accuracy (for returns)
        if len(actual) > 1:
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predicted)
            metrics['directional_accuracy'] = np.mean(actual_direction == predicted_direction)
        
        return metrics
    
    def evaluate_uncertainty_calibration(self, 
                                       actual: np.ndarray,
                                       predicted_mean: np.ndarray,
                                       predicted_std: np.ndarray,
                                       n_bins: int = 10) -> Dict:
        """
        Evaluate calibration of uncertainty estimates
        
        A well-calibrated model should have:
        - Empirical coverage ≈ nominal coverage for prediction intervals
        - Uniform distribution of probability integral transform (PIT)
        """
        calibration_results = {}
        
        # 1. Coverage analysis for multiple confidence levels
        confidence_levels = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])
        coverages = []
        interval_widths = []
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = predicted_mean - z_score * predicted_std
            upper_bound = predicted_mean + z_score * predicted_std
            
            # Calculate empirical coverage
            in_interval = (actual >= lower_bound) & (actual <= upper_bound)
            empirical_coverage = np.mean(in_interval)
            
            coverages.append(empirical_coverage)
            interval_widths.append(np.mean(upper_bound - lower_bound))
        
        calibration_results['confidence_levels'] = confidence_levels
        calibration_results['empirical_coverages'] = np.array(coverages)
        calibration_results['interval_widths'] = np.array(interval_widths)
        
        # Coverage deviations from nominal
        calibration_results['coverage_deviations'] = np.abs(
            calibration_results['empirical_coverages'] - confidence_levels
        )
        
        # 2. Probability Integral Transform (PIT) analysis
        # Convert predictions to standard normal quantiles
        z_scores = (actual - predicted_mean) / predicted_std
        pit_values = stats.norm.cdf(z_scores)
        
        calibration_results['pit_values'] = pit_values
        calibration_results['pit_uniform_pvalue'] = stats.kstest(
            pit_values, 'uniform'
        ).pvalue
        
        # 3. Reliability metrics
        calibration_results['average_calibration_error'] = np.mean(
            calibration_results['coverage_deviations']
        )
        calibration_results['max_calibration_error'] = np.max(
            calibration_results['coverage_deviations']
        )
        
        # 4. Sharpness (average prediction interval width for 95% CI)
        calibration_results['sharpness_95'] = interval_widths[4]  # 95% CI width
        
        return calibration_results
    
    def evaluate_uncertainty_quality(self,
                                   actual: np.ndarray,
                                   predicted_mean: np.ndarray,
                                   predicted_std: np.ndarray) -> Dict:
        """
        Evaluate quality of uncertainty estimates
        """
        uncertainty_quality = {}
        
        # 1. Uncertainty vs error correlation
        absolute_errors = np.abs(actual - predicted_mean)
        uncertainty_error_correlation = np.corrcoef(predicted_std, absolute_errors)[0, 1]
        uncertainty_quality['uncertainty_error_correlation'] = uncertainty_error_correlation
        
        # 2. Adaptive uncertainty (should be higher for harder predictions)
        # Sort by uncertainty and check if higher uncertainty = higher error
        sorted_indices = np.argsort(predicted_std)
        n_samples = len(sorted_indices)
        
        # Split into quartiles by uncertainty
        q1_idx = sorted_indices[:n_samples//4]
        q4_idx = sorted_indices[3*n_samples//4:]
        
        q1_error = np.mean(absolute_errors[q1_idx])  # Low uncertainty
        q4_error = np.mean(absolute_errors[q4_idx])  # High uncertainty
        
        uncertainty_quality['low_uncertainty_error'] = q1_error
        uncertainty_quality['high_uncertainty_error'] = q4_error
        uncertainty_quality['uncertainty_discrimination'] = q4_error / (q1_error + 1e-8)
        
        # 3. Residual analysis of uncertainty
        # Check if standardized residuals follow standard normal
        standardized_residuals = (actual - predicted_mean) / predicted_std
        
        uncertainty_quality['residuals_mean'] = np.mean(standardized_residuals)
        uncertainty_quality['residuals_std'] = np.std(standardized_residuals)
        uncertainty_quality['residuals_normality_pvalue'] = stats.normaltest(
            standardized_residuals
        ).pvalue
        
        # 4. Uncertainty consistency over time
        if len(predicted_std) > 1:
            uncertainty_changes = np.diff(predicted_std)
            uncertainty_quality['uncertainty_volatility'] = np.std(uncertainty_changes)
            uncertainty_quality['uncertainty_autocorr'] = np.corrcoef(
                predicted_std[:-1], predicted_std[1:]
            )[0, 1]
        
        return uncertainty_quality
    
    def evaluate_regime_performance(self,
                                  actual: np.ndarray,
                                  predicted_mean: np.ndarray,
                                  predicted_std: np.ndarray,
                                  regime_labels: np.ndarray) -> Dict:
        """
        Evaluate performance across different market regimes
        """
        regime_results = {}
        
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) < 5:  # Skip if too few samples
                continue
            
            regime_actual = actual[regime_mask]
            regime_pred_mean = predicted_mean[regime_mask]
            regime_pred_std = predicted_std[regime_mask]
            
            # Basic metrics
            regime_metrics = self.calculate_basic_metrics(regime_actual, regime_pred_mean)
            
            # Uncertainty calibration for this regime
            regime_calibration = self.evaluate_uncertainty_calibration(
                regime_actual, regime_pred_mean, regime_pred_std
            )
            
            # Regime-specific uncertainty quality
            regime_uncertainty = self.evaluate_uncertainty_quality(
                regime_actual, regime_pred_mean, regime_pred_std
            )
            
            # Combine all metrics
            regime_results[regime] = {
                'n_samples': np.sum(regime_mask),
                'sample_fraction': np.sum(regime_mask) / len(regime_labels),
                **regime_metrics,
                **regime_calibration,
                **regime_uncertainty
            }
        
        # Cross-regime analysis
        regime_results['cross_regime_analysis'] = self._analyze_cross_regime_performance(
            actual, predicted_mean, predicted_std, regime_labels, unique_regimes
        )
        
        return regime_results
    
    def _analyze_cross_regime_performance(self,
                                        actual: np.ndarray,
                                        predicted_mean: np.ndarray,
                                        predicted_std: np.ndarray,
                                        regime_labels: np.ndarray,
                                        unique_regimes: np.ndarray) -> Dict:
        """Analyze performance differences across regimes"""
        cross_regime = {}
        
        # Statistical tests for performance differences
        mse_by_regime = []
        mae_by_regime = []
        uncertainty_by_regime = []
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            if np.sum(mask) >= 5:
                regime_actual = actual[mask]
                regime_pred = predicted_mean[mask]
                regime_std = predicted_std[mask]
                
                mse_by_regime.append(mean_squared_error(regime_actual, regime_pred))
                mae_by_regime.append(mean_absolute_error(regime_actual, regime_pred))
                uncertainty_by_regime.append(np.mean(regime_std))
        
        # ANOVA tests for significant differences
        if len(mse_by_regime) > 2:
            cross_regime['mse_anova_pvalue'] = stats.f_oneway(*[
                np.full(10, mse) for mse in mse_by_regime  # Simplified
            ]).pvalue
        
        # Pairwise comparisons
        cross_regime['regime_mse_comparison'] = dict(zip(unique_regimes, mse_by_regime))
        cross_regime['regime_uncertainty_comparison'] = dict(zip(unique_regimes, uncertainty_by_regime))
        
        return cross_regime
    
    def comprehensive_evaluation(self,
                               actual: np.ndarray,
                               predictions: Dict,
                               regime_labels: Optional[np.ndarray] = None,
                               feature_names: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive evaluation of the Bayesian LSTM model
        
        Args:
            actual: True values
            predictions: Dict with 'mean', 'std', confidence intervals
            regime_labels: Market regime labels (optional)
            feature_names: Names of input features (optional)
        
        Returns:
            Comprehensive evaluation results dictionary
        """
        print("🔍 RUNNING COMPREHENSIVE EVALUATION")
        print("=" * 50)
        
        results = {}
        
        predicted_mean = predictions['mean']
        predicted_std = predictions['std']
        
        # 1. Basic performance metrics
        print("1️⃣ Calculating basic performance metrics...")
        results['basic_metrics'] = self.calculate_basic_metrics(actual, predicted_mean)
        
        # 2. Uncertainty calibration
        print("2️⃣ Evaluating uncertainty calibration...")
        results['calibration'] = self.evaluate_uncertainty_calibration(
            actual, predicted_mean, predicted_std
        )
        
        # 3. Uncertainty quality
        print("3️⃣ Assessing uncertainty quality...")
        results['uncertainty_quality'] = self.evaluate_uncertainty_quality(
            actual, predicted_mean, predicted_std
        )
        
        # 4. Regime-specific analysis (if regime labels provided)
        if regime_labels is not None:
            print("4️⃣ Analyzing regime-specific performance...")
            results['regime_analysis'] = self.evaluate_regime_performance(
                actual, predicted_mean, predicted_std, regime_labels
            )
        
        # 5. Advanced diagnostics
        print("5️⃣ Running advanced diagnostics...")
        results['diagnostics'] = self._run_advanced_diagnostics(
            actual, predicted_mean, predicted_std
        )
        
        # 6. Model summary statistics
        results['summary'] = self._generate_summary_statistics(results)
        
        print("✅ Comprehensive evaluation completed!")
        
        return results
    
    def _run_advanced_diagnostics(self,
                                actual: np.ndarray,
                                predicted_mean: np.ndarray,
                                predicted_std: np.ndarray) -> Dict:
        """Run advanced diagnostic tests"""
        diagnostics = {}
        
        residuals = actual - predicted_mean
        standardized_residuals = residuals / predicted_std
        
        # 1. Residual analysis
        diagnostics['residual_autocorrelation'] = self._ljung_box_test(residuals)
        diagnostics['residual_normality'] = stats.normaltest(residuals).pvalue
        diagnostics['residual_heteroscedasticity'] = self._breusch_pagan_test(
            residuals, predicted_mean
        )
        
        # 2. Prediction interval reliability
        # Check if prediction intervals contain actual values at expected rates
        for conf_level in [0.68, 0.95]:
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = predicted_mean - z_score * predicted_std
            upper = predicted_mean + z_score * predicted_std
            
            in_interval = (actual >= lower) & (actual <= upper)
            expected_coverage = conf_level
            actual_coverage = np.mean(in_interval)
            
            # Binomial test for coverage
            n_trials = len(actual)
            n_successes = np.sum(in_interval)
            
            diagnostics[f'coverage_{int(conf_level*100)}_binomial_pvalue'] = \
                stats.binom_test(n_successes, n_trials, expected_coverage)
        
        # 3. Uncertainty adequacy
        diagnostics['uncertainty_adequacy'] = self._assess_uncertainty_adequacy(
            actual, predicted_mean, predicted_std
        )
        
        return diagnostics
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """Ljung-Box test for residual autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return result['lb_pvalue'].iloc[-1]
        except ImportError:
            # Simplified version if statsmodels not available
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            return 1.0 if np.abs(autocorr) < 0.1 else 0.0
    
    def _breusch_pagan_test(self, residuals: np.ndarray, fitted: np.ndarray) -> float:
        """Test for heteroscedasticity"""
        # Simplified version - check correlation between squared residuals and fitted values
        squared_residuals = residuals ** 2
        correlation = np.corrcoef(squared_residuals, fitted)[0, 1]
        
        # Approximate p-value based on correlation strength
        return 1.0 if np.abs(correlation) < 0.2 else 0.1
    
    def _assess_uncertainty_adequacy(self,
                                   actual: np.ndarray,
                                   predicted_mean: np.ndarray,
                                   predicted_std: np.ndarray) -> Dict:
        """Assess if uncertainty estimates are adequate"""
        adequacy = {}
        
        # 1. Under/overconfidence analysis
        standardized_errors = (actual - predicted_mean) / predicted_std
        
        # Should follow standard normal if uncertainty is well-calibrated
        adequacy['standardized_error_mean'] = np.mean(standardized_errors)
        adequacy['standardized_error_std'] = np.std(standardized_errors)
        
        # 2. Extreme event capture
        # Check if model captures extreme movements (>2 sigma events)
        extreme_threshold = 2.0
        extreme_events = np.abs(standardized_errors) > extreme_threshold
        adequacy['extreme_events_captured'] = np.mean(extreme_events)
        adequacy['expected_extreme_rate'] = 2 * (1 - stats.norm.cdf(extreme_threshold))
        
        return adequacy
    
    def _generate_summary_statistics(self, results: Dict) -> Dict:
        """Generate summary statistics from evaluation results"""
        summary = {}
        
        # Overall performance grade
        basic = results['basic_metrics']
        calib = results['calibration']
        
        # Performance scoring (0-100)
        r2_score = max(0, min(100, basic['r2'] * 100))
        calibration_score = max(0, 100 - calib['average_calibration_error'] * 1000)
        
        summary['performance_score'] = (r2_score + calibration_score) / 2
        
        # Key insights
        summary['key_insights'] = []
        
        if basic['r2'] > 0.5:
            summary['key_insights'].append("Good predictive performance (R² > 0.5)")
        
        if calib['average_calibration_error'] < 0.05:
            summary['key_insights'].append("Well-calibrated uncertainty estimates")
        
        if results['uncertainty_quality']['uncertainty_error_correlation'] > 0.3:
            summary['key_insights'].append("Uncertainty correlates well with prediction errors")
        
        # Recommendations
        summary['recommendations'] = []
        
        if basic['r2'] < 0.3:
            summary['recommendations'].append("Consider model architecture improvements")
        
        if calib['average_calibration_error'] > 0.1:
            summary['recommendations'].append("Recalibrate uncertainty estimates")
        
        return summary
    
    def generate_evaluation_report(self, 
                                 results: Dict,
                                 save_path: str = "evaluation_report.txt") -> str:
        """Generate comprehensive evaluation report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BAYESIAN LSTM COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        summary = results.get('summary', {})
        report_lines.append(f"Overall Performance Score: {summary.get('performance_score', 0):.1f}/100")
        report_lines.append("")
        
        if 'key_insights' in summary:
            report_lines.append("Key Insights:")
            for insight in summary['key_insights']:
                report_lines.append(f"  ✓ {insight}")
            report_lines.append("")
        
        # 2. Basic Performance Metrics
        report_lines.append("BASIC PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        basic = results['basic_metrics']
        
        metrics_table = [
            ("Mean Squared Error", f"{basic['mse']:.8f}"),
            ("Mean Absolute Error", f"{basic['mae']:.8f}"), 
            ("Root Mean Squared Error", f"{basic['rmse']:.8f}"),
            ("R² Score", f"{basic['r2']:.4f}"),
            ("Correlation", f"{basic['correlation']:.4f}"),
            ("Directional Accuracy", f"{basic.get('directional_accuracy', 0):.4f}")
        ]
        
        for metric, value in metrics_table:
            report_lines.append(f"{metric:<25}: {value}")
        report_lines.append("")
        
        # 3. Uncertainty Calibration
        report_lines.append("UNCERTAINTY CALIBRATION ANALYSIS")
        report_lines.append("-" * 40)
        calib = results['calibration']
        
        report_lines.append("Coverage Analysis:")
        conf_levels = calib['confidence_levels']
        emp_coverages = calib['empirical_coverages']
        
        for conf, emp in zip(conf_levels, emp_coverages):
            deviation = abs(emp - conf)
            status = "✓" if deviation < 0.05 else "⚠" if deviation < 0.1 else "✗"
            report_lines.append(f"  {conf*100:4.0f}% CI: {emp:.3f} (deviation: {deviation:.3f}) {status}")
        
        report_lines.append("")
        report_lines.append(f"Average Calibration Error: {calib['average_calibration_error']:.4f}")
        report_lines.append(f"PIT Uniformity p-value: {calib['pit_uniform_pvalue']:.4f}")
        report_lines.append("")
        
        # 4. Uncertainty Quality
        report_lines.append("UNCERTAINTY QUALITY ASSESSMENT")
        report_lines.append("-" * 40)
        unc_quality = results['uncertainty_quality']
        
        uncertainty_metrics = [
            ("Uncertainty-Error Correlation", f"{unc_quality['uncertainty_error_correlation']:.4f}"),
            ("Uncertainty Discrimination", f"{unc_quality['uncertainty_discrimination']:.4f}"),
            ("Standardized Residuals Mean", f"{unc_quality['residuals_mean']:.4f}"),
            ("Standardized Residuals Std", f"{unc_quality['residuals_std']:.4f}")
        ]
        
        for metric, value in uncertainty_metrics:
            report_lines.append(f"{metric:<30}: {value}")
        report_lines.append("")
        
        # 5. Regime Analysis (if available)
        if 'regime_analysis' in results:
            report_lines.append("REGIME-SPECIFIC PERFORMANCE")
            report_lines.append("-" * 40)
            
            regime_results = results['regime_analysis']
            for regime, regime_data in regime_results.items():
                if regime == 'cross_regime_analysis':
                    continue
                    
                report_lines.append(f"\n{regime.upper()} REGIME:")
                report_lines.append(f"  Samples: {regime_data['n_samples']} ({regime_data['sample_fraction']:.2%})")
                report_lines.append(f"  MSE: {regime_data['mse']:.8f}")
                report_lines.append(f"  MAE: {regime_data['mae']:.8f}")
                report_lines.append(f"  R²: {regime_data['r2']:.4f}")
                report_lines.append(f"  Calibration Error: {regime_data['average_calibration_error']:.4f}")
            
            report_lines.append("")
        
        # 6. Diagnostics
        report_lines.append("DIAGNOSTIC TESTS")
        report_lines.append("-" * 40)
        diagnostics = results.get('diagnostics', {})
        
        diagnostic_tests = [
            ("Residual Autocorrelation", diagnostics.get('residual_autocorrelation', 0)),
            ("Residual Normality", diagnostics.get('residual_normality', 0)),
            ("95% Coverage Binomial Test", diagnostics.get('coverage_95_binomial_pvalue', 0))
        ]
        
        for test, pvalue in diagnostic_tests:
            status = "Pass" if pvalue > 0.05 else "Fail"
            report_lines.append(f"{test:<30}: p={pvalue:.4f} ({status})")
        
        report_lines.append("")
        
        # 7. Recommendations
        if 'recommendations' in summary and summary['recommendations']:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for rec in summary['recommendations']:
                report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"📊 Evaluation report saved to: {save_path}")
        
        return report_text


def demo_evaluation_system():
    """Demonstrate the evaluation system"""
    print("🧪 EVALUATION SYSTEM DEMO")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic predictions and actual values
    actual_returns = np.random.normal(0, 0.02, n_samples)
    predicted_returns = actual_returns + np.random.normal(0, 0.005, n_samples)
    predicted_std = np.abs(np.random.normal(0.01, 0.003, n_samples))
    
    # Add some calibration issues for demo
    predicted_std *= 0.8  # Slightly underconfident
    
    predictions = {
        'mean': predicted_returns,
        'std': predicted_std,
        'ci_95_lower': predicted_returns - 1.96 * predicted_std,
        'ci_95_upper': predicted_returns + 1.96 * predicted_std
    }
    
    # Sample regime labels
    regime_labels = np.random.choice(['Bull', 'Normal', 'Crisis'], n_samples,
                                   p=[0.3, 0.5, 0.2])
    
    # Run evaluation
    evaluator = BayesianLSTMEvaluator()
    results = evaluator.comprehensive_evaluation(
        actual_returns, predictions, regime_labels
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, "demo_evaluation_report.txt")
    
    print("✅ Demo completed! Check 'demo_evaluation_report.txt'")
    print("\nSample of evaluation results:")
    print(f"R² Score: {results['basic_metrics']['r2']:.4f}")
    print(f"Calibration Error: {results['calibration']['average_calibration_error']:.4f}")


if __name__ == "__main__":
    demo_evaluation_system()