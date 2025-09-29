"""
Complete End-to-End Pipeline for Regime-Aware Bayesian LSTM Stock Forecasting

This script orchestrates the entire machine learning pipeline:
1. Data loading and preprocessing  
2. Model training with monitoring
3. Evaluation and diagnostics
4. Prediction generation
5. Comprehensive visualization
6. Results saving and reporting

Usage:
    python run_complete_pipeline.py --data_path data/data_with_regimes.csv --config_path config.yaml
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add models directory to path for imports
sys.path.append('models')
sys.path.append('data')
sys.path.append('evaluation')

# Import our custom modules (with graceful TensorFlow handling)
try:
    # First test if TensorFlow can be imported directly
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
    
    # Now import our TensorFlow-dependent modules
    from bayesian_lstm import BayesianLSTM
    from train_model import BayesianLSTMTrainer
    from prediction_engine import BayesianLSTMPredictor
    from visualization_system import BayesianLSTMVisualizer
    from evaluation_metrics import BayesianLSTMEvaluator
    TF_AVAILABLE = True
    print("✅ All TensorFlow-dependent modules loaded successfully")
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    print("   Configuration creation will still work, but training requires TensorFlow")
    TF_AVAILABLE = False

# Always try to import data processor (no TensorFlow dependency)
try:
    from data_processor import RegimeAwareDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Data processor not available")
    DATA_PROCESSOR_AVAILABLE = False


class BayesianLSTMPipeline:
    """
    Complete pipeline for Regime-Aware Bayesian LSTM Stock Forecasting
    
    This class orchestrates the entire machine learning workflow:
    - Data processing and feature engineering
    - Model training with uncertainty quantification
    - Comprehensive evaluation and diagnostics
    - Prediction generation with confidence intervals
    - Visualization and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration"""
        self.config = config
        self.setup_directories()
        
        # Initialize components
        self.data_processor = None
        self.model = None
        self.trainer = None
        self.predictor = None
        self.visualizer = None
        self.evaluator = None
        
        # Results storage
        self.results = {}
        
        print("🚀 BAYESIAN LSTM PIPELINE INITIALIZED")
        print("=" * 60)
    
    def setup_directories(self):
        """Create necessary directories for results"""
        directories = [
            self.config['output_dir'],
            os.path.join(self.config['output_dir'], 'models'),
            os.path.join(self.config['output_dir'], 'predictions'), 
            os.path.join(self.config['output_dir'], 'visualizations'),
            os.path.join(self.config['output_dir'], 'evaluation'),
            os.path.join(self.config['output_dir'], 'reports')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.config['output_dir']}")
    
    def load_and_preprocess_data(self):
        """Load data and perform preprocessing"""
        print("\n1️⃣ DATA LOADING AND PREPROCESSING")
        print("-" * 40)
        
        # Load data
        print(f"📊 Loading data from: {self.config['data_path']}")
        data = pd.read_csv(self.config['data_path'])
        
        # Initialize data processor
        self.data_processor = RegimeAwareDataProcessor(
            sequence_length=self.config['model']['sequence_length'],
            feature_columns=self.config['data']['feature_columns'],
            target_column=self.config['data']['target_column'],
            regime_column=self.config['data']['regime_column'],
            exclude_columns=self.config['data']['exclude_columns']
        )
        
        # Auto-detect features if needed
        if self.config['data']['feature_columns'] == 'auto':
            feature_cols = self.data_processor.get_feature_columns(data)
            print(f"✨ Auto-detected {len(feature_cols)} features:")
            print(f"   First 10: {feature_cols[:10]}")
            if len(feature_cols) > 10:
                print(f"   ... and {len(feature_cols)-10} more features")
            self.config['data']['feature_columns'] = feature_cols
        
        # Process data
        print("🔧 Engineering features and creating sequences...")
        processed_data = self.data_processor.prepare_data_for_training(
            data, 
            test_size=self.config['data']['test_size']
        )
        
        self.processed_data = processed_data
        
        # Print data summary
        print(f"✅ Data preprocessing completed:")
        print(f"   Training samples: {len(processed_data['X_train'])}")
        print(f"   Test samples: {len(processed_data['X_test'])}")
        print(f"   Feature dimension: {processed_data['X_train'].shape}")
        print(f"   Unique regimes: {len(np.unique(processed_data['regime_labels_train']))}")
        
        # Save processed data info
        data_info = {
            'training_samples': len(processed_data['X_train']),
            'test_samples': len(processed_data['X_test']),
            'feature_shape': processed_data['X_train'].shape,
            'regimes': list(np.unique(processed_data['regime_labels_train'])),
            'preprocessing_completed': str(datetime.now())
        }
        
        with open(os.path.join(self.config['output_dir'], 'data_info.json'), 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
    
    def train_model(self):
        """Train the Bayesian LSTM model"""
        print("\n2️⃣ MODEL TRAINING")
        print("-" * 40)
        
        # Initialize model  
        n_features = self.processed_data['X_train'].shape[2]  # Get actual feature count from data
        input_shape = (self.config['model']['sequence_length'], n_features)
        
        print(f"🏗️ Model input shape: {input_shape}")
        print(f"   Sequence length: {input_shape[0]}")
        print(f"   Number of features: {input_shape[1]}")
        
        self.model = BayesianLSTM(
            input_shape=input_shape,
            lstm_units=self.config['model']['lstm_units'],
            dropout_rate=self.config['model']['dropout_rate'],
            n_monte_carlo=self.config['model']['n_monte_carlo']
        )
        
        # Build model
        print("🏗️ Building model architecture...")
        model_arch = self.model.build_model()
        print(f"   LSTM units: {self.config['model']['lstm_units']}")
        print(f"   Dropout rate: {self.config['model']['dropout_rate']}")
        print(f"   Monte Carlo samples: {self.config['model']['n_monte_carlo']}")
        
        # Initialize trainer
        self.trainer = BayesianLSTMTrainer(
            model=self.model,
            save_dir=os.path.join(self.config['output_dir'], 'models')
        )
        
        # Train model
        print("🎯 Starting training...")
        training_history = self.trainer.run_training_pipeline(
            X_train=self.processed_data['X_train'],
            y_train=self.processed_data['y_train'],
            X_val=self.processed_data['X_test'][:len(self.processed_data['X_test'])//2],
            y_val=self.processed_data['y_test'][:len(self.processed_data['y_test'])//2],
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate']
        )
        
        self.results['training_history'] = training_history
        print("✅ Model training completed!")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n3️⃣ MODEL EVALUATION")
        print("-" * 40)
        
        # Initialize predictor for evaluation
        self.predictor = BayesianLSTMPredictor(self.model)
        
        # Generate predictions on test set
        print("🔮 Generating predictions on test set...")
        predictions = self.predictor.predict_sequence(
            self.processed_data['X_test'],
            n_monte_carlo=self.config['model']['n_monte_carlo']
        )
        
        # Initialize evaluator
        self.evaluator = BayesianLSTMEvaluator()
        
        # Run comprehensive evaluation
        print("📊 Running comprehensive evaluation...")
        evaluation_results = self.evaluator.comprehensive_evaluation(
            actual=self.processed_data['y_test'],
            predictions=predictions,
            regime_labels=self.processed_data.get('regime_labels_test'),
            feature_names=self.config['data']['feature_columns']
        )
        
        self.results['evaluation'] = evaluation_results
        self.results['predictions'] = predictions
        
        # Generate evaluation report
        report_path = os.path.join(self.config['output_dir'], 'evaluation', 'comprehensive_report.txt')
        self.evaluator.generate_evaluation_report(evaluation_results, report_path)
        
        # Print key results
        basic = evaluation_results['basic_metrics']
        calib = evaluation_results['calibration']
        
        print(f"✅ Evaluation completed:")
        print(f"   R² Score: {basic['r2']:.4f}")
        print(f"   RMSE: {basic['rmse']:.6f}")
        print(f"   Calibration Error: {calib['average_calibration_error']:.4f}")
        print(f"   95% Coverage: {calib['empirical_coverages'][4]:.3f}")
    
    def generate_predictions(self):
        """Generate predictions for future periods"""
        print("\n4️⃣ PREDICTION GENERATION")
        print("-" * 40)
        
        # Generate one-step-ahead predictions
        print("🔮 Generating one-step-ahead predictions...")
        
        # Use last sequence for future prediction
        last_sequence = self.processed_data['X_test'][-1:] 
        
        future_prediction = self.predictor.predict_next_return(
            last_sequence,
            n_monte_carlo=self.config['prediction']['n_monte_carlo']
        )
        
        # Generate multi-step predictions if requested
        if self.config['prediction']['multi_step_horizon'] > 1:
            print(f"🔮 Generating {self.config['prediction']['multi_step_horizon']}-step predictions...")
            multi_step_predictions = self.predictor.multi_step_prediction(
                last_sequence,
                n_steps=self.config['prediction']['multi_step_horizon'],
                n_monte_carlo=self.config['prediction']['n_monte_carlo']
            )
            self.results['multi_step_predictions'] = multi_step_predictions
        
        # Analyze prediction confidence
        confidence_analysis = self.predictor.analyze_prediction_confidence(
            self.results['predictions']
        )
        
        self.results['future_prediction'] = future_prediction
        self.results['confidence_analysis'] = confidence_analysis
        
        # Save predictions
        predictions_data = {
            'test_predictions': {
                'mean': self.results['predictions']['mean'].tolist(),
                'std': self.results['predictions']['std'].tolist(),
                'ci_95_lower': self.results['predictions']['ci_95_lower'].tolist(),
                'ci_95_upper': self.results['predictions']['ci_95_upper'].tolist()
            },
            'future_prediction': {
                'mean': float(future_prediction['mean']),
                'std': float(future_prediction['std']),
                'ci_95_lower': float(future_prediction['ci_95_lower']),
                'ci_95_upper': float(future_prediction['ci_95_upper'])
            },
            'confidence_analysis': confidence_analysis,
            'generation_time': str(datetime.now())
        }
        
        predictions_path = os.path.join(self.config['output_dir'], 'predictions', 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2, default=str)
        
        print(f"✅ Predictions saved to: {predictions_path}")
        print(f"   Next period prediction: {future_prediction['mean']:.6f} ± {future_prediction['std']:.6f}")
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n5️⃣ VISUALIZATION GENERATION")
        print("-" * 40)
        
        # Initialize visualizer
        self.visualizer = BayesianLSTMVisualizer(
            save_dir=os.path.join(self.config['output_dir'], 'visualizations')
        )
        
        # Create dates for test period (assuming daily data)
        test_dates = pd.date_range(
            start='2023-01-01',  # Placeholder - should be actual dates
            periods=len(self.processed_data['y_test']),
            freq='D'
        )
        
        # Generate all visualizations
        print("🎨 Creating comprehensive visualizations...")
        self.visualizer.generate_all_visualizations(
            dates=test_dates,
            predictions=self.results['predictions'],
            evaluation_results=self.results['evaluation'],
            actual_values=self.processed_data['y_test'],
            regime_labels=self.processed_data.get('regime_labels_test')
        )
        
        print("✅ Visualizations completed!")
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print("\n6️⃣ FINAL REPORT GENERATION")
        print("-" * 40)
        
        # Compile final report
        report = {
            'pipeline_summary': {
                'execution_time': str(datetime.now()),
                'data_path': self.config['data_path'],
                'output_directory': self.config['output_dir'],
                'model_configuration': self.config['model'],
                'training_configuration': self.config['training']
            },
            'data_summary': {
                'training_samples': len(self.processed_data['X_train']),
                'test_samples': len(self.processed_data['X_test']),
                'features': self.config['data']['feature_columns'],
                'target': self.config['data']['target_column'],
                'regime_column': self.config['data']['regime_column']
            },
            'performance_summary': {
                'r2_score': self.results['evaluation']['basic_metrics']['r2'],
                'rmse': self.results['evaluation']['basic_metrics']['rmse'],
                'mae': self.results['evaluation']['basic_metrics']['mae'],
                'calibration_error': self.results['evaluation']['calibration']['average_calibration_error'],
                'coverage_95': self.results['evaluation']['calibration']['empirical_coverages'][4],
                'uncertainty_quality': self.results['evaluation']['uncertainty_quality']['uncertainty_error_correlation']
            },
            'key_findings': self.results['evaluation']['summary'].get('key_insights', []),
            'recommendations': self.results['evaluation']['summary'].get('recommendations', []),
            'files_generated': {
                'model_weights': 'models/bayesian_lstm_model.h5',
                'evaluation_report': 'evaluation/comprehensive_report.txt', 
                'predictions': 'predictions/predictions.json',
                'visualizations': 'visualizations/',
                'training_history': 'models/training_history.json'
            }
        }
        
        # Save final report
        report_path = os.path.join(self.config['output_dir'], 'reports', 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_path = os.path.join(self.config['output_dir'], 'reports', 'executive_summary.txt')
        self._create_executive_summary(report, summary_path)
        
        print(f"✅ Final reports generated:")
        print(f"   Detailed report: {report_path}")
        print(f"   Executive summary: {summary_path}")
    
    def _create_executive_summary(self, report: Dict, save_path: str):
        """Create executive summary report"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("REGIME-AWARE BAYESIAN LSTM STOCK FORECASTING")
        summary_lines.append("EXECUTIVE SUMMARY REPORT") 
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Project overview
        summary_lines.append("PROJECT OVERVIEW")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Execution Date: {report['pipeline_summary']['execution_time']}")
        summary_lines.append(f"Data Source: {report['pipeline_summary']['data_path']}")
        summary_lines.append(f"Training Samples: {report['data_summary']['training_samples']:,}")
        summary_lines.append(f"Test Samples: {report['data_summary']['test_samples']:,}")
        summary_lines.append("")
        
        # Model configuration
        summary_lines.append("MODEL CONFIGURATION")
        summary_lines.append("-" * 40)
        model_config = report['pipeline_summary']['model_configuration']
        summary_lines.append(f"Architecture: Bayesian LSTM with Monte Carlo Dropout")
        summary_lines.append(f"LSTM Units: {model_config['lstm_units']}")
        summary_lines.append(f"Sequence Length: {model_config['sequence_length']}")
        summary_lines.append(f"Dropout Rate: {model_config['dropout_rate']}")
        summary_lines.append(f"Monte Carlo Samples: {model_config['n_monte_carlo']}")
        summary_lines.append("")
        
        # Performance summary
        summary_lines.append("PERFORMANCE SUMMARY")
        summary_lines.append("-" * 40)
        perf = report['performance_summary']
        summary_lines.append(f"R² Score: {perf['r2_score']:.4f}")
        summary_lines.append(f"RMSE: {perf['rmse']:.6f}")
        summary_lines.append(f"MAE: {perf['mae']:.6f}")
        summary_lines.append(f"95% Coverage: {perf['coverage_95']:.3f} (Target: 0.950)")
        summary_lines.append(f"Calibration Error: {perf['calibration_error']:.4f}")
        summary_lines.append("")
        
        # Key insights
        if report.get('key_findings'):
            summary_lines.append("KEY FINDINGS")
            summary_lines.append("-" * 40)
            for finding in report['key_findings']:
                summary_lines.append(f"✓ {finding}")
            summary_lines.append("")
        
        # Recommendations
        if report.get('recommendations'):
            summary_lines.append("RECOMMENDATIONS")
            summary_lines.append("-" * 40)
            for rec in report['recommendations']:
                summary_lines.append(f"• {rec}")
            summary_lines.append("")
        
        # Files generated
        summary_lines.append("GENERATED FILES")
        summary_lines.append("-" * 40)
        files = report['files_generated']
        summary_lines.append(f"📊 Model: {files['model_weights']}")
        summary_lines.append(f"📈 Predictions: {files['predictions']}")
        summary_lines.append(f"📋 Evaluation: {files['evaluation_report']}")
        summary_lines.append(f"🎨 Visualizations: {files['visualizations']}")
        summary_lines.append("")
        
        summary_lines.append("=" * 80)
        
        # Save summary
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary_lines))
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        start_time = datetime.now()
        
        print("🚀 STARTING COMPLETE BAYESIAN LSTM PIPELINE")
        print("=" * 80)
        
        try:
            # Execute pipeline steps
            self.load_and_preprocess_data()
            self.train_model()
            self.evaluate_model()
            self.generate_predictions()
            self.create_visualizations()
            self.generate_final_report()
            
            # Calculate total execution time
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Total Execution Time: {execution_time}")
            print(f"Results Directory: {self.config['output_dir']}")
            print("Check the 'reports' directory for comprehensive summaries.")
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            raise


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults"""
    
    default_config = {
        'data_path': 'data/data_with_regimes.csv',
        'output_dir': 'results',
        'data': {
            'feature_columns': 'auto',  # Will auto-detect all available features
            'exclude_columns': ['Date', 'regime_viterbi', 'regime_0_prob', 'regime_1_prob', 'regime_2_prob'],  # Columns to exclude
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
        },
        'prediction': {
            'n_monte_carlo': 200,
            'multi_step_horizon': 5
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge with defaults (user config overrides)
        def merge_dicts(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
        
        merge_dicts(default_config, user_config)
    
    return default_config


def create_sample_config(path: str = "config.yaml"):
    """Create a sample configuration file"""
    sample_config = {
        'data_path': 'data/data_with_regimes.csv',
        'output_dir': 'results',
        'data': {
            'feature_columns': 'auto',  # Will auto-detect all available features
            'exclude_columns': ['Date', 'regime_viterbi', 'regime_0_prob', 'regime_1_prob', 'regime_2_prob'],  # Columns to exclude
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
        },
        'prediction': {
            'n_monte_carlo': 200,
            'multi_step_horizon': 5
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Sample configuration saved to: {path}")


def show_tensorflow_fix_guide():
    """Show comprehensive TensorFlow installation guide"""
    print("🔧 TENSORFLOW INSTALLATION FIX GUIDE")
    print("=" * 60)
    
    print("\n🔍 COMMON CAUSES & SOLUTIONS:")
    print("-" * 40)
    
    print("\n1️⃣ PYTHON VERSION COMPATIBILITY")
    print("   TensorFlow 2.15+ requires Python 3.9-3.12")
    print("   Check your version: python --version")
    
    print("\n2️⃣ REINSTALL TENSORFLOW (RECOMMENDED)")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow==2.15.0")
    
    print("\n3️⃣ VISUAL C++ REDISTRIBUTABLES")
    print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   Install Microsoft Visual C++ 2015-2022 Redistributable")
    
    print("\n4️⃣ CPU-ONLY VERSION (IF GPU ISSUES)")
    print("   pip uninstall tensorflow tensorflow-gpu")
    print("   pip install tensorflow-cpu==2.15.0")
    
    print("\n5️⃣ CLEAN REINSTALL (NUCLEAR OPTION)")
    print("   pip uninstall tensorflow tensorflow-gpu tensorflow-cpu")
    print("   pip cache purge")
    print("   pip install tensorflow==2.15.0")
    
    print("\n6️⃣ ALTERNATIVE: USE CONDA")
    print("   conda create -n tf_env python=3.11")
    print("   conda activate tf_env")
    print("   conda install tensorflow")
    
    print("\n7️⃣ CHECK INSTALLATION")
    print("   python -c \"import tensorflow as tf; print(tf.__version__)\"")
    
    print("\n🎯 QUICK FIX (TRY FIRST):")
    print("-" * 30)
    print("pip install --upgrade --force-reinstall tensorflow==2.15.0")
    
    print("\n✅ AFTER FIXING:")
    print("-" * 20)
    print("python run_complete_pipeline.py --create_config")
    print("python run_complete_pipeline.py")
    
    print("\n📚 MORE HELP:")
    print("   https://www.tensorflow.org/install/errors")
    print("   https://github.com/tensorflow/tensorflow/issues")
    print("\n" + "=" * 60)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Regime-Aware Bayesian LSTM Pipeline")
    parser.add_argument('--data_path', type=str, help='Path to input data CSV')
    parser.add_argument('--config_path', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--create_config', action='store_true', help='Create sample config file')
    parser.add_argument('--fix_tensorflow', action='store_true', help='Show TensorFlow installation help')
    
    args = parser.parse_args()
    
    # Show TensorFlow installation help
    if args.fix_tensorflow:
        show_tensorflow_fix_guide()
        return
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    # Check TensorFlow availability for training
    if not TF_AVAILABLE:
        print("❌ TensorFlow is not properly installed!")
        print("   Use --fix_tensorflow to see installation guide")
        print("   You can still create config files, but training requires TensorFlow")
        return
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Override with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Validate data path
    if not os.path.exists(config['data_path']):
        print(f"❌ Data file not found: {config['data_path']}")
        print("Please check the path or use --create_config to see sample configuration")
        return
    
    # Initialize and run pipeline
    pipeline = BayesianLSTMPipeline(config)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
