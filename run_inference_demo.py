"""
🎯 Demo Script: Complete Bayesian LSTM Inference Pipeline
========================================================

This script demonstrates how to:
1. Load a trained Bayesian LSTM model
2. Perform Monte Carlo Dropout inference
3. Evaluate predictions with comprehensive metrics
4. Generate detailed visualizations

Run this script after training your model with lstm/lstm.py

Requirements:
- Trained model artifacts in results/ directory
- Test data with same features as training

Author: AI Assistant
Date: October 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append('.')
sys.path.append('lstm')

# Import our inference engine
from bayesian_lstm_inference import BayesianLSTMInferenceEngine

def check_required_files():
    """
    Check if all required model artifacts exist
    """
    required_files = [
        "results/bayesian_lstm_model.h5",
        "results/scalers.pkl", 
        "results/regime_encoder.pkl",
        "results/model_metadata.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Run lstm/lstm.py first to train the model and save artifacts")
        return False
    
    print("✅ All required files found!")
    return True

def run_inference_demo():
    """
    Main demo function
    """
    print("🎯 Bayesian LSTM Inference Demo")
    print("=" * 50)
    
    # Check if model artifacts exist
    if not check_required_files():
        return
    
    # Initialize inference engine
    print("\n🔄 Initializing inference engine...")
    engine = BayesianLSTMInferenceEngine(model_path="results")
    
    try:
        # Load all artifacts
        engine.load_artifacts()
        
        # Check if test data exists
        data_path = "data/data_with_regimes.csv"
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure your data file exists with regime information")
            return
        
        print(f"\n📊 Running inference on: {data_path}")
        
        # Run complete inference pipeline
        # This will:
        # 1. Load and prepare data
        # 2. Perform MC Dropout inference  
        # 3. Evaluate predictions
        # 4. Create visualizations
        # 5. Save results
        
        results = engine.run_complete_inference(
            data_path=data_path,
            start_date="2023-01-01",  # Adjust based on your data
            end_date=None,            # Use all data from start_date
            save_results=True
        )
        
        # Display key results
        if results['evaluation']:
            print(f"\n🎉 INFERENCE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            eval_metrics = results['evaluation']
            print(f"📈 PERFORMANCE SUMMARY:")
            print(f"   Mean Absolute Error: {eval_metrics['mae']:.6f}")
            print(f"   Root Mean Square Error: {eval_metrics['rmse']:.6f}")
            print(f"   R-Squared: {eval_metrics['r2']:.4f}")
            print(f"   95% Coverage: {eval_metrics['coverage_95']:.3f}")
            print(f"   Hit Rate (Directional): {eval_metrics['hit_rate']:.3f}")
            print(f"   Information Coefficient: {eval_metrics['information_coefficient']:.4f}")
            
            # Regime-specific results
            if engine.regime_analysis:
                print(f"\n📊 REGIME-SPECIFIC PERFORMANCE:")
                for regime, metrics in engine.regime_analysis.items():
                    print(f"   {regime} Regime:")
                    print(f"     Samples: {metrics['count']}")
                    print(f"     MAE: {metrics['mae']:.6f}")
                    print(f"     95% Coverage: {metrics['coverage_95']:.3f}")
                    print(f"     Hit Rate: {metrics['hit_rate']:.3f}")
            
            print(f"\n📁 OUTPUTS SAVED TO:")
            print(f"   - results/inference_results.json (complete results)")
            print(f"   - results/inference_predictions.csv (detailed predictions)")
            print(f"   - results/inference_plots/ (visualizations)")
            
        else:
            print("⚠ No evaluation performed (missing target values)")
        
        # Show sample predictions
        if 'predictions' in results:
            pred_data = results['predictions']
            n_samples = min(10, len(pred_data['mean']))
            
            print(f"\n🔮 SAMPLE PREDICTIONS (Last {n_samples}):")
            print("-" * 80)
            print(f"{'Date':<12} {'Mean':<10} {'Std':<10} {'Lower_CI':<10} {'Upper_CI':<10}")
            print("-" * 80)
            
            for i in range(-n_samples, 0):
                date = pd.to_datetime(pred_data['dates'][i]).strftime('%Y-%m-%d')
                mean = pred_data['mean'][i]
                std = pred_data['std'][i]
                lower = pred_data['lower_ci_95'][i]
                upper = pred_data['upper_ci_95'][i]
                
                print(f"{date:<12} {mean:<10.6f} {std:<10.6f} {lower:<10.6f} {upper:<10.6f}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✅ Demo completed successfully!")
    print(f"Check the results/ directory for detailed outputs.")

def run_single_prediction_demo():
    """
    Demo: Single day prediction with uncertainty
    """
    print("\n🎯 Single Prediction Demo")
    print("=" * 30)
    
    # Initialize engine
    engine = BayesianLSTMInferenceEngine(model_path="results")
    engine.load_artifacts()
    
    # Load recent data
    df = pd.read_csv("data/data_with_regimes.csv", parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # Get last 30 days for prediction
    recent_data = df.tail(30)
    
    print(f"Using data from {recent_data.index[0]} to {recent_data.index[-1]}")
    
    # Prepare data and make prediction
    X, dates, _ = engine.prepare_inference_data(recent_data)
    
    # Get prediction for most recent sequence
    latest_sequence = X[-1:]
    prediction = engine.monte_carlo_dropout_inference(latest_sequence, n_samples=100)
    
    # Display results
    mean_return = prediction['mean'][0] 
    uncertainty = prediction['std'][0]
    lower_ci = prediction['lower_ci_95'][0]
    upper_ci = prediction['upper_ci_95'][0]
    
    print(f"\n📈 NEXT DAY FORECAST:")
    print(f"   Date: {dates[-1] + pd.Timedelta(days=1)}")
    print(f"   Predicted Return: {mean_return:.6f} ({mean_return*100:.3f}%)")
    print(f"   Uncertainty (±): {uncertainty:.6f} ({uncertainty*100:.3f}%)")
    print(f"   95% Confidence Interval: [{lower_ci:.6f}, {upper_ci:.6f}]")
    print(f"   Interval Width: {(upper_ci - lower_ci)*100:.3f}%")
    
    # Risk assessment
    if uncertainty > 0.01:  # 1% uncertainty threshold
        risk_level = "HIGH"
    elif uncertainty > 0.005:  # 0.5% uncertainty threshold  
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    print(f"   Risk Level: {risk_level}")

if _name_ == "_main_":
    print("🚀 Starting Bayesian LSTM Inference Demonstration")
    print("=" * 60)
    
    # Run main demo
    run_inference_demo()
    
    # Run single prediction demo if successful
    if os.path.exists("results/bayesian_lstm_model.h5"):
        print("\n" + "="*60)
        run_single_prediction_demo()
    
    print("\n🎉 All demos completed!")
    print("Explore the results/ directory for detailed analysis.")