"""
Complete pipeline runner for Regime-Aware Stock Forecasting
"""
import os
import sys
import subprocess

def run_pipeline():
    """Run the complete forecasting pipeline"""
    
    print("🚀 Starting Regime-Aware Stock Forecasting Pipeline")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    # Step 1: Feature engineering and regime detection
    print("\n📊 Step 1: Feature Engineering & Regime Detection")
    print("-" * 50)
    try:
        # Change to data directory and run preprocessing
        os.chdir("data")
        exec(open("feature_regime_preprocessing.py").read())
        os.chdir("..")
        print("✅ Regime detection completed")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False
    
    # Step 2: Bayesian LSTM training
    print("\n🧠 Step 2: Training Regime-Aware Bayesian LSTM")
    print("-" * 50)
    try:
        # Change to models directory and run LSTM training
        os.chdir("models")
        exec(open("bayesian_lstm.py").read())
        os.chdir("..")
        print("✅ Model training completed")
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False
    
    # Step 3: Comprehensive evaluation
    print("\n📈 Step 3: Model Evaluation & Analysis")
    print("-" * 50)
    try:
        # Change to evaluation directory and run evaluation
        os.chdir("evaluation")
        exec(open("model_evaluation.py").read())
        os.chdir("..")
        print("✅ Evaluation completed")
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return False
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 60)
    print("📁 Check the following outputs:")
    print("   • data/data_with_regimes.csv - Data with regime information")
    print("   • models/bayesian_lstm_model.h5 - Trained model")
    print("   • results/bayesian_lstm_predictions.csv - Predictions with uncertainty")
    
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tensorflow",
        "scipy"
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

if __name__ == "__main__":
    # Check if user wants to install requirements
    install_deps = input("Do you want to install required dependencies? (y/n): ").lower().strip()
    if install_deps == 'y':
        install_requirements()
    
    # Run the pipeline
    success = run_pipeline()
    
    if success:
        print("\n🎯 Next Steps:")
        print("- Review the evaluation report for model performance insights")
        print("- Experiment with different hyperparameters")
        print("- Try different sequence lengths or model architectures")
        print("- Consider adding more features or different regime detection methods")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
