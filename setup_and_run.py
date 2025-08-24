"""
Setup verification and component runner for Regime-Aware Stock Forecasting
"""
import os
import sys
import pandas as pd

def check_data_files():
    """Check if required data files exist"""
    print("🔍 Checking data files...")
    
    required_files = [
        "data/data_with_sentiment.csv",
        "data/data_with_regimes.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "sklearn", "tensorflow"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def run_regime_detection():
    """Run only the regime detection step"""
    print("\n📊 Running Regime Detection...")
    try:
        os.chdir("data")
        exec(open("feature_regime_preprocessing.py").read())
        os.chdir("..")
        print("✅ Regime detection completed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_bayesian_lstm():
    """Run only the Bayesian LSTM step"""
    print("\n🧠 Running Bayesian LSTM...")
    try:
        os.chdir("models")
        exec(open("bayesian_lstm.py").read())
        os.chdir("..")
        print("✅ Bayesian LSTM completed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_evaluation():
    """Run only the evaluation step"""
    print("\n📈 Running Evaluation...")
    try:
        os.chdir("evaluation")
        exec(open("model_evaluation.py").read())
        os.chdir("..")
        print("✅ Evaluation completed")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main menu for running individual components"""
    print("🚀 Regime-Aware Stock Forecasting - Component Runner")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Check setup (data files & dependencies)")
        print("2. Run regime detection only")
        print("3. Run Bayesian LSTM training only")
        print("4. Run evaluation only")
        print("5. Run complete pipeline")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            data_ok = check_data_files()
            deps_ok = check_dependencies()
            if data_ok and deps_ok:
                print("\n✅ Setup looks good!")
            else:
                print("\n❌ Please fix the issues above before proceeding.")
        
        elif choice == "2":
            run_regime_detection()
        
        elif choice == "3":
            if os.path.exists("data/data_with_regimes.csv"):
                run_bayesian_lstm()
            else:
                print("❌ Please run regime detection first (option 2)")
        
        elif choice == "4":
            if os.path.exists("results/bayesian_lstm_predictions.csv"):
                run_evaluation()
            else:
                print("❌ Please run Bayesian LSTM training first (option 3)")
        
        elif choice == "5":
            exec(open("run_complete_pipeline.py").read())
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
