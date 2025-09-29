"""
Alternative TensorFlow Installation Methods for Windows Long Path Issues
========================================================================

If TensorFlow keeps failing with Windows Long Path errors, try these alternatives:

Method 1: Use System-Wide Installation (Outside venv)
-----------------------------------------------------
1. Deactivate venv: deactivate
2. Install globally: pip install tensorflow==2.16.1
3. Run pipeline from global Python

Method 2: Use Conda (Recommended for Windows)
---------------------------------------------
1. Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Create conda environment:
   conda create -n tf_env python=3.11
   conda activate tf_env
   conda install tensorflow
3. Install other requirements:
   pip install -r requirements.txt

Method 3: Use Pre-compiled Wheels
---------------------------------
1. Download TensorFlow wheel directly
2. Install with: pip install tensorflow-2.16.1-cp312-cp312-win_amd64.whl

Method 4: Docker (Advanced)
---------------------------
1. Install Docker Desktop
2. Use TensorFlow Docker image
3. Mount your project directory

Quick Test Script:
"""

def test_tensorflow_alternatives():
    """Test different TensorFlow import methods"""
    
    # Test 1: Try importing TensorFlow
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully!")
        print(f"   Version: {tf.__version__}")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
    
    # Test 2: Try TensorFlow Lite (smaller alternative)
    try:
        import tensorflow.lite as tflite
        print("✅ TensorFlow Lite available (limited functionality)")
        return False
    except ImportError:
        print("❌ TensorFlow Lite also not available")
    
    # Test 3: Suggest alternatives
    print("\n🔄 ALTERNATIVES TO TRY:")
    print("1. Use PyTorch instead: pip install torch")
    print("2. Use Keras with TensorFlow backend: pip install keras")
    print("3. Use system-wide Python instead of venv")
    
    return False

if __name__ == "__main__":
    test_tensorflow_alternatives()