"""
Minimal TensorFlow Import Test
=============================

This script tests TensorFlow import in different ways to isolate the issue.
"""

import os
import sys

def test_tensorflow_import():
    """Test TensorFlow import with various methods"""
    
    print("🧪 TENSORFLOW IMPORT DIAGNOSTICS")
    print("=" * 50)
    
    print(f"📍 Python executable: {sys.executable}")
    print(f"📍 Python version: {sys.version}")
    print(f"📍 Working directory: {os.getcwd()}")
    
    # Test 1: Basic import
    print("\n1️⃣ Testing basic TensorFlow import...")
    try:
        import tensorflow as tf
        print(f"   ✅ Success! Version: {tf.__version__}")
        print(f"   📂 Location: {tf.__file__}")
        
        # Test GPU availability
        physical_devices = tf.config.list_physical_devices()
        print(f"   🖥️ Physical devices: {len(physical_devices)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}: {e}")
        return False
    
    # Test 2: Test with different import style
    print("\n2️⃣ Testing alternative import...")
    try:
        from tensorflow import keras
        print("   ✅ Keras import successful")
    except Exception as e:
        print(f"   ❌ Keras import failed: {e}")
    
    # Test 3: Environment check
    print(f"\n3️⃣ Environment variables:")
    tf_vars = [k for k in os.environ.keys() if 'TF_' in k or 'TENSOR' in k]
    if tf_vars:
        for var in tf_vars:
            print(f"   {var}: {os.environ[var]}")
    else:
        print("   No TensorFlow environment variables found")

if __name__ == "__main__":
    success = test_tensorflow_import()
    
    if success:
        print("\n🎉 TensorFlow is working!")
        print("   The issue might be in the pipeline script imports")
    else:
        print("\n❌ TensorFlow import failed")
        print("   Try the following solutions:")
        print("   1. Restart your terminal/IDE")
        print("   2. pip install --upgrade tensorflow")
        print("   3. Check Visual C++ Redistributables")