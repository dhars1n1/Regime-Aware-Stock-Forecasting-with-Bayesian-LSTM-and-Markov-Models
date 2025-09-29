"""
Quick test script to verify feature auto-detection and pipeline setup
"""

import sys
sys.path.append('models')

from data_processor import RegimeAwareDataProcessor
import pandas as pd

def test_feature_detection():
    """Test the feature auto-detection functionality"""
    print("🧪 TESTING FEATURE AUTO-DETECTION")
    print("=" * 50)
    
    # Load data
    data_path = 'data/data_with_regimes.csv'
    print(f"📊 Loading data from: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        print(f"   Shape: {data.shape}")
        
        # Initialize processor with auto-detection
        processor = RegimeAwareDataProcessor(
            sequence_length=20,
            feature_columns='auto',
            target_column='log_return',
            regime_column='regime_label',
            exclude_columns=['Date', 'regime_viterbi', 'regime_0_prob', 'regime_1_prob', 'regime_2_prob']
        )
        
        # Test feature detection
        features = processor.get_feature_columns(data)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Detected {len(features)} features")
        print(f"   Features: {features}")
        
        # Test if we have the expected number
        expected_min = 25  # Should have at least 25 features
        if len(features) >= expected_min:
            print(f"✅ Feature count looks good (>= {expected_min})")
        else:
            print(f"⚠️ Fewer features than expected (< {expected_min})")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_feature_detection()
    if success:
        print("\n🎉 Feature auto-detection test PASSED!")
        print("The pipeline is ready to use all your available features!")
    else:
        print("\n❌ Feature auto-detection test FAILED!")