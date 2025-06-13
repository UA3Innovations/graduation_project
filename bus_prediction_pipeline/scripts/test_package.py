#!/usr/bin/env python3
"""
Test script to verify the bus prediction pipeline package works correctly
"""

import sys
import os

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all models can be imported successfully"""
    print("Testing package imports...")
    
    try:
        # Test individual model imports
        from prediction_models.lstm_model import lstm_model
        print("✅ LSTM model imported successfully")
        
        from prediction_models.prophet_model import ProphetModel
        print("✅ Prophet model imported successfully")
        
        from prediction_models.hybrid_model import HybridModel
        print("✅ Hybrid model imported successfully")
        
        # Test package-level imports
        from prediction_models import lstm_model as lstm_pkg
        from prediction_models import ProphetModel as prophet_pkg
        from prediction_models import HybridModel as hybrid_pkg
        print("✅ Package-level imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_initialization():
    """Test that models can be initialized"""
    print("\nTesting model initialization...")
    
    try:
        from prediction_models import lstm_model, ProphetModel, HybridModel
        
        # Test LSTM initialization
        lstm = lstm_model(sequence_length=48)
        print("✅ LSTM model initialized successfully")
        
        # Test Prophet initialization
        prophet = ProphetModel()
        print("✅ Prophet model initialized successfully")
        
        # Test Hybrid initialization
        hybrid = HybridModel(sequence_length=48)
        print("✅ Hybrid model initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_data_file():
    """Test that the data file exists"""
    print("\nTesting data file...")
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ankara_bus_stops.csv')
    
    if os.path.exists(data_path):
        print(f"✅ Data file found: {data_path}")
        
        # Check file size
        size = os.path.getsize(data_path)
        print(f"   File size: {size:,} bytes")
        
        return True
    else:
        print(f"❌ Data file not found: {data_path}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("BUS PREDICTION PIPELINE - PACKAGE TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_initialization,
        test_data_file
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 