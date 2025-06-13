#!/usr/bin/env python3
"""
Test script to verify the bus evaluation pipeline package works correctly
"""

import sys
import os

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all components can be imported successfully"""
    print("Testing package imports...")
    
    try:
        # Test main package import
        from evaluation_engine import OptimizationEvaluator
        print("✅ OptimizationEvaluator imported successfully")
        
        # Test package-level import
        import evaluation_engine
        print("✅ evaluation_engine package imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_evaluator_initialization():
    """Test that the evaluator can be initialized"""
    print("\nTesting evaluator initialization...")
    
    try:
        from evaluation_engine import OptimizationEvaluator
        
        # Test default initialization
        evaluator = OptimizationEvaluator()
        print("✅ Default evaluator initialized successfully")
        
        # Test custom output directory
        evaluator_custom = OptimizationEvaluator(output_dir="test_outputs")
        print("✅ Custom evaluator initialized successfully")
        
        # Check if directories were created
        if os.path.exists("evaluation_outputs"):
            print("✅ Default output directory created")
        
        if os.path.exists("test_outputs"):
            print("✅ Custom output directory created")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    print("\nTesting data validation...")
    
    try:
        from evaluation_engine import OptimizationEvaluator
        import pandas as pd
        from datetime import datetime, timedelta
        
        evaluator = OptimizationEvaluator(output_dir="test_validation")
        
        # Create sample data
        dates = [datetime.now() + timedelta(hours=i) for i in range(24)]
        sample_data = pd.DataFrame({
            'datetime': dates,
            'line_id': ['101'] * 24,
            'stop_id': [f'stop_{i%5}' for i in range(24)],
            'boarding': [10 + i for i in range(24)],
            'alighting': [8 + i for i in range(24)],
            'occupancy_rate': [0.5 + (i * 0.02) for i in range(24)],
            'hour': [i for i in range(24)]
        })
        
        # Save sample data
        sample_data.to_csv("test_original.csv", index=False)
        sample_data.to_csv("test_optimized.csv", index=False)
        
        # Test data loading
        evaluator.load_data(
            original_file="test_original.csv",
            optimized_file="test_optimized.csv"
        )
        
        print("✅ Data loading and validation successful")
        
        # Cleanup
        os.remove("test_original.csv")
        os.remove("test_optimized.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files and directories"""
    print("\nCleaning up test files...")
    
    import shutil
    
    test_dirs = ["evaluation_outputs", "test_outputs", "test_validation"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Removed {test_dir}")
            except Exception as e:
                print(f"⚠️  Could not remove {test_dir}: {e}")

def main():
    """Run all tests"""
    print("🧪 BUS EVALUATION PIPELINE - PACKAGE TESTING")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Initialization Tests", test_evaluator_initialization),
        ("Data Validation Tests", test_data_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 