#!/usr/bin/env python3
"""
Test script for the bus optimization pipeline.
This script runs a quick test to verify everything is working correctly.
"""

import os
import sys
import time
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test optimization imports
        from optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig
        print("✅ Optimization modules imported successfully")
        
        # Test simulation imports (should work through path manipulation)
        from core.data_models import BusTransitData, SimulationConfig
        from components.transit_network import TransitNetwork
        print("✅ Simulation modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded correctly."""
    print("\n📁 Testing data loading...")
    
    try:
        from optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig
        
        # Check if data file exists
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ankara_bus_stops.csv')
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        # Test loading data
        config = OptimizationConfig(population_size=5, generations=2)
        optimizer = GeneticScheduleOptimizer(config, data_file)
        
        if optimizer.base_data and len(optimizer.line_ids) > 0:
            print(f"✅ Data loaded successfully: {len(optimizer.line_ids)} lines, {len(optimizer.base_data.stops)} stops")
            return True
        else:
            print("❌ Failed to load data properly")
            return False
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_quick_optimization():
    """Test a very quick optimization run."""
    print("\n🧬 Testing quick optimization...")
    
    try:
        from optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig
        
        # Create minimal configuration for testing
        config = OptimizationConfig(
            population_size=4,
            generations=3,
            simulation_duration_hours=1,
            time_step=15
        )
        
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ankara_bus_stops.csv')
        optimizer = GeneticScheduleOptimizer(config, data_file)
        
        if not optimizer.base_data:
            print("❌ Failed to initialize optimizer")
            return False
        
        # Test optimizing one line
        test_date = datetime(2025, 6, 2)
        line_id = optimizer.line_ids[0]  # Get first available line
        
        print(f"   Optimizing line {line_id} with minimal parameters...")
        start_time = time.time()
        
        result = optimizer.optimize_line_schedule(line_id, test_date)
        
        elapsed_time = time.time() - start_time
        
        if result and result.fitness is not None:
            print(f"✅ Optimization completed in {elapsed_time:.1f}s")
            print(f"   Line: {result.line_id}")
            print(f"   Fitness: {result.fitness:.4f}")
            print(f"   Departures: {len(result.departure_times)}")
            return True
        else:
            print("❌ Optimization failed or returned invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'optimization_config.yaml')
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['optimization', 'data', 'output']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration file loaded and validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚌 Bus Optimization Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Configuration Test", test_configuration),
        ("Quick Optimization Test", test_quick_optimization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The optimization pipeline is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 