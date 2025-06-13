#!/usr/bin/env python3
"""
Integration Test Script for Schedule Optimization Project

This script tests the integrated pipeline setup and basic functionality.
"""

import sys
import os
from pathlib import Path
import subprocess

def test_project_structure():
    """Test that all required directories and files exist"""
    print("🔍 Testing project structure...")
    
    required_dirs = [
        "bus_simulation_pipeline",
        "bus_optimization_pipeline", 
        "bus_prediction_pipeline",
        "bus_evaluation_pipeline",
        "main_pipeline",
        "docker",
        "config",
        "data",
        "outputs",
        "logs"
    ]
    
    required_files = [
        "main_pipeline/integrated_pipeline.py",
        "docker/Dockerfile",
        "docker-compose.yml",
        "README.md",
        "data/ankara_bus_stops.csv"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is complete")
    return True

def test_pipeline_imports():
    """Test that the integrated pipeline can import all components"""
    print("\n🔍 Testing pipeline imports...")
    
    try:
        # Add pipeline paths
        project_root = Path(".")
        sys.path.extend([
            str(project_root / "bus_simulation_pipeline" / "src"),
            str(project_root / "bus_optimization_pipeline" / "src"),
            str(project_root / "bus_prediction_pipeline" / "src"),
            str(project_root / "bus_evaluation_pipeline" / "src")
        ])
        
        # Test individual pipeline imports
        print("  Testing simulation pipeline import...")
        from core.simulation_engine import SimulationEngine
        print("  ✅ Simulation pipeline imported")
        
        print("  Testing optimization pipeline import...")
        from optimization.ga_optimize import GeneticScheduleOptimizer
        print("  ✅ Optimization pipeline imported")
        
        print("  Testing prediction pipeline import...")
        from prediction_models.hybrid_model import HybridModel
        print("  ✅ Prediction pipeline imported")
        
        print("  Testing evaluation pipeline import...")
        from evaluation_engine import OptimizationEvaluator
        print("  ✅ Evaluation pipeline imported")
        
        print("✅ All pipeline imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_integrated_pipeline():
    """Test the integrated pipeline script"""
    print("\n🔍 Testing integrated pipeline script...")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, "main_pipeline/integrated_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Integrated pipeline script is executable")
            print("✅ Help command works")
            return True
        else:
            print(f"❌ Pipeline script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Pipeline script timed out")
        return False
    except Exception as e:
        print(f"❌ Pipeline script test failed: {e}")
        return False

def test_docker_setup():
    """Test Docker configuration"""
    print("\n🔍 Testing Docker setup...")
    
    # Check if Docker is available
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("⚠️  Docker not available - skipping Docker tests")
            return True
    except:
        print("⚠️  Docker not available - skipping Docker tests")
        return True
    
    # Test docker-compose syntax
    try:
        result = subprocess.run(["docker-compose", "config"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Docker Compose configuration is valid")
            return True
        else:
            print(f"❌ Docker Compose configuration error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False

def test_data_files():
    """Test that required data files exist and are valid"""
    print("\n🔍 Testing data files...")
    
    data_file = "data/ankara_bus_stops.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file missing: {data_file}")
        return False
    
    # Check file size
    file_size = os.path.getsize(data_file)
    if file_size < 1000:  # Should be at least 1KB
        print(f"❌ Data file too small: {file_size} bytes")
        return False
    
    print(f"✅ Data file exists and has reasonable size: {file_size:,} bytes")
    
    # Test that each pipeline has its data file
    pipeline_data_files = [
        "bus_simulation_pipeline/data/ankara_bus_stops.csv",
        "bus_optimization_pipeline/data/ankara_bus_stops.csv",
        "bus_prediction_pipeline/data/ankara_bus_stops.csv",
        "bus_evaluation_pipeline/data/ankara_bus_stops.csv"
    ]
    
    for data_file in pipeline_data_files:
        if os.path.exists(data_file):
            print(f"✅ {data_file} exists")
        else:
            print(f"⚠️  {data_file} missing (may be optional)")
    
    return True

def main():
    """Run all integration tests"""
    print("🧪 SCHEDULE OPTIMIZATION PROJECT - INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Pipeline Imports", test_pipeline_imports),
        ("Integrated Pipeline", test_integrated_pipeline),
        ("Docker Setup", test_docker_setup),
        ("Data Files", test_data_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("🚀 The project is ready for use!")
        print("\nNext steps:")
        print("  1. Run quick test: python main_pipeline/integrated_pipeline.py --mode quick")
        print("  2. Or use Docker: docker-compose up schedule-optimizer-quick")
    else:
        print("\n⚠️  Some integration tests failed.")
        print("Please check the errors above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 