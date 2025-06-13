#!/usr/bin/env python3
"""
Setup and test script for the bus simulation pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main setup and test function."""
    print("🚌 Bus Simulation Pipeline - Setup and Test")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"📁 Working directory: {project_root}")
    
    # Step 1: Check Python version
    print("\n1️⃣ Checking Python version...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Need 3.8+")
        return 1
    
    # Step 2: Install dependencies
    print("\n2️⃣ Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return 1
    
    # Step 3: Run tests
    print("\n3️⃣ Running pipeline tests...")
    try:
        result = subprocess.run([sys.executable, "tests/test_pipeline.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed")
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1
    
    # Step 4: Test configuration validation
    print("\n4️⃣ Testing configuration validation...")
    try:
        # Check if config file exists and is valid
        config_file = project_root / "config" / "simulation_config.yaml"
        if config_file.exists():
            print("✅ Configuration file found")
            
            # Try to load and validate the config
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['simulation', 'data', 'output']
            for section in required_sections:
                if section in config:
                    print(f"✅ Configuration section '{section}' found")
                else:
                    print(f"❌ Missing configuration section: {section}")
                    return 1
        else:
            print("❌ Configuration file not found")
            return 1
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1
    
    # Step 5: Quick local simulation test
    print("\n5️⃣ Running quick local simulation test...")
    print("   (This may take a few minutes...)")
    
    # Create a test config for quick run
    test_config_content = """
# Quick test configuration
simulation:
  start_date: "2025-06-02"
  end_date: "2025-06-02"  # Just one day for quick test
  time_step: 15           # Larger time step for speed
  buses_per_line: 2       # Fewer buses for speed
  randomize_travel_times: true
  randomize_passenger_demand: true
  weather_effects_probability: 0.15
  seed: 42

data:
  stops_file: "data/ankara_bus_stops.csv"

output:
  directory: "test_output"
  summary: true
  debug: false
"""
    
    # Write test config
    with open("config/test_config.yaml", "w") as f:
        f.write(test_config_content)
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/run_simulation.py", 
            "--config", "config/test_config.yaml"
        ], capture_output=True, text=True, cwd=project_root, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Local simulation test completed successfully")
            
            # Check if output files were created
            output_dir = project_root / "test_output"
            if output_dir.exists():
                output_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.json"))
                print(f"   Generated {len(output_files)} output files")
                for file in output_files[:3]:  # Show first 3 files
                    print(f"   - {file.name}")
                if len(output_files) > 3:
                    print(f"   ... and {len(output_files) - 3} more")
            else:
                print("   ⚠️ No output directory created")
                
        else:
            print("❌ Local simulation test failed:")
            print(result.stdout)
            print(result.stderr)
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Local simulation test timed out (>5 minutes)")
        return 1
    except Exception as e:
        print(f"❌ Error running local simulation: {e}")
        return 1
    
    # Step 6: Summary
    print("\n" + "=" * 50)
    print("🎉 SETUP AND TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Run a full simulation:")
    print("   python scripts/run_simulation.py")
    print("\n2. Customize simulation parameters:")
    print("   Edit config/simulation_config.yaml")
    print("\n3. Run with custom config:")
    print("   python scripts/run_simulation.py --config your_config.yaml")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1) 