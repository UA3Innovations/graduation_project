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
    if python_version.major >= 3 and python_version.minor >= 11:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} - Need 3.11+")
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
        result = subprocess.run([
            sys.executable, "scripts/deploy_to_azure.py", "--test-only"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed:")
            print(result.stdout)
            print(result.stderr)
            return 1
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return 1
    
    # Step 5: Quick local simulation test
    print("\n5️⃣ Running quick local simulation test...")
    print("   (This may take a few minutes...)")
    
    # Create a test config for quick run
    test_config_content = """
# Quick test configuration
simulation:
  start_date: "2024-01-01"
  end_date: "2024-01-01"  # Just one day for quick test
  time_step: 15           # Larger time step for speed
  buses_per_line: 2       # Fewer buses for speed
  randomize_travel_times: true
  randomize_passenger_demand: true
  weather_effects_probability: 0.15
  seed: 42

data:
  stops_file: "data/ankara_bus_stops_10.csv"

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
            sys.executable, "scripts/deploy_to_azure.py", 
            "--local-run", "--config", "config/test_config.yaml"
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
    print("1. Set up Azure credentials:")
    print("   export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
    print("\n2. Deploy to Azure development environment:")
    print("   python scripts/deploy_to_azure.py --environment development")
    print("\n3. For production deployment:")
    print("   python scripts/deploy_to_azure.py --environment production")
    
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