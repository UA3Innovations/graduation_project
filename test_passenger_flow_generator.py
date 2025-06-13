#!/usr/bin/env python3
"""
Test script for OptimizedPassengerFlowGenerator
Uses existing optimized schedules from the genetic algorithm
"""

import sys
from pathlib import Path

# Add the main_pipeline to the path
sys.path.insert(0, str(Path(__file__).parent / "main_pipeline"))

from optimized_passenger_flow_generator import OptimizedPassengerFlowGenerator

def test_passenger_flow_generator():
    """Test the OptimizedPassengerFlowGenerator with existing data"""
    
    print("🧪 TESTING OPTIMIZED PASSENGER FLOW GENERATOR")
    print("=" * 60)
    
    # Set up paths
    project_root = Path(__file__).parent
    stops_file = str(project_root / "data" / "ankara_bus_stops.csv")
    
    # Use the most recent optimization results
    schedule_file = str(project_root / "outputs" / "pipeline_run_quick_20250613_023422" / "optimization_results" / "optimized_schedules.csv")
    output_file = str(project_root / "test_optimized_passenger_flow.csv")
    
    print(f"📄 Input schedule file: {schedule_file}")
    print(f"📄 Output file: {output_file}")
    print(f"🚏 Stops file: {stops_file}")
    
    try:
        # Initialize the generator
        print("\n🔧 Initializing OptimizedPassengerFlowGenerator...")
        generator = OptimizedPassengerFlowGenerator(stops_file, project_root)
        
        # Generate the passenger flow structure
        print("\n🚌 Generating passenger flow structure...")
        result_df = generator.generate_optimized_passenger_flow(
            schedule_file=schedule_file,
            output_file=output_file
        )
        
        if result_df is not None:
            print(f"\n✅ SUCCESS! Generated passenger flow structure with {len(result_df):,} records")
            print(f"📊 Output saved to: {output_file}")
            
            # Show some sample data
            print(f"\n🔍 Sample records:")
            sample_cols = ['datetime', 'line_id', 'stop_id', 'bus_id', 'boarding', 'alighting', 'occupancy_rate']
            print(result_df[sample_cols].head(10).to_string(index=False))
            
            return True
        else:
            print("❌ FAILED! No data generated")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_passenger_flow_generator()
    if success:
        print(f"\n🎉 Test completed successfully!")
    else:
        print(f"\n💥 Test failed!") 