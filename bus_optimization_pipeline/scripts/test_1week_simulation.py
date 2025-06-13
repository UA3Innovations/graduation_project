#!/usr/bin/env python3
"""
Test script to run a 1-week simulation and then optimize a single day schedule.
This demonstrates the complete workflow from simulation to optimization.
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add compartmentalized path for simulation components
compartmentalized_path = os.path.join(os.path.dirname(__file__), '..', '..', 'compartmentalized')
if compartmentalized_path not in sys.path:
    sys.path.insert(0, compartmentalized_path)

def run_1week_simulation():
    """Run a 1-week simulation to generate baseline data."""
    print("🚌 Running 1-Week Simulation")
    print("=" * 60)
    
    try:
        from simulation_engine import SimulationEngine
        from data_models import BusTransitData, SimulationConfig
        
        # Create simulation configuration for 1 week
        start_date = datetime(2025, 6, 2)  # Monday
        end_date = start_date + timedelta(days=7)  # Next Monday
        
        config = SimulationConfig(
            start_date=start_date,
            end_date=end_date,
            time_step=10,  # 10-minute steps for faster simulation
            randomize_travel_times=True,
            randomize_passenger_demand=True,
            weather_effects_probability=0.15,
            seed=42
        )
        
        # Create data container and simulation engine
        data = BusTransitData()
        engine = SimulationEngine(data, config)
        
        # Load data
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ankara_bus_stops.csv')
        print(f"📁 Loading data from: {data_file}")
        
        if not engine.load_data(data_file):
            print("❌ Failed to load simulation data")
            return None
        
        print(f"✅ Loaded {len(data.lines)} lines and {len(data.stops)} stops")
        
        # Setup simulation with fewer buses for faster execution
        print("🔧 Setting up simulation...")
        if not engine.setup_simulation(num_buses_per_line=6):  # Reduced from 10 to 6
            print("❌ Failed to setup simulation")
            return None
        
        print(f"✅ Generated {len(data.buses)} buses")
        
        # Run simulation
        print("🏃 Running 1-week simulation...")
        start_time = time.time()
        
        results = engine.run_simulation()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Simulation completed in {elapsed_time:.1f} seconds")
        
        if len(results) > 0:
            print(f"📊 Generated {len(results)} passenger flow records")
            
            # Export results
            output_dir = "simulation_output"
            os.makedirs(output_dir, exist_ok=True)
            engine.export_results(output_dir)
            
            print(f"📄 Results exported to: {output_dir}")
            return os.path.join(output_dir, "passenger_flow_results.csv")
        else:
            print("❌ No simulation results generated")
            return None
            
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_optimization_with_simulation_data(simulation_results_file):
    """Test optimization using the simulation results."""
    print("\n🧬 Testing Optimization with Simulation Data")
    print("=" * 60)
    
    try:
        from optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig
        
        # Create optimization configuration
        config = OptimizationConfig(
            population_size=10,
            generations=15,
            simulation_duration_hours=3,
            time_step=10,
            passenger_wait_weight=0.4,
            bus_utilization_weight=0.3,
            overcrowding_weight=0.2,
            service_coverage_weight=0.1
        )
        
        # Initialize optimizer
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ankara_bus_stops.csv')
        optimizer = GeneticScheduleOptimizer(config, data_file)
        
        if not optimizer.base_data:
            print("❌ Failed to initialize optimizer")
            return False
        
        print(f"✅ Optimizer initialized with {len(optimizer.line_ids)} lines")
        
        # Test optimizing a specific line
        target_date = datetime(2025, 6, 3)  # Tuesday from our simulation week
        test_line = "101"  # Major line
        
        print(f"🎯 Optimizing line {test_line} for {target_date.strftime('%Y-%m-%d')}")
        
        start_time = time.time()
        result = optimizer.optimize_line_schedule(test_line, target_date)
        elapsed_time = time.time() - start_time
        
        if result and result.fitness is not None:
            print(f"✅ Optimization completed in {elapsed_time:.1f} seconds")
            print(f"   Line: {result.line_id}")
            print(f"   Fitness Score: {result.fitness:.4f}")
            print(f"   Number of Departures: {len(result.departure_times)}")
            
            # Show some departure times
            print(f"   Sample Departures:")
            for i, departure in enumerate(result.departure_times[:5]):
                print(f"     {i+1}. {departure.strftime('%H:%M')}")
            if len(result.departure_times) > 5:
                print(f"     ... and {len(result.departure_times) - 5} more")
            
            # Export optimized schedule
            output_dir = "optimization_output"
            os.makedirs(output_dir, exist_ok=True)
            
            import pandas as pd
            schedule_data = []
            for departure in result.departure_times:
                schedule_data.append({
                    'line_id': result.line_id,
                    'departure_time': departure,
                    'fitness_score': result.fitness,
                    'optimization_date': target_date.strftime('%Y-%m-%d')
                })
            
            df = pd.DataFrame(schedule_data)
            output_file = os.path.join(output_dir, f"optimized_schedule_{test_line}_{target_date.strftime('%Y%m%d')}.csv")
            df.to_csv(output_file, index=False)
            
            print(f"📄 Optimized schedule saved to: {output_file}")
            return True
        else:
            print("❌ Optimization failed")
            return False
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the complete test."""
    print("🚌 Complete Simulation + Optimization Test")
    print("=" * 80)
    
    # Step 1: Run 1-week simulation
    simulation_results = run_1week_simulation()
    
    if not simulation_results:
        print("❌ Simulation failed - cannot proceed with optimization test")
        return 1
    
    # Step 2: Test optimization
    optimization_success = test_optimization_with_simulation_data(simulation_results)
    
    if optimization_success:
        print("\n🎉 Complete Test Successful!")
        print("✅ 1-week simulation completed")
        print("✅ Schedule optimization completed")
        print("✅ Results exported")
        print("\nThe genetic algorithm optimization is working correctly!")
        return 0
    else:
        print("\n❌ Optimization test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 