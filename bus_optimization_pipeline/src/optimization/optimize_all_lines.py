#!/usr/bin/env python3
"""
Optimize all bus lines and combine schedules into one CSV file.
This is a test version with reduced parameters for faster execution.
"""

import sys
import os

# Add the simulation package to the path
simulation_src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bus_simulation_pipeline', 'src')
if simulation_src_path not in sys.path:
    sys.path.insert(0, simulation_src_path)

# Import from compartmentalized folder as fallback
compartmentalized_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'compartmentalized')
if compartmentalized_path not in sys.path:
    sys.path.insert(0, compartmentalized_path)

try:
    from .ga_optimize import run_optimization, OptimizationConfig, GeneticScheduleOptimizer
except ImportError:
    from ga_optimize import run_optimization, OptimizationConfig, GeneticScheduleOptimizer
from datetime import datetime
import pandas as pd
import time

def optimize_all_lines_fast():
    """Optimize all lines with reduced parameters for testing"""
    
    print("🚌 Optimizing ALL Bus Lines - Fast Test Mode")
    print("=" * 60)
    
    # Get all available line IDs
    config = OptimizationConfig(
        population_size=8,  # Very small for fast testing
        generations=10,     # Reduced generations
        simulation_duration_hours=2  # Short simulation
    )
    temp_optimizer = GeneticScheduleOptimizer(config, "ankara_bus_stops.csv")
    
    if not temp_optimizer.base_data:
        print("❌ Failed to load network data")
        return None
    
    all_lines = temp_optimizer.line_ids
    print(f"📋 Found {len(all_lines)} lines to optimize: {all_lines}")
    
    # Use reduced parameters for faster testing
    print("\n⚡ Using ultra-fast test parameters:")
    print("   - Population size: 8 (instead of 20)")
    print("   - Generations: 10 (instead of 30)")
    print("   - Simulation duration: 2 hours (instead of 6)")
    print("   - Expected time: ~2-3 minutes per line")
    
    # Track optimization results
    all_results = []
    start_time = time.time()
    
    # Optimize each line individually (for better control and reporting)
    for i, line_id in enumerate(all_lines, 1):
        print(f"\n🧬 [{i}/{len(all_lines)}] Optimizing line {line_id}...")
        
        line_start = time.time()
        
        try:
            # Create custom config for this line
            custom_config = OptimizationConfig(
                population_size=8,
                generations=10,
                simulation_duration_hours=2,
                passenger_wait_weight=0.4,
                bus_utilization_weight=0.3,
                overcrowding_weight=0.2,
                service_coverage_weight=0.1
            )
            
            # Create optimizer with custom config
            optimizer = GeneticScheduleOptimizer(custom_config, "ankara_bus_stops.csv")
            
            if optimizer.base_data:
                # Optimize this specific line
                test_date = datetime(2025, 6, 2)
                schedule = optimizer.optimize_line_schedule(line_id, test_date)
                
                line_time = time.time() - line_start
                
                print(f"   ✅ Complete! Fitness: {schedule.fitness:.4f}, "
                      f"Departures: {len(schedule.departure_times)}, "
                      f"Time: {line_time:.1f}s")
                
                # Store results
                for departure_time in schedule.departure_times:
                    all_results.append({
                        'line_id': line_id,
                        'departure_time': departure_time,
                        'fitness_score': schedule.fitness,
                        'status': 'optimized'
                    })
            else:
                print(f"   ❌ Failed to setup optimizer for line {line_id}")
                
        except Exception as e:
            print(f"   ❌ Error optimizing line {line_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_file = "../output/all_lines_optimized_schedule.csv"
        combined_df.to_csv(output_file, index=False)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"📄 Combined schedule saved to: {output_file}")
        print(f"📊 Total departures: {len(all_results)}")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📈 Average fitness: {combined_df['fitness_score'].mean():.4f}")
        
        # Show summary by line
        print(f"\n📋 Summary by line:")
        summary = combined_df.groupby('line_id').agg({
            'departure_time': 'count',
            'fitness_score': 'first'
        }).rename(columns={'departure_time': 'departures'})
        
        for line_id, row in summary.iterrows():
            print(f"   Line {line_id}: {row['departures']} departures, fitness: {row['fitness_score']:.4f}")
        
        return combined_df
    else:
        print("❌ No schedules were successfully optimized")
        return None

if __name__ == "__main__":
    result = optimize_all_lines_fast() 