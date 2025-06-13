#!/usr/bin/env python3
"""
Simple 1-week local simulation test using original files.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the compartmentalized directory to Python path
compartmentalized_path = os.path.join(os.path.dirname(__file__), '..', 'compartmentalized')
sys.path.insert(0, compartmentalized_path)

# Import from the original files
from data_models import BusTransitData, SimulationConfig
from simulation_engine import SimulationEngine

def main():
    """Run a simple 1-week simulation test."""
    print("🚌 Bus Transit Simulation - 1 Week Local Test")
    print("=" * 50)
    
    # Configuration for 1-week test
    start_date = datetime(2024, 1, 1, 5, 0)  # Start at 5 AM on Monday
    end_date = datetime(2024, 1, 7, 23, 59)  # End at 11:59 PM on Sunday
    
    print(f"📅 Simulating: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (7 days)")
    print(f"⏱️  Time step: 15 minutes (for faster processing)")
    print(f"🚍 Buses per line: 4")
    print()
    
    # Create simulation configuration
    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        time_step=15,  # 15-minute steps for faster processing over a week
        randomize_travel_times=True,
        randomize_passenger_demand=True,
        weather_effects_probability=0.15,
        seed=42
    )
    
    # Create data container
    data = BusTransitData()
    
    # Create simulation engine
    engine = SimulationEngine(data, config)
    
    # Set debug flag
    engine.debug = False  # Set to True for detailed output
    
    print("Step 1: Loading transit data...")
    csv_path = os.path.join(compartmentalized_path, 'ankara_bus_stops.csv')
    success = engine.load_data(csv_path)
    if not success:
        print("❌ Failed to load data")
        return False
    
    print(f"✅ Loaded {len(data.lines)} lines and {len(data.stops)} stops")
    
    # Print line information
    print("\n📊 Lines loaded:")
    for line_id, line in data.lines.items():
        print(f"   Line {line_id}: {len(line.stops)} stops")
    print()
    
    print("Step 2: Setting up simulation...")
    success = engine.setup_simulation(num_buses_per_line=4)
    if not success:
        print("❌ Failed to setup simulation")
        return False
    
    print(f"✅ Generated {len(data.buses)} buses")
    print()
    
    print("Step 3: Running 1-week simulation...")
    print("⏳ This will take several minutes...")
    
    import time
    start_time = time.time()
    
    try:
        results = engine.run_simulation()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Simulation completed in {elapsed_time:.2f} seconds")
        print()
        
        if len(results) > 0:
            print("📈 Results Summary:")
            print(f"   📝 Total records: {len(results)}")
            print(f"   🚶 Total boardings: {results['boarding'].sum()}")
            print(f"   🚪 Total alightings: {results['alighting'].sum()}")
            
            # Day-by-day analysis
            if 'date' in results.columns:
                daily_stats = results.groupby('date').agg({
                    'boarding': 'sum',
                    'alighting': 'sum'
                }).reset_index()
                print(f"\n📊 Daily passenger activity:")
                for _, day in daily_stats.iterrows():
                    day_name = datetime.strptime(str(day['date']), '%Y-%m-%d').strftime('%A')
                    print(f"   {day['date']} ({day_name}): {day['boarding']} boardings, {day['alighting']} alightings")
            
            # Overcrowding analysis
            overcrowded = results[results['occupancy_rate'] > 1.0]
            if len(overcrowded) > 0:
                print(f"\n   ⚠️  Overcrowded instances: {len(overcrowded)} ({len(overcrowded)/len(results)*100:.1f}%)")
                max_occupancy = results['occupancy_rate'].max()
                print(f"   📊 Maximum occupancy: {max_occupancy:.1f}x capacity")
            else:
                print(f"\n   ✅ No overcrowding detected")
            
            # Peak hours analysis
            if 'hour' in results.columns:
                peak_hours = results.groupby('hour')['boarding'].sum().sort_values(ascending=False)
                print(f"   🕐 Peak boarding hour: {peak_hours.index[0]}:00 ({peak_hours.iloc[0]} boardings)")
            
            # Weekend vs Weekday analysis
            if 'date' in results.columns:
                results['day_of_week'] = results['date'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').weekday())
                weekday_total = results[results['day_of_week'] < 5]['boarding'].sum()
                weekend_total = results[results['day_of_week'] >= 5]['boarding'].sum()
                print(f"   📅 Weekday total: {weekday_total} boardings")
                print(f"   🎉 Weekend total: {weekend_total} boardings")
            
            print()
        else:
            print("⚠️  No simulation results generated")
            return False
        
        print("Step 4: Exporting results...")
        
        # Create output directory
        output_dir = "test_output_1week"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export results
        engine.export_results(output_dir)
        
        print(f"✅ Results saved to '{output_dir}/' directory")
        
        # List output files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.csv') or f.endswith('.json')]
            print("\n📁 Generated files:")
            total_size = 0
            for file in sorted(files):
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    # Get file size
                    size = os.path.getsize(file_path)
                    total_size += size
                    if size > 1024*1024:
                        size_str = f"{size/(1024*1024):.1f} MB"
                    elif size > 1024:
                        size_str = f"{size/1024:.1f} KB"
                    else:
                        size_str = f"{size} bytes"
                    print(f"   📄 {file} ({size_str})")
            
            # Total size
            if total_size > 1024*1024:
                total_str = f"{total_size/(1024*1024):.1f} MB"
            elif total_size > 1024:
                total_str = f"{total_size/1024:.1f} KB"
            else:
                total_str = f"{total_size} bytes"
            print(f"\n💾 Total output size: {total_str}")
        
        print("\n" + "=" * 50)
        print("🎉 1-Week Simulation Test Completed Successfully!")
        print("=" * 50)
        print("\n💡 Analysis suggestions:")
        print("   - Compare weekday vs weekend patterns")
        print("   - Analyze rush hour consistency across the week")
        print("   - Look for overcrowding trends by day and line")
        print("   - Use the data for genetic algorithm optimization")
        print("   - Check transfer patterns at major hubs")
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Simulation failed after {elapsed_time:.2f} seconds: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 