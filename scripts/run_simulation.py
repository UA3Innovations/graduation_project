"""
Main script to run the bus transit simulation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Handle imports with fallback for different execution contexts
try:
    from core.data_models import BusTransitData, SimulationConfig
    from core.simulation_engine import SimulationEngine
except ImportError:
    # Fallback for direct execution
from data_models import BusTransitData, SimulationConfig
from simulation_engine import SimulationEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the bus transit simulation.')
    
    parser.add_argument('--config', type=str, 
                       help='Path to YAML configuration file')
    
    parser.add_argument('--stops-file', type=str, default='ankara_bus_stops.csv',
                       help='Path to CSV file with stops and lines information')
    
    parser.add_argument('--start-date', type=str, 
                       default=datetime.now().replace(day=1).strftime('%Y-%m-%d'),
                       help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--end-date', type=str, 
                       default=(datetime.now().replace(day=1) + timedelta(days=7)).strftime('%Y-%m-%d'),
                       help='End date in YYYY-MM-DD format')
    
    parser.add_argument('--time-step', type=int, default=5,
                       help='Simulation time step in minutes')
    
    parser.add_argument('--buses-per-line', type=int, default=6,
                       help='Number of buses to allocate per line')
    
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save output files')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics after simulation')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    args = parse_args()
    
    # Load configuration from file if provided
    config_data = {}
    if args.config:
        import yaml
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")
    
    # Extract configuration values with config file override
    if 'simulation' in config_data:
        sim_config = config_data['simulation']
        start_date_str = sim_config.get('start_date', args.start_date)
        end_date_str = sim_config.get('end_date', args.end_date)
        time_step = sim_config.get('time_step', args.time_step)
        buses_per_line = sim_config.get('buses_per_line', args.buses_per_line)
        seed = sim_config.get('seed', args.seed)
    else:
        start_date_str = args.start_date
        end_date_str = args.end_date
        time_step = args.time_step
        buses_per_line = args.buses_per_line
        seed = args.seed
    
    if 'data' in config_data:
        data_config = config_data['data']
        stops_file = data_config.get('stops_file', args.stops_file)
    else:
        stops_file = args.stops_file
    
    if 'output' in config_data:
        output_config = config_data['output']
        output_dir = output_config.get('directory', args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Print banner
    print("\n" + "="*60)
    print(" Bus Transit Simulator ".center(60, "="))
    print("="*60 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse dates
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    end_date = end_date.replace(hour=23, minute=59)  # End at the end of the day
    
    # Print simulation parameters
    print(f"Simulation Parameters:")
    print(f"  - Transit Data: {stops_file}")
    print(f"  - Period: {start_date_str} to {end_date_str}")
    print(f"  - Time Step: {time_step} minutes")
    print(f"  - Buses per Line: {buses_per_line}")
    print(f"  - Random Seed: {seed}")
    print()
    
    # Create simulation configuration
    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        time_step=time_step,
        randomize_travel_times=True,
        randomize_passenger_demand=True,
        weather_effects_probability=0.15,
        seed=seed
    )
    
    # Record start time
    overall_start_time = time.time()
    
    # Create data container
    data = BusTransitData()
    
    # Create and initialize the simulation engine
    engine = SimulationEngine(data, config)
    
    # Set debug flag if requested
    if args.debug:
        engine.debug = True
    
    print("Step 1: Loading transit data...")
    start_time = time.time()
    success = engine.load_data(stops_file)
    if not success:
        print("Failed to load data. Exiting.")
        return
    print(f"Data loaded in {time.time() - start_time:.2f} seconds.\n")
    
    print(f"Step 2: Setting up simulation...")
    start_time = time.time()
    success = engine.setup_simulation(buses_per_line)
    if not success:
        print("Failed to set up simulation. Exiting.")
        return
    print(f"Setup completed in {time.time() - start_time:.2f} seconds.\n")
    
    print(f"Step 3: Running simulation...")
    start_time = time.time()
    results = engine.run_simulation()
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.\n")
    
    print(f"Step 4: Exporting results...")
    start_time = time.time()
    engine.export_results(output_dir)
    print(f"Results exported in {time.time() - start_time:.2f} seconds.\n")
    
    if args.summary:
        print("\nSummary Statistics:")
        summary = engine.get_summary_statistics()
        
        if "status" in summary and summary["status"] == "No results available":
            print("No results available for summary.")
        else:
            print(f"Total passenger records: {summary['total_passenger_records']}")
            print(f"Total boardings: {summary['total_boardings']}")
            print(f"Total alightings: {summary['total_alightings']}")
            
            # Calculate boarding/alighting ratio
            if summary['total_alightings'] > 0:
                ratio = summary['total_boardings'] / summary['total_alightings']
                print(f"Boarding to alighting ratio: {ratio:.2f}")
            
            print(f"Overcrowded instances (>100%): {summary['overcrowded_records']} ({summary['overcrowding_percentage']:.1f}%)")
            print(f"Severely overcrowded (>150%): {summary['severe_overcrowding_records']} ({summary['severe_overcrowding_percentage']:.1f}%)")
            
            print("\nOvercrowding by line:")
            for line_id, stats in summary['overcrowding_by_line'].items():
                if stats['total_records'] > 0:
                    print(f"  Line {line_id}: {stats['overcrowded_records']} out of {stats['total_records']} records ({stats['percentage']:.1f}%)")
            
            print("\nMost overcrowded buses:")
            for i, bus in enumerate(summary['max_loads'][:5]):
                print(f"  {i+1}. Bus {bus['bus_id']} on line {bus['line_id']}: {bus['occupancy_rate']:.2f} occupancy ratio (max load: {bus['new_load']})")
            
            # Save summary to JSON
            with open(f"{output_dir}/summary_statistics.json", 'w') as f:
                json.dump(summary, f, indent=2)
    
    # Generate visualization instructions
    viz_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization.py')
    if os.path.exists(viz_script_path):
        print("\nTo visualize results, run:")
        print(f"python visualization.py --results-dir {output_dir}")
    
    # Print overall time
    overall_time = time.time() - overall_start_time
    print(f"\nTotal processing time: {overall_time:.2f} seconds")
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()