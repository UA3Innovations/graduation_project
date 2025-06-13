#!/usr/bin/env python3
"""
Optimization Pipeline Runner

Main entry point for running bus schedule optimization using genetic algorithms.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import optimization components
from optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig, run_optimization
from optimization.optimization_evaluation import OptimizationEvaluator


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}


def validate_config(config: dict) -> bool:
    """Validate the configuration."""
    required_sections = ['optimization', 'data', 'output']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required config section: {section}")
            return False
    
    # Check for required optimization parameters
    opt_config = config['optimization']
    required_opt_params = ['population_size', 'generations', 'target_date']
    
    for param in required_opt_params:
        if param not in opt_config:
            print(f"Missing required optimization parameter: {param}")
            return False
    
    # Check data file exists
    data_config = config['data']
    if 'stops_file' not in data_config:
        print("Missing required data parameter: stops_file")
        return False
    
    stops_file = data_config['stops_file']
    if not os.path.exists(stops_file):
        print(f"Data file not found: {stops_file}")
        return False
    
    return True


def run_optimization_pipeline(config: dict) -> bool:
    """Run the complete optimization pipeline."""
    print("🧬 Bus Schedule Optimization Pipeline")
    print("=" * 60)
    
    # Extract configuration
    opt_config = config['optimization']
    data_config = config['data']
    output_config = config['output']
    
    # Create optimization configuration
    optimization_config = OptimizationConfig(
        population_size=opt_config.get('population_size', 30),
        generations=opt_config.get('generations', 50),
        crossover_rate=opt_config.get('crossover_rate', 0.8),
        mutation_rate=opt_config.get('mutation_rate', 0.2),
        elite_size=opt_config.get('elite_size', 5),
        tournament_size=opt_config.get('tournament_size', 3),
        passenger_wait_weight=opt_config.get('passenger_wait_weight', 0.4),
        bus_utilization_weight=opt_config.get('bus_utilization_weight', 0.3),
        overcrowding_weight=opt_config.get('overcrowding_weight', 0.2),
        service_coverage_weight=opt_config.get('service_coverage_weight', 0.1),
        min_interval=opt_config.get('min_interval', 5),
        max_interval=opt_config.get('max_interval', 60),
        operational_hours=tuple(opt_config.get('operational_hours', [5, 24])),
        simulation_duration_hours=opt_config.get('simulation_duration_hours', 6),
        buses_per_line=opt_config.get('buses_per_line', 10),
        time_step=opt_config.get('time_step', 10),
        random_seed=opt_config.get('random_seed', 42)
    )
    
    # Parse target date
    target_date_str = opt_config['target_date']
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    
    # Get target lines (if specified)
    target_lines = opt_config.get('target_lines', None)
    
    # Create output directory
    output_dir = output_config.get('directory', 'optimization_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization
    print(f"🎯 Target date: {target_date_str}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Configuration: {optimization_config.population_size} pop, {optimization_config.generations} gen")
    
    try:
        # Use the run_optimization function from ga_optimize
        output_file = os.path.join(output_dir, f"optimized_schedule_{target_date_str}.csv")
        
        result = run_optimization(
            stops_file=data_config['stops_file'],
            optimization_date=target_date_str,
            output_file=output_file,
            target_lines=target_lines
        )
        
        if result:
            print(f"✅ Optimization completed successfully!")
            print(f"📄 Results saved to: {output_file}")
            
            # Run evaluation if requested
            if output_config.get('run_evaluation', False):
                print("\n📊 Running optimization evaluation...")
                
                # Check if we have baseline data for comparison
                baseline_file = data_config.get('baseline_simulation_file')
                if baseline_file and os.path.exists(baseline_file):
                    evaluator = OptimizationEvaluator()
                    
                    # For now, we'll skip the full evaluation since we need baseline data
                    # This would be implemented when we have simulation results to compare against
                    print("⚠️  Evaluation requires baseline simulation data - skipping for now")
                else:
                    print("⚠️  No baseline simulation file provided - skipping evaluation")
            
            return True
        else:
            print("❌ Optimization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run bus schedule optimization pipeline')
    
    parser.add_argument('--config', type=str, default='config/optimization_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-only', action='store_true',
                       help='Only validate configuration without running optimization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Load configuration
    config_file = args.config
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return 1
    
    config = load_config(config_file)
    if not config:
        print("Failed to load configuration")
        return 1
    
    # Override with quick test parameters if requested
    if args.quick_test:
        print("🚀 Quick test mode enabled")
        config['optimization']['population_size'] = 8
        config['optimization']['generations'] = 10
        config['optimization']['simulation_duration_hours'] = 2
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed")
        return 1
    
    print("✅ Configuration validated successfully")
    
    if args.test_only:
        print("Test-only mode - exiting without running optimization")
        return 0
    
    # Run optimization pipeline
    success = run_optimization_pipeline(config)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 