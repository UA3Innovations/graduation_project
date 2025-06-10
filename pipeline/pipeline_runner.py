"""
Main pipeline runner for orchestrating bus transit simulation on Azure.
"""

import os
import sys
import yaml
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import BusTransitData, SimulationConfig
from core.simulation_engine import SimulationEngine


class SimulationPipeline:
    """
    Main pipeline orchestrator for running bus transit simulation in cloud environments.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        """
        self.config_path = config_path or "config/simulation_config.yaml"
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize results storage
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            # Return default config
            return {
                'simulation': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-07',
                    'time_step': 5,
                    'buses_per_line': 6,
                    'randomize_travel_times': True,
                    'randomize_passenger_demand': True,
                    'weather_effects_probability': 0.15,
                    'seed': 42
                },
                'data': {
                    'stops_file': 'data/ankara_bus_stops_10.csv'
                },
                'output': {
                    'directory': 'output',
                    'summary': True,
                    'debug': False
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        # Create output directory if it doesn't exist
        output_dir = self.config.get('output', {}).get('directory', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level = logging.DEBUG if self.config.get('output', {}).get('debug', False) else logging.INFO
        
        # File handler
        log_file = os.path.join(output_dir, f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger('SimulationPipeline')
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger
    
    def validate_inputs(self) -> bool:
        """
        Validate input data and configuration.
        
        Returns:
        --------
        bool
            True if validation passes, False otherwise.
        """
        self.logger.info("Starting input validation...")
        
        try:
            # Check data file exists
            data_file = self.config['data']['stops_file']
            if not os.path.exists(data_file):
                self.logger.error(f"Data file not found: {data_file}")
                return False
            
            # Validate simulation parameters
            sim_config = self.config['simulation']
            
            # Check date format and validity
            try:
                start_date = datetime.strptime(sim_config['start_date'], '%Y-%m-%d')
                end_date = datetime.strptime(sim_config['end_date'], '%Y-%m-%d')
                
                if end_date <= start_date:
                    self.logger.error("End date must be after start date")
                    return False
                    
                # Check simulation duration (warn if too long)
                duration = (end_date - start_date).days
                if duration > 30:
                    self.logger.warning(f"Long simulation duration: {duration} days")
                    
            except ValueError as e:
                self.logger.error(f"Invalid date format: {str(e)}")
                return False
            
            # Validate numeric parameters
            if sim_config.get('time_step', 0) <= 0:
                self.logger.error("Time step must be positive")
                return False
                
            if sim_config.get('buses_per_line', 0) <= 0:
                self.logger.error("Buses per line must be positive")
                return False
            
            self.logger.info("Input validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
    
    def run_simulation(self) -> bool:
        """
        Run the main simulation.
        
        Returns:
        --------
        bool
            True if simulation completes successfully, False otherwise.
        """
        self.logger.info("Starting simulation...")
        self.start_time = time.time()
        
        try:
            # Create simulation configuration
            sim_config_dict = self.config['simulation']
            
            start_date = datetime.strptime(sim_config_dict['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(sim_config_dict['end_date'], '%Y-%m-%d')
            end_date = end_date.replace(hour=23, minute=59)  # End at the end of the day
            
            config = SimulationConfig(
                start_date=start_date,
                end_date=end_date,
                time_step=sim_config_dict.get('time_step', 5),
                randomize_travel_times=sim_config_dict.get('randomize_travel_times', True),
                randomize_passenger_demand=sim_config_dict.get('randomize_passenger_demand', True),
                weather_effects_probability=sim_config_dict.get('weather_effects_probability', 0.15),
                seed=sim_config_dict.get('seed', 42)
            )
            
            # Create data container and simulation engine
            data = BusTransitData()
            engine = SimulationEngine(data, config)
            
            # Set debug flag if requested
            if self.config.get('output', {}).get('debug', False):
                engine.debug = True
            
            # Load data
            self.logger.info("Loading transit data...")
            data_file = self.config['data']['stops_file']
            success = engine.load_data(data_file)
            if not success:
                self.logger.error("Failed to load data")
                return False
            
            self.logger.info(f"Loaded {len(data.lines)} lines and {len(data.stops)} stops")
            
            # Setup simulation
            self.logger.info("Setting up simulation...")
            buses_per_line = sim_config_dict.get('buses_per_line', 6)
            success = engine.setup_simulation(buses_per_line)
            if not success:
                self.logger.error("Failed to set up simulation")
                return False
            
            self.logger.info(f"Generated {len(data.buses)} buses")
            
            # Run simulation
            self.logger.info("Running simulation...")
            results_df = engine.run_simulation()
            
            if results_df.empty:
                self.logger.error("Simulation produced no results")
                return False
            
            self.logger.info(f"Simulation completed with {len(results_df)} records")
            
            # Export results
            output_dir = self.config.get('output', {}).get('directory', 'output')
            self.logger.info(f"Exporting results to {output_dir}...")
            engine.export_results(output_dir)
            
            # Store results for pipeline
            self.results = {
                'passenger_flow_records': len(results_df),
                'total_boardings': results_df['boarding'].sum(),
                'total_alightings': results_df['alighting'].sum(),
                'simulation_days': (end_date.date() - start_date.date()).days + 1,
                'output_directory': output_dir
            }
            
            # Generate summary if requested
            if self.config.get('output', {}).get('summary', True):
                summary = engine.get_summary_statistics()
                self.results['summary'] = summary
                self.logger.info("Summary statistics generated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def post_process_results(self) -> bool:
        """
        Post-process simulation results.
        
        Returns:
        --------
        bool
            True if post-processing succeeds, False otherwise.
        """
        self.logger.info("Starting post-processing...")
        
        try:
            output_dir = self.config.get('output', {}).get('directory', 'output')
            
            # Validate output files exist
            expected_files = [
                'passenger_flow_results.csv',
                'bus_positions_results.csv',
                'buses.csv',
                'line_schedules.csv',
                'bus_assignments.csv',
                'summary_statistics.json'
            ]
            
            missing_files = []
            for file in expected_files:
                file_path = os.path.join(output_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                self.logger.warning(f"Missing output files: {missing_files}")
            else:
                self.logger.info("All expected output files present")
            
            # Calculate file sizes
            total_size = 0
            for file in expected_files:
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    self.logger.info(f"{file}: {size / 1024 / 1024:.2f} MB")
            
            self.results['total_output_size_mb'] = total_size / 1024 / 1024
            
            self.logger.info("Post-processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Post-processing error: {str(e)}")
            return False
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
        --------
        Dict[str, Any]
            Pipeline results and metadata.
        """
        pipeline_start_time = time.time()
        self.logger.info("="*60)
        self.logger.info(" Bus Transit Simulation Pipeline ".center(60, "="))
        self.logger.info("="*60)
        
        # Stage 1: Validate inputs
        self.logger.info("Stage 1: Input Validation")
        if not self.validate_inputs():
            return {
                'success': False,
                'stage': 'validation',
                'error': 'Input validation failed',
                'duration_seconds': time.time() - pipeline_start_time
            }
        
        # Stage 2: Run simulation
        self.logger.info("Stage 2: Simulation Execution")
        if not self.run_simulation():
            return {
                'success': False,
                'stage': 'simulation',
                'error': 'Simulation execution failed',
                'duration_seconds': time.time() - pipeline_start_time
            }
        
        # Stage 3: Post-process results
        self.logger.info("Stage 3: Post-Processing")
        if not self.post_process_results():
            return {
                'success': False,
                'stage': 'post_processing',
                'error': 'Post-processing failed',
                'duration_seconds': time.time() - pipeline_start_time
            }
        
        # Pipeline completed successfully
        pipeline_duration = time.time() - pipeline_start_time
        
        self.logger.info("="*60)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info(f"Total duration: {pipeline_duration:.2f} seconds")
        self.logger.info("="*60)
        
        # Return comprehensive results
        return {
            'success': True,
            'duration_seconds': pipeline_duration,
            'results': self.results,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main function for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Bus Transit Simulation Pipeline')
    parser.add_argument('--config', type=str, default='config/simulation_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = SimulationPipeline(config_path=args.config)
    results = pipeline.run_pipeline()
    
    # Exit with appropriate code
    if results['success']:
        print(f"Pipeline completed successfully in {results['duration_seconds']:.2f} seconds")
        sys.exit(0)
    else:
        print(f"Pipeline failed at stage '{results['stage']}': {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main() 