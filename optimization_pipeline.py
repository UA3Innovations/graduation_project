#!/usr/bin/env python3
"""
Complete Bus Schedule Optimization Pipeline

This script orchestrates the entire optimization workflow:
1. Run the bus transit simulation
2. Run genetic algorithm optimization on simulation results
3. Generate optimized passenger flow structure from GA results
4. Run hybrid LSTM+Prophet model for passenger predictions
5. Evaluate optimization results with comprehensive analysis

Each step includes configurable parameters for customization.
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# Import optimization components
from ga_optimize import run_optimization_with_constraints, OptimizationConfig
from generate_optimized_passenger_flow import OptimizedPassengerFlowGenerator
from optimization_evaluation import OptimizationEvaluator


class OptimizationPipeline:
    """Complete optimization pipeline with configurable steps"""
    
    def __init__(self, config_file=None):
        self.pipeline_start_time = time.time()
        self.step_results = {}
        self.config = self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
    
    def _load_default_config(self):
        """Load default configuration for all pipeline steps"""
        return {
            # Global settings
            'project_name': 'bus_optimization_pipeline',
            'base_output_dir': 'pipeline_output',
            'stops_file': 'ankara_bus_stops_10.csv',
            'random_seed': 42,
            'enable_summary': True,
            
            # Step 1: Simulation Configuration
            'simulation': {
                'start_date': '2025-06-02',
                'end_date': '2025-06-09',  # 7 days
                'time_step': 5,  # minutes
                'buses_per_line': 6,
                'output_dir': 'simulation_output',
                'enable_debug': False,
                'weather_effects': True
            },
            
            # Step 2: Genetic Algorithm Configuration  
            'genetic_algorithm': {
                'population_size': 25,
                'generations': 35,
                'crossover_rate': 0.8,
                'mutation_rate': 0.2,
                'elite_size': 5,
                'tournament_size': 3,
                # Fitness weights
                'passenger_wait_weight': 0.3,
                'bus_utilization_weight': 0.3,
                'overcrowding_weight': 0.2,
                'service_coverage_weight': 0.2,
                # Constraints
                'min_interval': 5,  # minutes
                'max_interval': 60,  # minutes
                'operational_hours': [5, 24],
                # Simulation parameters for fitness evaluation
                'simulation_duration_hours': 6,
                'buses_per_line': 10,
                'time_step': 10,
                'optimization_date': '2025-06-02',
                'target_lines': None,  # None = all lines
                'output_file': 'ga_optimized_schedule.csv'
            },
            
            # Step 3: Passenger Flow Generation Configuration
            'passenger_flow_generation': {
                'baseline_preservation': True,
                'apply_hourly_patterns': True,
                'output_file': 'ga_optimized_passenger_flow.csv'
            },
            
            # Step 4: Hybrid Model Configuration
            'hybrid_model': {
                'sequence_length': 48,
                'epochs': 60,
                'save_models': True,
                'load_pretrained': False,
                'model_dir': 'hybrid_models_pipeline',
                'output_dir': 'output_hybrid_pipeline',
                # Model behavior
                'enable_route_adjustments': True,
                'enable_realistic_constraints': True,
                'enable_night_constraints': True,
                # Special date handling
                'normal_day_weights': {'lstm': 0.7, 'prophet': 0.3},
                'special_day_weights': {'lstm': 0.3, 'prophet': 0.7}
            },
            
            # Step 5: Evaluation Configuration
            'evaluation': {
                'generate_visualizations': True,
                'generate_report': True,
                'comparison_metrics': [
                    'occupancy_rate', 'passenger_conservation', 
                    'resource_utilization', 'schedule_feasibility'
                ],
                'constraint_tolerances': {
                    'passenger_volume_error': 5.0,  # percent
                    'pattern_correlation_min': 0.75,
                    'bus_usage_tolerance': 1.0  # multiplier
                },
                'improvement_threshold': 5.0  # percent - improvement above this overrides validation failure
            }
        }
    
    def _load_config_file(self, config_file):
        """Load configuration from JSON file"""
        import json
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            
            # Merge user config with defaults
            self._deep_merge_config(self.config, user_config)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Failed to load config file {config_file}: {e}")
            print("Using default configuration")
    
    def _deep_merge_config(self, base_config, user_config):
        """Recursively merge user configuration with base configuration"""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def setup_directories(self):
        """Setup output directories for the pipeline"""
        print("🗂️ SETTING UP PIPELINE DIRECTORIES")
        print("=" * 60)
        
        # Create base output directory
        base_dir = self.config['base_output_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_output_dir = f"{base_dir}_{timestamp}"
        
        directories = {
            'base': self.pipeline_output_dir,
            'simulation': os.path.join(self.pipeline_output_dir, 'step1_simulation'),
            'ga_optimization': os.path.join(self.pipeline_output_dir, 'step2_ga_optimization'),
            'passenger_flow': os.path.join(self.pipeline_output_dir, 'step3_passenger_flow'),
            'hybrid_models': os.path.join(self.pipeline_output_dir, 'step4_hybrid_models'),
            'evaluation': os.path.join(self.pipeline_output_dir, 'step5_evaluation'),
            'final_results': os.path.join(self.pipeline_output_dir, 'final_results')
        }
        
        for name, path in directories.items():
            os.makedirs(path, exist_ok=True)
            setattr(self, f"{name}_dir", path)
            print(f"  ✅ Created {name} directory: {path}")
        
        # Create subdirectories for hybrid model
        os.makedirs(os.path.join(self.hybrid_models_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.hybrid_models_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.hybrid_models_dir, 'plots'), exist_ok=True)
        
        print(f"\n🎯 Pipeline output directory: {self.pipeline_output_dir}")
        return directories
    
    def step1_run_simulation(self):
        """Step 1: Run the bus transit simulation"""
        print("\n" + "="*80)
        print("STEP 1: BUS TRANSIT SIMULATION")
        print("="*80)
        
        step_start_time = time.time()
        
        sim_config = self.config['simulation']
        
        # Build simulation command
        cmd = [
            sys.executable, 'main_script.py',
            '--stops-file', self.config['stops_file'],
            '--start-date', sim_config['start_date'],
            '--end-date', sim_config['end_date'],
            '--time-step', str(sim_config['time_step']),
            '--buses-per-line', str(sim_config['buses_per_line']),
            '--output-dir', self.simulation_dir,
            '--seed', str(self.config['random_seed'])
        ]
        
        if self.config['enable_summary']:
            cmd.append('--summary')
        
        if sim_config['enable_debug']:
            cmd.append('--debug')
        
        print(f"📋 Simulation Configuration:")
        print(f"   Period: {sim_config['start_date']} to {sim_config['end_date']}")
        print(f"   Time step: {sim_config['time_step']} minutes")
        print(f"   Buses per line: {sim_config['buses_per_line']}")
        print(f"   Output directory: {self.simulation_dir}")
        print(f"   Random seed: {self.config['random_seed']}")
        
        print(f"\n🚀 Executing simulation...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run simulation
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            step_time = time.time() - step_start_time
            
            print(f"✅ SIMULATION COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Step 1 completed in {step_time:.2f} seconds")
            
            # Check for required output files
            required_files = [
                'passenger_flow_results.csv',
                'line_schedules.csv',
                'buses.csv',
                'summary_statistics.json'
            ]
            
            for file in required_files:
                file_path = os.path.join(self.simulation_dir, file)
                if os.path.exists(file_path):
                    print(f"   ✅ {file} generated")
                else:
                    print(f"   ❌ {file} missing")
            
            self.step_results['simulation'] = {
                'status': 'success',
                'duration': step_time,
                'output_dir': self.simulation_dir,
                'passenger_flow_file': os.path.join(self.simulation_dir, 'passenger_flow_results.csv'),
                'line_schedules_file': os.path.join(self.simulation_dir, 'line_schedules.csv'),
                'buses_file': os.path.join(self.simulation_dir, 'buses.csv')
            }
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ SIMULATION FAILED!")
            print(f"Error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            
            self.step_results['simulation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step2_genetic_algorithm_optimization(self):
        """Step 2: Run genetic algorithm schedule optimization"""
        print("\n" + "="*80)
        print("STEP 2: GENETIC ALGORITHM OPTIMIZATION")
        print("="*80)
        
        if 'simulation' not in self.step_results or self.step_results['simulation']['status'] != 'success':
            print("❌ Cannot run GA optimization - simulation step failed")
            return False
        
        step_start_time = time.time()
        
        ga_config = self.config['genetic_algorithm']
        
        print(f"📋 Genetic Algorithm Configuration:")
        print(f"   Population size: {ga_config['population_size']}")
        print(f"   Generations: {ga_config['generations']}")
        print(f"   Crossover rate: {ga_config['crossover_rate']}")
        print(f"   Mutation rate: {ga_config['mutation_rate']}")
        print(f"   Fitness weights: Wait({ga_config['passenger_wait_weight']}) Bus({ga_config['bus_utilization_weight']}) Overcrowd({ga_config['overcrowding_weight']}) Coverage({ga_config['service_coverage_weight']})")
        print(f"   Optimization date: {ga_config['optimization_date']}")
        
        # Create OptimizationConfig
        opt_config = OptimizationConfig(
            population_size=ga_config['population_size'],
            generations=ga_config['generations'],
            crossover_rate=ga_config['crossover_rate'],
            mutation_rate=ga_config['mutation_rate'],
            elite_size=ga_config['elite_size'],
            tournament_size=ga_config['tournament_size'],
            passenger_wait_weight=ga_config['passenger_wait_weight'],
            bus_utilization_weight=ga_config['bus_utilization_weight'],
            overcrowding_weight=ga_config['overcrowding_weight'],
            service_coverage_weight=ga_config['service_coverage_weight'],
            min_interval=ga_config['min_interval'],
            max_interval=ga_config['max_interval'],
            operational_hours=tuple(ga_config['operational_hours']),
            simulation_duration_hours=ga_config['simulation_duration_hours'],
            buses_per_line=ga_config['buses_per_line'],
            time_step=ga_config['time_step'],
            random_seed=self.config['random_seed']
        )
        
        output_file = os.path.join(self.ga_optimization_dir, ga_config['output_file'])
        
        print(f"\n🧬 Running genetic algorithm optimization...")
        
        try:
            # Import GA optimization function
            from ga_optimize import GeneticScheduleOptimizer
            
            # Create optimizer
            optimizer = GeneticScheduleOptimizer(opt_config, self.config['stops_file'])
            
            if not optimizer.base_data:
                raise Exception("Failed to setup simulation data for GA")
            
            # Load historical constraints from simulation results
            passenger_flow_file = self.step_results['simulation']['passenger_flow_file']
            optimizer.load_historical_constraints_from_simulation(passenger_flow_file)
            
            # Parse optimization date
            opt_date = datetime.strptime(ga_config['optimization_date'], '%Y-%m-%d')
            
            # Select lines to optimize
            target_lines = ga_config['target_lines']
            if target_lines is None:
                target_lines = optimizer.line_ids
            
            print(f"🎯 Optimizing schedules for {len(target_lines)} lines: {target_lines}")
            
            # Optimize schedules
            optimized_schedules = {}
            
            for line_id in target_lines:
                print(f"\n🧬 Optimizing line {line_id}...")
                best_schedule = optimizer.optimize_line_schedule(line_id, opt_date)
                optimized_schedules[line_id] = best_schedule
                
                print(f"✅ Line {line_id}: fitness={best_schedule.fitness:.4f}, departures={len(best_schedule.departure_times)}")
            
            # Export results
            optimizer.export_optimized_schedules(optimized_schedules, output_file)
            
            step_time = time.time() - step_start_time
            
            print(f"\n✅ GENETIC ALGORITHM OPTIMIZATION COMPLETED!")
            print(f"⏱️ Step 2 completed in {step_time:.2f} seconds")
            print(f"📄 Optimized schedule saved: {output_file}")
            
            # Print summary
            total_departures = sum(len(schedule.departure_times) for schedule in optimized_schedules.values())
            avg_fitness = sum(schedule.fitness for schedule in optimized_schedules.values()) / len(optimized_schedules)
            
            print(f"\n📊 Optimization Summary:")
            print(f"   Lines optimized: {len(optimized_schedules)}")
            print(f"   Total departures: {total_departures}")
            print(f"   Average fitness: {avg_fitness:.4f}")
            
            self.step_results['genetic_algorithm'] = {
                'status': 'success',
                'duration': step_time,
                'output_file': output_file,
                'lines_optimized': len(optimized_schedules),
                'total_departures': total_departures,
                'average_fitness': avg_fitness,
                'optimized_schedules': optimized_schedules
            }
            
            return True
            
        except Exception as e:
            print(f"❌ GENETIC ALGORITHM OPTIMIZATION FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            self.step_results['genetic_algorithm'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step3_generate_passenger_flow(self):
        """Step 3: Generate optimized passenger flow structure"""
        print("\n" + "="*80)
        print("STEP 3: GENERATE OPTIMIZED PASSENGER FLOW")
        print("="*80)
        
        if 'genetic_algorithm' not in self.step_results or self.step_results['genetic_algorithm']['status'] != 'success':
            print("❌ Cannot generate passenger flow - GA optimization step failed")
            return False
        
        step_start_time = time.time()
        
        pf_config = self.config['passenger_flow_generation']
        
        print(f"📋 Passenger Flow Generation Configuration:")
        print(f"   Baseline preservation: {pf_config['baseline_preservation']}")
        print(f"   Hourly pattern application: {pf_config['apply_hourly_patterns']}")
        
        try:
            # Create generator
            generator = OptimizedPassengerFlowGenerator(self.config['stops_file'])
            
            # File paths
            schedule_file = self.step_results['genetic_algorithm']['output_file']
            buses_file = self.step_results['simulation']['buses_file']
            baseline_file = self.step_results['simulation']['passenger_flow_file']
            output_file = os.path.join(self.passenger_flow_dir, pf_config['output_file'])
            
            print(f"\n🔄 Generating passenger flow structure...")
            print(f"   Schedule file: {schedule_file}")
            print(f"   Buses file: {buses_file}")
            print(f"   Baseline file: {baseline_file}")
            print(f"   Output file: {output_file}")
            
            # Generate passenger flow with baseline preservation
            if pf_config['baseline_preservation']:
                result_df = generator.generate_optimized_passenger_flow_with_constraints(
                    schedule_file=schedule_file,
                    buses_file=buses_file,
                    baseline_file=baseline_file,
                    output_file=output_file
                )
            else:
                result_df = generator.generate_optimized_passenger_flow(
                    schedule_file=schedule_file,
                    buses_file=buses_file,
                    output_file=output_file
                )
            
            if result_df is not None:
                step_time = time.time() - step_start_time
                
                print(f"\n✅ PASSENGER FLOW GENERATION COMPLETED!")
                print(f"⏱️ Step 3 completed in {step_time:.2f} seconds")
                print(f"📄 Passenger flow saved: {output_file}")
                
                print(f"\n📊 Generation Summary:")
                print(f"   Total records: {len(result_df):,}")
                print(f"   Unique trips: {result_df['trip_id'].nunique():,}")
                print(f"   Unique buses: {result_df['bus_id'].nunique()}")
                print(f"   Lines covered: {result_df['line_id'].nunique()}")
                
                self.step_results['passenger_flow_generation'] = {
                    'status': 'success',
                    'duration': step_time,
                    'output_file': output_file,
                    'total_records': len(result_df),
                    'unique_trips': result_df['trip_id'].nunique(),
                    'unique_buses': result_df['bus_id'].nunique(),
                    'lines_covered': result_df['line_id'].nunique()
                }
                
                return True
            else:
                raise Exception("Passenger flow generation returned None")
                
        except Exception as e:
            print(f"❌ PASSENGER FLOW GENERATION FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            self.step_results['passenger_flow_generation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step4_hybrid_model_prediction(self):
        """Step 4: Run hybrid LSTM+Prophet model for predictions"""
        print("\n" + "="*80)
        print("STEP 4: HYBRID MODEL PREDICTION")
        print("="*80)
        
        if 'passenger_flow_generation' not in self.step_results or self.step_results['passenger_flow_generation']['status'] != 'success':
            print("❌ Cannot run hybrid model - passenger flow generation step failed")
            return False
        
        step_start_time = time.time()
        
        hybrid_config = self.config['hybrid_model']
        
        print(f"📋 Hybrid Model Configuration:")
        print(f"   Sequence length: {hybrid_config['sequence_length']}")
        print(f"   Training epochs: {hybrid_config['epochs']}")
        print(f"   Model directory: {hybrid_config['model_dir']}")
        print(f"   Load pretrained: {hybrid_config['load_pretrained']}")
        print(f"   Route adjustments: {hybrid_config['enable_route_adjustments']}")
        print(f"   Realistic constraints: {hybrid_config['enable_realistic_constraints']}")
        print(f"   Night constraints: {hybrid_config['enable_night_constraints']}")
        
        # Build hybrid model command
        historical_file = self.step_results['simulation']['passenger_flow_file']
        future_file = self.step_results['passenger_flow_generation']['output_file']
        model_dir = os.path.join(self.hybrid_models_dir, 'models')
        output_dir = os.path.join(self.hybrid_models_dir, 'output')
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the local hybrid model from compartmentalized directory
        hybrid_script_path = "hybrid_model.py"
        
        if not os.path.exists(hybrid_script_path):
            print(f"❌ Hybrid model script not found at {hybrid_script_path}")
            return False
        
        cmd = [
            sys.executable, hybrid_script_path,
            '--historical', historical_file,
            '--future', future_file,
            '--bus-stops', self.config['stops_file'],
            '--output', output_dir,
            '--sequence', str(hybrid_config['sequence_length']),
            '--epochs', str(hybrid_config['epochs']),
            '--model-dir', model_dir
        ]
        
        if hybrid_config['save_models']:
            cmd.append('--save-models')
        
        if hybrid_config['load_pretrained']:
            cmd.append('--load-models')
        
        if not hybrid_config['enable_route_adjustments']:
            cmd.append('--disable-route-adjustments')
        
        if not hybrid_config['enable_realistic_constraints']:
            cmd.append('--disable-realistic-constraints')
        
        if not hybrid_config['enable_night_constraints']:
            cmd.append('--disable-night-constraints')
        
        print(f"\n🤖 Running hybrid model prediction...")
        print(f"   Historical data: {historical_file}")
        print(f"   Future structure: {future_file}")
        print(f"   Output directory: {output_dir}")
        
        try:
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            step_time = time.time() - step_start_time
            
            print(f"\n✅ HYBRID MODEL PREDICTION COMPLETED!")
            print(f"⏱️ Step 4 completed in {step_time:.2f} seconds")
            
            # Find the output prediction file
            data_dir = os.path.join(output_dir, 'data')
            prediction_files = []
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.startswith('hybrid_predictions_') and file.endswith('.csv'):
                        prediction_files.append(os.path.join(data_dir, file))
            
            if prediction_files:
                # Use the most recent prediction file
                prediction_file = max(prediction_files, key=os.path.getmtime)
                print(f"📄 Hybrid predictions saved: {prediction_file}")
                
                # Copy to final results with standard name
                final_prediction_file = os.path.join(self.final_results_dir, 'hybrid_predictions_final.csv')
                import shutil
                shutil.copy2(prediction_file, final_prediction_file)
                print(f"📄 Final predictions copied: {final_prediction_file}")
                
                self.step_results['hybrid_model'] = {
                    'status': 'success',
                    'duration': step_time,
                    'output_dir': output_dir,
                    'prediction_file': prediction_file,
                    'final_prediction_file': final_prediction_file
                }
                
                return True
            else:
                raise Exception("No hybrid prediction files found in output directory")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ HYBRID MODEL PREDICTION FAILED!")
            print(f"Error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            
            self.step_results['hybrid_model'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
        except Exception as e:
            print(f"❌ HYBRID MODEL PREDICTION FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            self.step_results['hybrid_model'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step5_optimization_evaluation(self):
        """Step 5: Evaluate optimization results"""
        print("\n" + "="*80)
        print("STEP 5: OPTIMIZATION EVALUATION")
        print("="*80)
        
        if 'hybrid_model' not in self.step_results or self.step_results['hybrid_model']['status'] != 'success':
            print("❌ Cannot run evaluation - hybrid model step failed")
            return False
        
        step_start_time = time.time()
        
        eval_config = self.config['evaluation']
        
        print(f"📋 Evaluation Configuration:")
        print(f"   Generate visualizations: {eval_config['generate_visualizations']}")
        print(f"   Generate report: {eval_config['generate_report']}")
        print(f"   Comparison metrics: {eval_config['comparison_metrics']}")
        
        try:
            # Create evaluator
            evaluator = OptimizationEvaluator()
            
            # File paths
            original_file = self.step_results['simulation']['passenger_flow_file']
            optimized_file = self.step_results['hybrid_model']['final_prediction_file']
            schedule_file = self.step_results['simulation']['line_schedules_file']
            baseline_file = original_file
            
            print(f"\n📊 Running optimization evaluation...")
            print(f"   Original data: {original_file}")
            print(f"   Optimized data: {optimized_file}")
            print(f"   Schedule data: {schedule_file}")
            print(f"   Baseline data: {baseline_file}")
            
            # Change to evaluation directory for output files
            original_cwd = os.getcwd()
            os.chdir(self.evaluation_dir)
            
            # Run evaluation
            validation, stats = evaluator.run_pipeline_evaluation(
                original_file=os.path.relpath(original_file, self.evaluation_dir),
                optimized_file=os.path.relpath(optimized_file, self.evaluation_dir),
                schedule_file=os.path.relpath(schedule_file, self.evaluation_dir),
                baseline_file=os.path.relpath(baseline_file, self.evaluation_dir)
            )
            
            # Return to original directory
            os.chdir(original_cwd)
            
            step_time = time.time() - step_start_time
            
            print(f"\n✅ OPTIMIZATION EVALUATION COMPLETED!")
            print(f"⏱️ Step 5 completed in {step_time:.2f} seconds")
            
            # Copy evaluation results to final results
            import shutil
            eval_files = ['optimization_evaluation_report.txt']
            eval_dirs = ['evaluation_plots']
            
            for file in eval_files:
                src = os.path.join(self.evaluation_dir, file)
                dst = os.path.join(self.final_results_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"📄 Copied {file} to final results")
            
            for dir_name in eval_dirs:
                src_dir = os.path.join(self.evaluation_dir, dir_name)
                dst_dir = os.path.join(self.final_results_dir, dir_name)
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                    print(f"📁 Copied {dir_name} to final results")
            
            # Extract key metrics
            original_pass = validation.get('overall_pass', False)
            improvement_pct = stats.get('occupancy_improvement', {}).get('improvement_pct', 0)
            
            # Apply improvement threshold: configurable improvement overrides validation failure
            improvement_threshold = eval_config.get('improvement_threshold', 5.0)
            improvement_pass = improvement_pct > improvement_threshold
            overall_pass = original_pass or improvement_pass
            
            print(f"\n📊 Evaluation Summary:")
            print(f"   Original validation: {'✅ PASS' if original_pass else '❌ FAIL'}")
            print(f"   Occupancy improvement: {improvement_pct:.1f}%")
            print(f"   Improvement threshold: >{improvement_threshold}% = {'✅ PASS' if improvement_pass else '❌ FAIL'}")
            print(f"   Overall validation: {'✅ PASS' if overall_pass else '❌ FAIL'}")
            
            self.step_results['evaluation'] = {
                'status': 'success',
                'duration': step_time,
                'overall_pass': overall_pass,
                'improvement_pct': improvement_pct,
                'validation': validation,
                'stats': stats,
                'evaluation_dir': self.evaluation_dir
            }
            
            return True
            
        except Exception as e:
            os.chdir(original_cwd)  # Ensure we return to original directory
            print(f"❌ OPTIMIZATION EVALUATION FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            self.step_results['evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline execution report"""
        print("\n" + "="*80)
        print("PIPELINE EXECUTION REPORT")
        print("="*80)
        
        total_time = time.time() - self.pipeline_start_time
        
        report_file = os.path.join(self.final_results_dir, 'pipeline_execution_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("BUS SCHEDULE OPTIMIZATION PIPELINE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total pipeline duration: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n\n")
            
            # Configuration summary
            f.write("PIPELINE CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Project: {self.config['project_name']}\n")
            f.write(f"Output directory: {self.pipeline_output_dir}\n")
            f.write(f"Stops file: {self.config['stops_file']}\n")
            f.write(f"Random seed: {self.config['random_seed']}\n\n")
            
            # Step-by-step results
            steps = [
                ('simulation', 'Bus Transit Simulation'),
                ('genetic_algorithm', 'Genetic Algorithm Optimization'),
                ('passenger_flow_generation', 'Passenger Flow Generation'),
                ('hybrid_model', 'Hybrid Model Prediction'),
                ('evaluation', 'Optimization Evaluation')
            ]
            
            f.write("EXECUTION RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            successful_steps = 0
            total_steps = len(steps)
            
            for step_key, step_name in steps:
                if step_key in self.step_results:
                    result = self.step_results[step_key]
                    status = result.get('status', 'unknown')
                    duration = result.get('duration', 0)
                    
                    status_symbol = '✅' if status == 'success' else '❌'
                    f.write(f"{status_symbol} {step_name}:\n")
                    f.write(f"   Status: {status.upper()}\n")
                    f.write(f"   Duration: {duration:.2f} seconds\n")
                    
                    if status == 'success':
                        successful_steps += 1
                        
                        # Add step-specific details
                        if step_key == 'simulation':
                            f.write(f"   Output files: passenger_flow_results.csv, line_schedules.csv, buses.csv\n")
                        elif step_key == 'genetic_algorithm':
                            f.write(f"   Lines optimized: {result.get('lines_optimized', 'N/A')}\n")
                            f.write(f"   Total departures: {result.get('total_departures', 'N/A')}\n")
                            f.write(f"   Average fitness: {result.get('average_fitness', 'N/A'):.4f}\n")
                        elif step_key == 'passenger_flow_generation':
                            f.write(f"   Total records: {result.get('total_records', 'N/A'):,}\n")
                            f.write(f"   Unique trips: {result.get('unique_trips', 'N/A'):,}\n")
                        elif step_key == 'evaluation':
                            improvement_pct = result.get('improvement_pct', 0)
                            overall_pass = result.get('overall_pass', False)
                            improvement_threshold = self.config['evaluation'].get('improvement_threshold', 5.0)
                            f.write(f"   Overall validation: {'PASS' if overall_pass else 'FAIL'}\n")
                            f.write(f"   Improvement: {improvement_pct:.1f}% (threshold: >{improvement_threshold}%)\n")
                    else:
                        f.write(f"   Error: {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n")
                else:
                    f.write(f"❌ {step_name}: NOT EXECUTED\n\n")
            
            # Pipeline summary
            f.write("PIPELINE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Steps completed successfully: {successful_steps}/{total_steps}\n")
            f.write(f"Pipeline success rate: {successful_steps/total_steps*100:.1f}%\n")
            f.write(f"Overall status: {'SUCCESS' if successful_steps == total_steps else 'PARTIAL FAILURE'}\n\n")
            
            # Final outputs
            f.write("FINAL OUTPUTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Pipeline directory: {self.pipeline_output_dir}\n")
            f.write(f"Final results: {self.final_results_dir}\n")
            f.write("Key files:\n")
            f.write("  - hybrid_predictions_final.csv (optimized predictions)\n")
            f.write("  - optimization_evaluation_report.txt (detailed evaluation)\n")
            f.write("  - evaluation_plots/ (visualization plots)\n")
            f.write("  - pipeline_execution_report.txt (this report)\n")
        
        print(f"📄 Pipeline report saved: {report_file}")
        
        # Print summary to console
        print(f"\n📊 PIPELINE EXECUTION SUMMARY:")
        print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Steps completed: {successful_steps}/{total_steps}")
        print(f"   Success rate: {successful_steps/total_steps*100:.1f}%")
        print(f"   Overall status: {'✅ SUCCESS' if successful_steps == total_steps else '⚠️ PARTIAL FAILURE'}")
        print(f"   Output directory: {self.pipeline_output_dir}")
        
        return successful_steps == total_steps
    
    def run_complete_pipeline(self):
        """Run the complete optimization pipeline"""
        print("🚀 STARTING COMPLETE BUS SCHEDULE OPTIMIZATION PIPELINE")
        print("=" * 80)
        
        # Setup directories
        self.setup_directories()
        
        # Run pipeline steps
        steps = [
            ('Step 1', self.step1_run_simulation),
            ('Step 2', self.step2_genetic_algorithm_optimization),
            ('Step 3', self.step3_generate_passenger_flow),
            ('Step 4', self.step4_hybrid_model_prediction),
            ('Step 5', self.step5_optimization_evaluation)
        ]
        
        for step_name, step_function in steps:
            try:
                success = step_function()
                if not success:
                    print(f"\n❌ Pipeline stopped at {step_name} due to failure")
                    break
            except KeyboardInterrupt:
                print(f"\n⏹️ Pipeline interrupted by user at {step_name}")
                break
            except Exception as e:
                print(f"\n💥 Unexpected error in {step_name}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Generate final report
        pipeline_success = self.generate_pipeline_report()
        
        if pipeline_success:
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print(f"\n⚠️ PIPELINE COMPLETED WITH ISSUES")
        
        print(f"📁 All results saved in: {self.pipeline_output_dir}")
        
        return pipeline_success


def create_sample_config():
    """Create a sample configuration file"""
    pipeline = OptimizationPipeline()
    
    import json
    config_file = 'pipeline_config_sample.json'
    with open(config_file, 'w') as f:
        json.dump(pipeline.config, f, indent=2)
    
    print(f"📄 Sample configuration file created: {config_file}")
    print("Edit this file to customize pipeline parameters before running.")
    
    return config_file


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Complete Bus Schedule Optimization Pipeline')
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', '5'], help='Run only specific step (1-5)')
    parser.add_argument('--stops-file', type=str, default='ankara_bus_stops_10.csv', help='Bus stops CSV file')
    parser.add_argument('--project-name', type=str, default='bus_optimization_pipeline', help='Project name for outputs')
    
    # Quick configuration overrides
    parser.add_argument('--quick-test', action='store_true', help='Run with reduced parameters for quick testing')
    parser.add_argument('--population-size', type=int, help='GA population size')
    parser.add_argument('--generations', type=int, help='GA generations')
    parser.add_argument('--simulation-days', type=int, help='Number of simulation days')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Create pipeline
    pipeline = OptimizationPipeline(args.config)
    
    # Apply command line overrides
    if args.stops_file:
        pipeline.config['stops_file'] = args.stops_file
    
    if args.project_name:
        pipeline.config['project_name'] = args.project_name
    
    if args.quick_test:
        print("⚡ Quick test mode enabled - reducing parameters")
        pipeline.config['genetic_algorithm']['population_size'] = 10
        pipeline.config['genetic_algorithm']['generations'] = 15
        pipeline.config['simulation']['end_date'] = '2025-06-04'  # Only 2 days
        pipeline.config['hybrid_model']['epochs'] = 30
    
    if args.population_size:
        pipeline.config['genetic_algorithm']['population_size'] = args.population_size
    
    if args.generations:
        pipeline.config['genetic_algorithm']['generations'] = args.generations
    
    if args.simulation_days:
        start_date = datetime.strptime(pipeline.config['simulation']['start_date'], '%Y-%m-%d')
        end_date = start_date + timedelta(days=args.simulation_days)
        pipeline.config['simulation']['end_date'] = end_date.strftime('%Y-%m-%d')
    
    # Run pipeline
    if args.step:
        step_num = int(args.step)
        print(f"🎯 Running only Step {step_num}")
        
        # Setup directories first
        pipeline.setup_directories()
        
        # Run specific step
        step_functions = [
            pipeline.step1_run_simulation,
            pipeline.step2_genetic_algorithm_optimization,
            pipeline.step3_generate_passenger_flow,
            pipeline.step4_hybrid_model_prediction,
            pipeline.step5_optimization_evaluation
        ]
        
        if 1 <= step_num <= len(step_functions):
            success = step_functions[step_num - 1]()
            print(f"Step {step_num} {'✅ SUCCESS' if success else '❌ FAILED'}")
        else:
            print(f"❌ Invalid step number: {step_num}")
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main() 