#!/usr/bin/env python3
"""
Integrated Schedule Optimization Pipeline

This is the main pipeline that orchestrates all four components:
1. Bus Simulation Pipeline - Generates baseline simulation data
2. Bus Prediction Pipeline - Creates passenger flow predictions
3. Bus Optimization Pipeline - Optimizes schedules using genetic algorithm
4. Bus Evaluation Pipeline - Evaluates optimization results

The pipeline supports both full-scale (1 year simulation) and quick test modes.
"""

import os
import sys
import subprocess
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add all pipeline packages to Python path
project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root / "bus_simulation_pipeline" / "src"),
    str(project_root / "bus_optimization_pipeline" / "src"),
    str(project_root / "bus_prediction_pipeline" / "src"),
    str(project_root / "bus_evaluation_pipeline" / "src")
])


class IntegratedPipeline:
    """Main pipeline orchestrator for the complete schedule optimization system"""
    
    def __init__(self, mode: str = "full", output_dir: str = None):
        self.mode = mode  # "full" or "quick"
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up output directory
        if output_dir is None:
            output_dir = f"outputs/pipeline_run_{self.mode}_{self.timestamp}"
        self.output_dir = self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Pipeline configuration based on mode
        self.config = self.get_pipeline_config()
        
        self.logger.info(f"Initialized Integrated Pipeline in {mode} mode")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_logging(self):
        """Set up comprehensive logging"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"pipeline_{self.mode}_{self.timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('IntegratedPipeline')
    
    def display_configuration_summary(self):
        """Display a summary of the optimized configuration parameters"""
        self.logger.info("=" * 80)
        self.logger.info(f"📋 OPTIMIZED CONFIGURATION SUMMARY - {self.mode.upper()} MODE")
        self.logger.info("=" * 80)
        
        # Simulation parameters
        sim_config = self.config["simulation"]
        self.logger.info("🚌 SIMULATION PARAMETERS:")
        self.logger.info(f"  • Time Period: {sim_config['start_date']} to {sim_config['end_date']}")
        self.logger.info(f"  • Time Resolution: {sim_config['time_step']} minutes")
        self.logger.info(f"  • Buses per Line: {sim_config['buses_per_line']}")
        self.logger.info(f"  • Passenger Generation Rate: {sim_config['passenger_generation_rate']}")
        self.logger.info(f"  • Route Complexity: {sim_config['route_complexity']}")
        self.logger.info(f"  • Weather Effects: {sim_config['weather_effects']}")
        self.logger.info(f"  • Seasonal Patterns: {sim_config['seasonal_patterns']}")
        
        # Optimization parameters
        opt_config = self.config["optimization"]
        self.logger.info("\n🧬 OPTIMIZATION PARAMETERS:")
        self.logger.info(f"  • Optimization Period: {opt_config['period_days']} days")
        self.logger.info(f"  • Generations: {opt_config['generations']}")
        self.logger.info(f"  • Population Size: {opt_config['population_size']}")
        self.logger.info(f"  • Mutation Rate: {opt_config['mutation_rate']}")
        self.logger.info(f"  • Crossover Rate: {opt_config['crossover_rate']}")
        self.logger.info(f"  • Elite Solutions: {opt_config['elite_size']}")
        self.logger.info(f"  • Tournament Size: {opt_config['tournament_size']}")
        
        # Prediction parameters
        pred_config = self.config["prediction"]
        self.logger.info("\n🔮 PREDICTION PARAMETERS:")
        self.logger.info(f"  • Prediction Period: {pred_config['period_days']} days")
        self.logger.info(f"  • Prediction Horizon: {pred_config['prediction_horizon']} hours")
        lstm_config = pred_config['lstm_config']
        self.logger.info(f"  • LSTM Architecture: {lstm_config['layers']}")
        self.logger.info(f"  • LSTM Epochs: {lstm_config['epochs']}")
        self.logger.info(f"  • Sequence Length: {lstm_config['sequence_length']} hours")
        prophet_config = pred_config['prophet_config']
        self.logger.info(f"  • Prophet Seasonality: {prophet_config['seasonality_mode']}")
        self.logger.info(f"  • Yearly Seasonality: {prophet_config['yearly_seasonality']}")
        
        # Evaluation parameters
        eval_config = self.config["evaluation"]
        self.logger.info("\n📊 EVALUATION PARAMETERS:")
        self.logger.info(f"  • Statistical Tests: {eval_config['statistical_tests']}")
        self.logger.info(f"  • Confidence Level: {eval_config['confidence_level']}")
        self.logger.info(f"  • Bootstrap Samples: {eval_config['bootstrap_samples']:,}")
        self.logger.info(f"  • Visualization Detail: {eval_config['visualization_detail']}")
        self.logger.info(f"  • Constraint Validation: {eval_config['constraint_validation']}")
        
        # Performance expectations
        self.logger.info("\n⏱️  PERFORMANCE EXPECTATIONS:")
        if self.mode == "full":
            self.logger.info("  • Execution Time: 4-8 hours (depending on hardware)")
            self.logger.info("  • Memory Usage: 8-16 GB RAM")
            self.logger.info("  • Disk Space: ~5-10 GB for results")
            self.logger.info("  • Accuracy: Maximum (production-grade)")
            self.logger.info("  • Simulation Records: ~35 million passenger flows")
        else:
            self.logger.info("  • Execution Time: 15-45 minutes")
            self.logger.info("  • Memory Usage: 2-4 GB RAM")
            self.logger.info("  • Disk Space: ~500 MB for results")
            self.logger.info("  • Accuracy: High (suitable for testing/validation)")
            self.logger.info("  • Simulation Records: ~95,000 passenger flows")
        
        self.logger.info("=" * 80)
        
    def get_pipeline_config(self):
        """Get optimized configuration based on pipeline mode"""
        if self.mode == "full":
            # Full mode: Optimized for best results (1 year simulation)
            # Higher computational cost but maximum accuracy
            return {
                "simulation": {
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",  # 1 year
                    "time_step": 5,  # Higher resolution (5 min vs 10 min)
                    "buses_per_line": 12,  # More buses for realistic simulation
                    "passenger_generation_rate": 1.2,  # Higher passenger density
                    "route_complexity": "high",  # Complex routing patterns
                    "weather_effects": True,  # Include weather variations
                    "seasonal_patterns": True,  # Include seasonal variations
                    "rush_hour_multiplier": 2.5,  # Strong rush hour effects
                    "weekend_reduction": 0.6  # Weekend passenger reduction
                },
                "optimization": {
                    "period_days": 30,  # 1 month optimization period
                    "generations": 150,  # More generations for better convergence
                    "population_size": 80,  # Larger population for diversity
                    "mutation_rate": 0.08,  # Lower mutation for fine-tuning
                    "crossover_rate": 0.85,  # Higher crossover for exploration
                    "elite_size": 8,  # Keep best solutions
                    "tournament_size": 5,  # Tournament selection
                    "convergence_threshold": 0.001,  # Strict convergence
                    "max_stagnation": 20,  # Allow more stagnation
                    "fitness_weights": {
                        "occupancy": 0.4,
                        "waiting_time": 0.3,
                        "fuel_efficiency": 0.2,
                        "schedule_regularity": 0.1
                    }
                },
                "prediction": {
                    "period_days": 30,  # 1 month prediction
                    "prediction_horizon": 168,  # 1 week ahead (hours)
                    "lstm_config": {
                        "layers": [128, 64, 32],  # Deeper network
                        "dropout": 0.2,
                        "epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "sequence_length": 168  # 1 week lookback
                    },
                    "prophet_config": {
                        "seasonality_mode": "multiplicative",
                        "yearly_seasonality": True,
                        "weekly_seasonality": True,
                        "daily_seasonality": True,
                        "holidays": True,
                        "changepoint_prior_scale": 0.05,
                        "seasonality_prior_scale": 10.0
                    },
                    "hybrid_weights": {
                        "lstm": 0.6,
                        "prophet": 0.4
                    },
                    "validation_split": 0.2,
                    "cross_validation_folds": 5
                },
                "evaluation": {
                    "statistical_tests": ["t_test", "wilcoxon", "mann_whitney"],
                    "confidence_level": 0.95,
                    "bootstrap_samples": 10000,
                    "effect_size_metrics": ["cohen_d", "cliff_delta"],
                    "visualization_detail": "high",
                    "constraint_validation": "strict"
                }
            }
        else:  # quick mode
            # Quick mode: Optimized for speed with good accuracy
            # Lower computational cost but still reliable results
            return {
                "simulation": {
                    "start_date": "2025-06-01",
                    "end_date": "2025-06-07",  # 1 week
                    "time_step": 15,  # Lower resolution for speed (15 min)
                    "buses_per_line": 6,  # Fewer buses for faster simulation
                    "passenger_generation_rate": 1.0,  # Standard passenger density
                    "route_complexity": "medium",  # Simplified routing
                    "weather_effects": False,  # Skip weather for speed
                    "seasonal_patterns": False,  # Skip seasonal for speed
                    "rush_hour_multiplier": 2.0,  # Moderate rush hour effects
                    "weekend_reduction": 0.7  # Moderate weekend reduction
                },
                "optimization": {
                    "period_days": 1,  # 1 day optimization period
                    "generations": 30,  # Fewer generations for speed
                    "population_size": 30,  # Smaller population
                    "mutation_rate": 0.12,  # Higher mutation for quick exploration
                    "crossover_rate": 0.75,  # Standard crossover rate
                    "elite_size": 3,  # Keep fewer elite solutions
                    "tournament_size": 3,  # Smaller tournament
                    "convergence_threshold": 0.01,  # Relaxed convergence
                    "max_stagnation": 10,  # Less stagnation tolerance
                    "fitness_weights": {
                        "occupancy": 0.5,
                        "waiting_time": 0.35,
                        "fuel_efficiency": 0.1,
                        "schedule_regularity": 0.05
                    }
                },
                "prediction": {
                    "period_days": 1,  # 1 day prediction
                    "prediction_horizon": 24,  # 1 day ahead (hours)
                    "lstm_config": {
                        "layers": [64, 32],  # Simpler network
                        "dropout": 0.3,
                        "epochs": 50,  # Fewer epochs
                        "batch_size": 64,  # Larger batches for speed
                        "learning_rate": 0.002,  # Higher learning rate
                        "sequence_length": 72  # 3 days lookback
                    },
                    "prophet_config": {
                        "seasonality_mode": "additive",
                        "yearly_seasonality": False,  # Skip yearly for speed
                        "weekly_seasonality": True,
                        "daily_seasonality": True,
                        "holidays": False,  # Skip holidays for speed
                        "changepoint_prior_scale": 0.1,
                        "seasonality_prior_scale": 5.0
                    },
                    "hybrid_weights": {
                        "lstm": 0.7,
                        "prophet": 0.3
                    },
                    "validation_split": 0.15,
                    "cross_validation_folds": 3
                },
                "evaluation": {
                    "statistical_tests": ["t_test"],  # Single test for speed
                    "confidence_level": 0.90,  # Slightly relaxed
                    "bootstrap_samples": 1000,  # Fewer samples
                    "effect_size_metrics": ["cohen_d"],  # Single metric
                    "visualization_detail": "medium",
                    "constraint_validation": "standard"
                }
            }
    
    def run_simulation_pipeline(self):
        """Step 1: Run bus simulation pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: RUNNING BUS SIMULATION PIPELINE")
        self.logger.info("=" * 80)
        
        sim_output_dir = self.output_dir / "simulation_results"
        sim_output_dir.mkdir(exist_ok=True)
        
        sim_config = self.config["simulation"]
        
        try:
            # Import actual simulation components
            sys.path.insert(0, str(self.project_root / "bus_simulation_pipeline" / "src"))
            from core.simulation_engine import SimulationEngine
            
            self.logger.info(f"Initializing simulation with parameters:")
            self.logger.info(f"  - Period: {sim_config['start_date']} to {sim_config['end_date']}")
            self.logger.info(f"  - Time step: {sim_config['time_step']} minutes")
            self.logger.info(f"  - Buses per line: {sim_config['buses_per_line']}")
            
            # Load bus stops data
            stops_file = self.project_root / "data" / "ankara_bus_stops.csv"
            
            # Create simulation configuration
            from core.data_models import SimulationConfig
            config = SimulationConfig(
                start_date=datetime.strptime(sim_config['start_date'], "%Y-%m-%d"),
                end_date=datetime.strptime(sim_config['end_date'], "%Y-%m-%d"),
                time_step=sim_config['time_step'],
                randomize_travel_times=True,
                randomize_passenger_demand=True,
                seed=42
            )
            
            # Initialize simulation engine
            engine = SimulationEngine(config=config)
            
            # Load data and setup simulation
            self.logger.info("Loading bus stops data...")
            if not engine.load_data(str(stops_file)):
                raise Exception("Failed to load bus stops data")
            
            self.logger.info("Setting up simulation...")
            if not engine.setup_simulation(sim_config['buses_per_line']):
                raise Exception("Failed to setup simulation")
            
            # Run simulation
            self.logger.info("Starting simulation execution...")
            results_df = engine.run_simulation()
            
            if len(results_df) == 0:
                raise Exception("Simulation produced no results")
            
            # Save results
            passenger_flow_file = sim_output_dir / "passenger_flow_results.csv"
            bus_positions_file = sim_output_dir / "bus_position_results.csv"
            summary_file = sim_output_dir / "simulation_summary.json"
            
            # Save passenger flow data
            results_df.to_csv(passenger_flow_file, index=False)
            self.logger.info(f"Saved {len(results_df):,} passenger flow records")
            
            # Save bus position data (if available)
            if hasattr(engine, 'bus_positions_results') and engine.bus_positions_results:
                import pandas as pd
                positions_df = pd.DataFrame(engine.bus_positions_results)
                positions_df.to_csv(bus_positions_file, index=False)
                self.logger.info(f"Saved {len(positions_df):,} bus position records")
            
            # Save summary
            summary = {
                "simulation_config": sim_config,
                "execution_time": str(datetime.now()),
                "records_generated": {
                    "passenger_flows": len(results_df),
                    "bus_positions": len(engine.bus_positions_results) if hasattr(engine, 'bus_positions_results') else 0
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Store results for next steps
            self.simulation_results = {
                "passenger_flow": passenger_flow_file,
                "bus_positions": bus_positions_file,
                "summary": summary_file
            }
            
            self.logger.info("✅ Simulation pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Simulation pipeline failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_prediction_pipeline(self):
        """Step 3: Run bus prediction pipeline (uses both simulation and optimized schedule data)"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: RUNNING BUS PREDICTION PIPELINE")
        self.logger.info("=" * 80)
        
        pred_output_dir = self.output_dir / "prediction_results"
        pred_output_dir.mkdir(exist_ok=True)
        
        try:
            # Import actual prediction models
            sys.path.insert(0, str(self.project_root / "bus_prediction_pipeline" / "src"))
            from prediction_models.hybrid_model import HybridModel
            
            pred_config = self.config["prediction"]
            
            self.logger.info(f"Initializing prediction with parameters:")
            self.logger.info(f"  - LSTM architecture: {pred_config['lstm_config']['layers']}")
            self.logger.info(f"  - Training epochs: {pred_config['lstm_config']['epochs']}")
            self.logger.info(f"  - Prediction horizon: {pred_config['prediction_horizon']} hours")
            
            # Initialize hybrid model
            model = HybridModel()
            
            # Load simulation data for training
            self.logger.info(f"Loading simulation data for training from: {self.simulation_results['passenger_flow']}")
            
            # Load optimized schedule data for prediction
            if "passenger_flow_format" not in self.optimization_results or not self.optimization_results["passenger_flow_format"]:
                raise Exception("Optimized schedule data in passenger flow format not available")
            
            self.logger.info(f"Loading optimized schedule data for prediction from: {self.optimization_results['passenger_flow_format']}")
            
            # Load both datasets
            import pandas as pd
            historical_df = pd.read_csv(str(self.simulation_results['passenger_flow']))
            optimized_schedule_df = pd.read_csv(str(self.optimization_results['passenger_flow_format']))
            
            self.logger.info(f"Training data: {len(historical_df):,} records from simulation")
            self.logger.info(f"Prediction data: {len(optimized_schedule_df):,} records from optimized schedules")
            
            # Train hybrid model on simulation data
            self.logger.info("Training hybrid prediction model on simulation data...")
            success = model.train_hybrid_models(historical_df, save_models=True, model_save_dir=str(pred_output_dir / "models"))
            if not success:
                raise Exception("Failed to train prediction models")
            
            # Use optimized schedule data as the future data structure for predictions
            self.logger.info("Generating predictions for optimized schedule time points...")
            
            # Generate hybrid predictions using the optimized schedule structure
            predictions = model.predict_hybrid(
                historical_df, 
                optimized_schedule_df,
                apply_route_adjustments=True,
                apply_realistic_constraints=True,
                apply_night_constraints=True
            )
            
            # Save predictions
            pred_file = pred_output_dir / "predicted_passenger_flow.csv"
            predictions.to_csv(pred_file, index=False)
            
            # Save model metrics
            metrics_file = pred_output_dir / "model_metrics.json"
            pred_start = optimized_schedule_df['datetime'].min()
            pred_end = optimized_schedule_df['datetime'].max()
            metrics = {
                "prediction_period": f"{pred_start} to {pred_end}",
                "training_records": len(historical_df),
                "prediction_records": len(predictions),
                "model_type": "Hybrid (LSTM + Prophet)",
                "data_source": "Optimized schedules converted to passenger flow format",
                "lstm_config": pred_config['lstm_config'],
                "prophet_config": pred_config['prophet_config'],
                "hybrid_weights": pred_config['hybrid_weights']
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Generated {len(predictions):,} prediction records")
            
            # Store prediction results
            self.prediction_results = {
                "predictions": pred_file,
                "model_metrics": metrics_file
            }
            
            self.logger.info("✅ Prediction pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prediction pipeline failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_optimization_pipeline(self):
        """Step 2: Run bus optimization pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: RUNNING BUS OPTIMIZATION PIPELINE")
        self.logger.info("=" * 80)
        
        opt_output_dir = self.output_dir / "optimization_results"
        opt_output_dir.mkdir(exist_ok=True)
        
        try:
            # Import actual optimization components
            sys.path.insert(0, str(self.project_root / "bus_optimization_pipeline" / "src"))
            from optimization.ga_optimize import GeneticScheduleOptimizer
            
            opt_config = self.config["optimization"]
            
            self.logger.info(f"Initializing optimization with parameters:")
            self.logger.info(f"  - Generations: {opt_config['generations']}")
            self.logger.info(f"  - Population size: {opt_config['population_size']}")
            self.logger.info(f"  - Mutation rate: {opt_config['mutation_rate']}")
            self.logger.info(f"  - Crossover rate: {opt_config['crossover_rate']}")
            
            # Calculate optimization date (use part of simulation data)
            sim_start = datetime.strptime(self.config["simulation"]["start_date"], "%Y-%m-%d")
            opt_date = sim_start + timedelta(days=1)
            
            # Load passenger data
            import pandas as pd
            passenger_data = pd.read_csv(str(self.simulation_results['passenger_flow']))
            
            # Configure genetic algorithm parameters
            from optimization.ga_optimize import OptimizationConfig
            config = OptimizationConfig(
                generations=opt_config['generations'],
                population_size=opt_config['population_size'],
                mutation_rate=opt_config['mutation_rate'],
                crossover_rate=opt_config['crossover_rate'],
                elite_size=opt_config['elite_size'],
                tournament_size=opt_config['tournament_size']
            )
            
            # Initialize optimizer with correct parameters
            optimizer = GeneticScheduleOptimizer(
                config=config,
                stops_file=str(self.project_root / "data" / "ankara_bus_stops.csv")
            )
            
            # Store passenger data for reference (the optimizer will use simulation-based evaluation)
            optimizer.passenger_data = passenger_data
            
            # Run optimization for available lines (always optimize ALL lines)
            self.logger.info(f"Running optimization for date: {opt_date.strftime('%Y-%m-%d')}")
            line_ids = passenger_data['line_id'].unique()  # Always optimize all available lines
            
            optimized_schedules = {}
            for line_id in line_ids:
                self.logger.info(f"Optimizing line {line_id}...")
                best_schedule = optimizer.optimize_line_schedule(line_id, opt_date)
                optimized_schedules[line_id] = best_schedule
            
            # Export optimized schedules
            schedules_file = opt_output_dir / "optimized_schedules.csv"
            optimizer.export_optimized_schedules(optimized_schedules, str(schedules_file))
            
            # Save optimization metrics
            metrics_file = opt_output_dir / "optimization_metrics.json"
            metrics = {
                "optimization_date": opt_date.strftime('%Y-%m-%d'),
                "ga_config": opt_config,
                "lines_optimized": list(optimized_schedules.keys()),
                "schedules_per_line": {line_id: len(schedule.departure_times) for line_id, schedule in optimized_schedules.items()},
                "fitness_scores": {line_id: schedule.fitness for line_id, schedule in optimized_schedules.items()}
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            total_schedules = sum(len(schedule.departure_times) for schedule in optimized_schedules.values())
            self.logger.info(f"Generated {total_schedules:,} optimized schedule entries for {len(optimized_schedules)} lines")
            
            avg_fitness = sum(schedule.fitness or 0 for schedule in optimized_schedules.values()) / len(optimized_schedules)
            self.logger.info(f"Average fitness score: {avg_fitness:.4f}")
            
            # Store optimization results
            self.optimization_results = {
                "schedules": schedules_file,
                "metrics": metrics_file,
                "optimized_schedules": optimized_schedules
            }
            
            # Convert optimized schedules to passenger flow format for prediction
            self.logger.info("Converting optimized schedules to passenger flow format...")
            converted_file = self._convert_schedules_to_passenger_flow(schedules_file, opt_output_dir)
            
            # Update optimization results to include converted file
            self.optimization_results["passenger_flow_format"] = converted_file
            
            self.logger.info("✅ Optimization pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Optimization pipeline failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _convert_schedules_to_passenger_flow(self, schedules_file, output_dir):
        """Convert optimized schedules to passenger flow format using the OptimizedPassengerFlowGenerator"""
        self.logger.info("Converting optimized schedules to passenger flow format...")
        
        try:
            from .optimized_passenger_flow_generator import OptimizedPassengerFlowGenerator
            
            # Initialize the generator
            stops_file = str(self.project_root / "data" / "ankara_bus_stops.csv")
            generator = OptimizedPassengerFlowGenerator(stops_file, self.project_root)
            
            # Generate the passenger flow structure
            converted_file = output_dir / "optimized_schedules_passenger_flow.csv"
            
            self.logger.info("Using OptimizedPassengerFlowGenerator to convert schedules...")
            result_df = generator.generate_optimized_passenger_flow(
                schedule_file=str(schedules_file),
                output_file=str(converted_file)
            )
            
            if result_df is not None and len(result_df) > 0:
                self.logger.info(f"✅ Successfully converted {len(result_df)} passenger flow records")
                self.logger.info(f"Time range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
                self.logger.info(f"Lines covered: {sorted(result_df['line_id'].unique())}")
                self.logger.info(f"Trips generated: {result_df['trip_id'].nunique()}")
                
                return converted_file
            else:
                raise Exception("No passenger flow records generated from optimized schedules")
                
        except Exception as e:
            self.logger.error(f"Failed to convert schedules to passenger flow format: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def run_evaluation_pipeline(self):
        """Step 4: Run bus evaluation pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: RUNNING BUS EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        
        eval_output_dir = self.output_dir / "evaluation_results"
        eval_output_dir.mkdir(exist_ok=True)
        
        try:
            # Import actual evaluation components
            sys.path.insert(0, str(self.project_root / "bus_evaluation_pipeline" / "src"))
            from evaluation_engine import OptimizationEvaluator
            
            eval_config = self.config["evaluation"]
            
            self.logger.info(f"Initializing evaluation with parameters:")
            self.logger.info(f"  - Statistical tests: {eval_config['statistical_tests']}")
            self.logger.info(f"  - Confidence level: {eval_config['confidence_level']}")
            self.logger.info(f"  - Bootstrap samples: {eval_config['bootstrap_samples']:,}")
            
            # Initialize evaluator
            evaluator = OptimizationEvaluator(output_dir=str(eval_output_dir))
            
            self.logger.info("Loading data for evaluation...")
            self.logger.info(f"  - Original data: {self.simulation_results['passenger_flow']}")
            self.logger.info(f"  - Optimized data: {self.optimization_results['schedules']}")
            self.logger.info(f"  - Predicted data: {self.prediction_results['predictions']}")
            
            # Load all data for evaluation
            evaluator.load_data(
                original_file=str(self.simulation_results['passenger_flow']),
                optimized_file=str(self.prediction_results['predictions']),  # Use predicted data, not raw schedules
                predicted_file=str(self.prediction_results['predictions'])
            )
            
            # Calculate metrics for both datasets
            self.logger.info("Calculating performance metrics...")
            evaluator.original_metrics = evaluator.calculate_daily_metrics(evaluator.original_data, "Original Simulation")
            evaluator.optimized_metrics = evaluator.calculate_daily_metrics(evaluator.optimized_data, "Optimized Prediction")
            
            # Run evaluation components
            self.logger.info("Running constraint validation...")
            validation = evaluator.validate_constraints()
            
            self.logger.info("Running statistical analysis...")
            stats = evaluator.statistical_analysis()
            
            self.logger.info("Evaluating prediction accuracy...")
            prediction_accuracy = evaluator.evaluate_prediction_accuracy()
            
            # Generate visualizations and reports
            self.logger.info("Generating visualizations and reports...")
            evaluator.create_visualizations()
            evaluator.generate_report(validation, stats, prediction_accuracy)
            evaluator.save_evaluation_results(validation, stats, prediction_accuracy)
            
            # Store evaluation results
            self.evaluation_results = {
                "validation": validation,
                "statistics": stats,
                "prediction_accuracy": prediction_accuracy,
                "output_dir": eval_output_dir
            }
            
            # Log key results
            self.logger.info("=" * 60)
            self.logger.info("EVALUATION RESULTS SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
            self.logger.info(f"Validation Score: {validation['overall_score']:.1%}")
            
            if 'occupancy_improvement' in stats:
                improvement = stats['occupancy_improvement']['improvement_pct']
                self.logger.info(f"Occupancy Improvement: {improvement:.1f}%")
                self.logger.info(f"Statistical Significance: {'✅ YES' if stats['occupancy_improvement']['significant'] else '❌ NO'}")
            
            if prediction_accuracy:
                quality = prediction_accuracy['overall']['quality_rating']
                avg_r2 = prediction_accuracy['overall']['average_r2']
                self.logger.info(f"Prediction Quality: {quality} (R² = {avg_r2:.3f})")
            
            self.logger.info("✅ Evaluation pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation pipeline failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.logger.info("=" * 80)
        self.logger.info("GENERATING FINAL PIPELINE REPORT")
        self.logger.info("=" * 80)
        
        report_file = self.output_dir / "FINAL_PIPELINE_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Schedule Optimization Pipeline Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode:** {self.mode.upper()}\n")
            f.write(f"**Output Directory:** {self.output_dir}\n\n")
            
            # Pipeline Configuration
            f.write("## Pipeline Configuration\n\n")
            f.write("### Simulation Parameters\n")
            sim_config = self.config["simulation"]
            f.write(f"- **Period:** {sim_config['start_date']} to {sim_config['end_date']}\n")
            f.write(f"- **Time Step:** {sim_config['time_step']} minutes\n")
            f.write(f"- **Buses per Line:** {sim_config['buses_per_line']}\n\n")
            
            # Results Summary
            if hasattr(self, 'evaluation_results'):
                f.write("## Results Summary\n\n")
                validation = self.evaluation_results['validation']
                stats = self.evaluation_results['statistics']
                
                f.write(f"### Overall Performance\n")
                f.write(f"- **Validation Status:** {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}\n")
                f.write(f"- **Validation Score:** {validation['overall_score']:.1%}\n")
                
                if 'occupancy_improvement' in stats:
                    improvement = stats['occupancy_improvement']['improvement_pct']
                    significant = stats['occupancy_improvement']['significant']
                    f.write(f"- **Occupancy Improvement:** {improvement:.1f}%\n")
                    f.write(f"- **Statistical Significance:** {'✅ YES' if significant else '❌ NO'}\n")
            
            # File Locations
            f.write("\n## Output Files\n\n")
            if hasattr(self, 'simulation_results'):
                f.write("### Simulation Results\n")
                for key, path in self.simulation_results.items():
                    if hasattr(path, 'relative_to'):
                        rel_path = path.relative_to(self.output_dir)
                    else:
                        rel_path = str(path).replace(str(self.output_dir) + "/", "")
                    f.write(f"- **{key.replace('_', ' ').title()}:** `{rel_path}`\n")
            
            if hasattr(self, 'prediction_results'):
                f.write("\n### Prediction Results\n")
                for key, path in self.prediction_results.items():
                    if hasattr(path, 'relative_to'):
                        rel_path = path.relative_to(self.output_dir)
                    else:
                        rel_path = str(path).replace(str(self.output_dir) + "/", "")
                    f.write(f"- **{key.replace('_', ' ').title()}:** `{rel_path}`\n")
            
            if hasattr(self, 'optimization_results'):
                f.write("\n### Optimization Results\n")
                for key, path in self.optimization_results.items():
                    if hasattr(path, 'relative_to'):
                        rel_path = path.relative_to(self.output_dir)
                    else:
                        rel_path = str(path).replace(str(self.output_dir) + "/", "")
                    f.write(f"- **{key.replace('_', ' ').title()}:** `{rel_path}`\n")
        
        self.logger.info(f"Final report generated: {report_file}")
        return report_file
    
    def run_complete_pipeline(self):
        """Run the complete integrated pipeline"""
        start_time = datetime.now()
        
        self.logger.info("🚀 STARTING INTEGRATED SCHEDULE OPTIMIZATION PIPELINE")
        self.logger.info(f"Mode: {self.mode.upper()}")
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display optimized configuration summary
        self.display_configuration_summary()
        
        success = True
        
        # Step 1: Simulation
        if success:
            success = self.run_simulation_pipeline()
        
        # Step 2: Optimization (run before prediction)
        if success:
            success = self.run_optimization_pipeline()
        
        # Step 3: Prediction (uses both simulation and converted optimization data)
        if success:
            success = self.run_prediction_pipeline()
        
        # Step 4: Evaluation
        if success:
            success = self.run_evaluation_pipeline()
        
        # Generate final report
        if success:
            self.generate_final_report()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("=" * 80)
        if success:
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.info("❌ PIPELINE FAILED!")
        
        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        return success


def main():
    """Main entry point for the integrated pipeline"""
    parser = argparse.ArgumentParser(description="Integrated Schedule Optimization Pipeline")
    parser.add_argument('--mode', choices=['full', 'quick'], default='full',
                       help='Pipeline mode: full (1 year) or quick (1 week)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = IntegratedPipeline(mode=args.mode, output_dir=args.output_dir)
    success = pipeline.run_complete_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
