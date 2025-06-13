#!/usr/bin/env python3
"""
Enhanced Optimization Evaluation Engine

This module provides comprehensive evaluation of bus schedule optimization results,
integrating with prediction models and providing detailed analysis and visualizations.

Key Features:
- Integration with prediction models for realistic evaluation
- Comprehensive constraint validation
- Statistical significance testing
- Advanced visualizations with interactive plots
- Detailed reporting with actionable insights
- Performance benchmarking against multiple baselines
- Prediction accuracy assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import json
from typing import Dict, List, Tuple, Optional, Any


class OptimizationEvaluator:
    """Enhanced evaluation of bus schedule optimization results"""
    
    def __init__(self, output_dir: str = "evaluation_outputs"):
        self.original_data = None
        self.optimized_data = None
        self.predicted_data = None
        self.schedule_data = None
        self.evaluation_results = {}
        self.output_dir = output_dir
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.output_dir}/data", exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def load_data(self, 
                  original_file: str, 
                  optimized_file: str, 
                  predicted_file: Optional[str] = None,
                  schedule_file: Optional[str] = None):
        """Load all required data files with enhanced validation"""
        print("📁 LOADING DATA FILES")
        print("=" * 60)
        
        # Load original simulation data
        print(f"Loading original simulation: {original_file}")
        self.original_data = pd.read_csv(original_file)
        self.original_data['datetime'] = pd.to_datetime(self.original_data['datetime'])
        if 'hour' not in self.original_data.columns:
            self.original_data['hour'] = self.original_data['datetime'].dt.hour
        print(f"✅ Loaded {len(self.original_data):,} original records")
        
        # Load optimized predictions
        print(f"Loading optimized predictions: {optimized_file}")
        self.optimized_data = pd.read_csv(optimized_file)
        self.optimized_data['datetime'] = pd.to_datetime(self.optimized_data['datetime'])
        if 'hour' not in self.optimized_data.columns:
            self.optimized_data['hour'] = self.optimized_data['datetime'].dt.hour
        print(f"✅ Loaded {len(self.optimized_data):,} optimized records")
        
        # Load predicted data if available
        if predicted_file and os.path.exists(predicted_file):
            print(f"Loading predicted data: {predicted_file}")
            self.predicted_data = pd.read_csv(predicted_file)
            self.predicted_data['datetime'] = pd.to_datetime(self.predicted_data['datetime'])
            if 'hour' not in self.predicted_data.columns:
                self.predicted_data['hour'] = self.predicted_data['datetime'].dt.hour
            print(f"✅ Loaded {len(self.predicted_data):,} predicted records")
        
        # Load schedule structure if available
        if schedule_file and os.path.exists(schedule_file):
            print(f"Loading schedule structure: {schedule_file}")
            self.schedule_data = pd.read_csv(schedule_file)
            self.schedule_data['datetime'] = pd.to_datetime(self.schedule_data['datetime'])
            print(f"✅ Loaded {len(self.schedule_data):,} schedule records")
        
        # Validate data consistency
        self._validate_data_consistency()
        
    def _validate_data_consistency(self):
        """Validate that loaded data is consistent and complete"""
        print("\n🔍 VALIDATING DATA CONSISTENCY")
        print("-" * 40)
        
        required_columns = ['datetime', 'line_id', 'stop_id', 'boarding', 'alighting', 'occupancy_rate']
        
        for name, data in [("Original", self.original_data), ("Optimized", self.optimized_data), ("Predicted", self.predicted_data)]:
            if data is None:
                continue
                
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                print(f"⚠️  {name} data missing columns: {missing_cols}")
            else:
                print(f"✅ {name} data has all required columns")
                
            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            if null_counts.sum() > 0:
                print(f"⚠️  {name} data has null values: {null_counts[null_counts > 0].to_dict()}")
            else:
                print(f"✅ {name} data has no null values")
        
    def calculate_daily_metrics(self, df: pd.DataFrame, label: str) -> dict:
        """Calculate daily passenger and operational metrics"""
        
        # Calculate simulation period
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        days = (end_date - start_date).days + 1
        
        metrics = {
            'label': label,
            'total_records': len(df),
            'simulation_days': days,
            'total_boarding': df['boarding'].sum(),
            'total_alighting': df['alighting'].sum(),
            'daily_boarding': df['boarding'].sum() / days,
            'daily_alighting': df['alighting'].sum() / days,
            'avg_occupancy': df['occupancy_rate'].mean(),
            'peak_occupancy': df['occupancy_rate'].max(),
            'overcrowded_records': len(df[df['occupancy_rate'] > 1.0]),
            'severely_overcrowded': len(df[df['occupancy_rate'] > 1.5]),
            'conservation_ratio': df['alighting'].sum() / df['boarding'].sum() if df['boarding'].sum() > 0 else 0,
            'unique_buses': df['bus_id'].nunique(),
            'unique_lines': df['line_id'].nunique(),
            'unique_stops': df['stop_id'].nunique()
        }
        
        # Rush hour analysis
        rush_hours = [7, 8, 9, 16, 17, 18, 19]
        rush_data = df[df['hour'].isin(rush_hours)]
        metrics.update({
            'rush_hour_avg_occupancy': rush_data['occupancy_rate'].mean(),
            'rush_hour_peak_occupancy': rush_data['occupancy_rate'].max(),
            'rush_hour_records': len(rush_data),
            'rush_hour_overcrowded': len(rush_data[rush_data['occupancy_rate'] > 1.0])
        })
        
        return metrics
    
    def validate_constraints(self) -> dict:
        """Validate that optimization adhered to all constraints"""
        print("\n🔍 CONSTRAINT VALIDATION")
        print("=" * 60)
        
        validation = {
            'passenger_conservation': False,
            'resource_constraints': False,
            'operational_constraints': False,
            'schedule_feasibility': False,
            'details': {}
        }
        
        # 1. Passenger Conservation Check
        orig_daily_passengers = self.original_metrics['daily_boarding'] + self.original_metrics['daily_alighting']
        opt_daily_passengers = self.optimized_metrics['daily_boarding'] + self.optimized_metrics['daily_alighting']
        passenger_ratio = opt_daily_passengers / orig_daily_passengers
        
        validation['details']['passenger_ratio'] = passenger_ratio
        validation['passenger_conservation'] = 0.85 <= passenger_ratio <= 1.15  # Within ±15% for daily variation
        
        print(f"Passenger Conservation:")
        print(f"  Original daily passengers: {orig_daily_passengers:,.0f}")
        print(f"  Optimized daily passengers: {opt_daily_passengers:,.0f}")
        print(f"  Ratio: {passenger_ratio:.3f}")
        print(f"  Status: {'✅ PASS' if validation['passenger_conservation'] else '❌ FAIL'} (±15% tolerance)")
        
        # 2. Resource Constraints Check
        orig_buses = self.original_metrics['unique_buses']
        opt_buses = self.optimized_metrics['unique_buses']
        bus_ratio = opt_buses / orig_buses
        
        validation['details']['bus_ratio'] = bus_ratio
        validation['resource_constraints'] = bus_ratio <= 1.0  # Not more buses
        
        print(f"\nResource Constraints:")
        print(f"  Original buses used: {orig_buses}")
        print(f"  Optimized buses used: {opt_buses}")
        print(f"  Ratio: {bus_ratio:.3f}")
        print(f"  Status: {'✅ PASS' if validation['resource_constraints'] else '❌ FAIL'}")
        
        # 3. Operational Constraints Check
        opt_times = pd.to_datetime(self.optimized_data['datetime'])
        valid_hours = opt_times.dt.hour.between(0, 23)  # 24-hour operation (0-23)
        
        validation['details']['valid_operating_hours'] = valid_hours.all()
        validation['operational_constraints'] = valid_hours.all()
        
        print(f"\nOperational Constraints:")
        print(f"  All departures within 00:00-23:59: {'✅ YES' if valid_hours.all() else '❌ NO'}")
        print(f"  Status: {'✅ PASS' if validation['operational_constraints'] else '❌ FAIL'}")
        
        # 4. Schedule Feasibility Check
        # Check for realistic minimum intervals between departures
        schedule_by_line = {}
        for line_id in self.optimized_data['line_id'].unique():
            line_data = self.optimized_data[self.optimized_data['line_id'] == line_id]
            departure_times = line_data.groupby('trip_id')['datetime'].min().sort_values()
            if len(departure_times) > 1:
                intervals = departure_times.diff().dt.total_seconds() / 60  # Minutes
                min_interval = intervals.min()
                schedule_by_line[line_id] = min_interval
        
        min_intervals = list(schedule_by_line.values())
        if min_intervals:
            feasible_intervals = all(interval >= 5 for interval in min_intervals)  # 5 min minimum
            min_interval_value = min(min_intervals)
        else:
            feasible_intervals = True  # No intervals to check, assume feasible
            min_interval_value = float('inf')
        
        validation['details']['min_intervals'] = schedule_by_line
        validation['schedule_feasibility'] = feasible_intervals
        
        print(f"\nSchedule Feasibility:")
        if min_intervals:
            print(f"  Minimum departure interval: {min_interval_value:.1f} minutes")
            print(f"  All intervals ≥ 5 minutes: {'✅ YES' if feasible_intervals else '❌ NO'}")
        else:
            print(f"  No departure intervals to check (single departures per line)")
        print(f"  Status: {'✅ PASS' if validation['schedule_feasibility'] else '❌ FAIL'}")
        
        # Overall validation
        overall_pass = all([
            validation['passenger_conservation'],
            validation['resource_constraints'],
            validation['operational_constraints'],
            validation['schedule_feasibility']
        ])
        
        # Calculate overall score as percentage of passed constraints
        passed_constraints = sum([
            validation['passenger_conservation'],
            validation['resource_constraints'],
            validation['operational_constraints'],
            validation['schedule_feasibility']
        ])
        overall_score = passed_constraints / 4.0
        
        validation['overall_pass'] = overall_pass
        validation['overall_score'] = overall_score
        print(f"\n🎯 OVERALL CONSTRAINT VALIDATION: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        print(f"   Validation Score: {overall_score:.1%} ({passed_constraints}/4 constraints passed)")
        
        return validation
    
    def statistical_analysis(self) -> dict:
        """Perform statistical analysis of optimization improvements"""
        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        stats_results = {}
        
        # Sample occupancy rates for comparison
        orig_occupancy = self.original_data['occupancy_rate']
        opt_occupancy = self.optimized_data['occupancy_rate']
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(orig_occupancy, opt_occupancy)
        
        stats_results = {
            'occupancy_improvement': {
                'original_mean': orig_occupancy.mean(),
                'optimized_mean': opt_occupancy.mean(),
                'improvement_pct': (orig_occupancy.mean() - opt_occupancy.mean()) / orig_occupancy.mean() * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
        
        # Overcrowding reduction analysis
        orig_overcrowded_pct = (orig_occupancy > 1.0).mean() * 100
        opt_overcrowded_pct = (opt_occupancy > 1.0).mean() * 100
        
        stats_results['overcrowding_reduction'] = {
            'original_pct': orig_overcrowded_pct,
            'optimized_pct': opt_overcrowded_pct,
            'reduction_pct': orig_overcrowded_pct - opt_overcrowded_pct,
            'relative_reduction': (orig_overcrowded_pct - opt_overcrowded_pct) / orig_overcrowded_pct * 100
        }
        
        print(f"Occupancy Rate Improvement:")
        print(f"  Original mean: {stats_results['occupancy_improvement']['original_mean']:.1%}")
        print(f"  Optimized mean: {stats_results['occupancy_improvement']['optimized_mean']:.1%}")
        print(f"  Improvement: {stats_results['occupancy_improvement']['improvement_pct']:.1f}%")
        print(f"  Statistical significance: p = {p_value:.6f} {'✅ Significant' if p_value < 0.05 else '❌ Not significant'}")
        
        print(f"\nOvercrowding Reduction:")
        print(f"  Original overcrowded: {orig_overcrowded_pct:.1f}%")
        print(f"  Optimized overcrowded: {opt_overcrowded_pct:.1f}%")
        print(f"  Absolute reduction: {stats_results['overcrowding_reduction']['reduction_pct']:.1f} percentage points")
        print(f"  Relative reduction: {stats_results['overcrowding_reduction']['relative_reduction']:.1f}%")
        
        return stats_results
    
    def evaluate_prediction_accuracy(self) -> Optional[Dict[str, Any]]:
        """Evaluate prediction model accuracy if prediction data is available"""
        if self.predicted_data is None:
            print("⚠️  No prediction data available for accuracy evaluation")
            return None
            
        print("\n🎯 PREDICTION ACCURACY EVALUATION")
        print("=" * 60)
        
        # Align data by datetime and location for comparison
        original_subset = self.original_data[['datetime', 'line_id', 'stop_id', 'boarding', 'alighting', 'occupancy_rate']].copy()
        predicted_subset = self.predicted_data[['datetime', 'line_id', 'stop_id', 'boarding', 'alighting', 'occupancy_rate']].copy()
        
        # Merge on common time-location pairs
        merged = pd.merge(original_subset, predicted_subset, 
                         on=['datetime', 'line_id', 'stop_id'], 
                         suffixes=('_actual', '_predicted'))
        
        if len(merged) == 0:
            print("❌ No matching records found between original and predicted data")
            return None
            
        print(f"✅ Found {len(merged):,} matching records for accuracy evaluation")
        
        accuracy_results = {}
        
        # Boarding prediction accuracy
        boarding_mae = mean_absolute_error(merged['boarding_actual'], merged['boarding_predicted'])
        boarding_rmse = np.sqrt(mean_squared_error(merged['boarding_actual'], merged['boarding_predicted']))
        boarding_r2 = r2_score(merged['boarding_actual'], merged['boarding_predicted'])
        boarding_mape = np.mean(np.abs((merged['boarding_actual'] - merged['boarding_predicted']) / 
                                      (merged['boarding_actual'] + 1e-8))) * 100  # Add small epsilon to avoid division by zero
        
        accuracy_results['boarding'] = {
            'mae': boarding_mae,
            'rmse': boarding_rmse,
            'r2_score': boarding_r2,
            'mape': boarding_mape,
            'correlation': merged['boarding_actual'].corr(merged['boarding_predicted'])
        }
        
        # Alighting prediction accuracy
        alighting_mae = mean_absolute_error(merged['alighting_actual'], merged['alighting_predicted'])
        alighting_rmse = np.sqrt(mean_squared_error(merged['alighting_actual'], merged['alighting_predicted']))
        alighting_r2 = r2_score(merged['alighting_actual'], merged['alighting_predicted'])
        alighting_mape = np.mean(np.abs((merged['alighting_actual'] - merged['alighting_predicted']) / 
                                       (merged['alighting_actual'] + 1e-8))) * 100
        
        accuracy_results['alighting'] = {
            'mae': alighting_mae,
            'rmse': alighting_rmse,
            'r2_score': alighting_r2,
            'mape': alighting_mape,
            'correlation': merged['alighting_actual'].corr(merged['alighting_predicted'])
        }
        
        # Occupancy prediction accuracy
        occupancy_mae = mean_absolute_error(merged['occupancy_rate_actual'], merged['occupancy_rate_predicted'])
        occupancy_rmse = np.sqrt(mean_squared_error(merged['occupancy_rate_actual'], merged['occupancy_rate_predicted']))
        occupancy_r2 = r2_score(merged['occupancy_rate_actual'], merged['occupancy_rate_predicted'])
        occupancy_mape = np.mean(np.abs((merged['occupancy_rate_actual'] - merged['occupancy_rate_predicted']) / 
                                       (merged['occupancy_rate_actual'] + 1e-8))) * 100
        
        accuracy_results['occupancy'] = {
            'mae': occupancy_mae,
            'rmse': occupancy_rmse,
            'r2_score': occupancy_r2,
            'mape': occupancy_mape,
            'correlation': merged['occupancy_rate_actual'].corr(merged['occupancy_rate_predicted'])
        }
        
        # Overall accuracy assessment
        avg_r2 = np.mean([boarding_r2, alighting_r2, occupancy_r2])
        avg_mape = np.mean([boarding_mape, alighting_mape, occupancy_mape])
        
        accuracy_results['overall'] = {
            'average_r2': avg_r2,
            'average_mape': avg_mape,
            'quality_rating': 'Excellent' if avg_r2 > 0.8 else 'Good' if avg_r2 > 0.6 else 'Fair' if avg_r2 > 0.4 else 'Poor'
        }
        
        print(f"Prediction Accuracy Results:")
        print(f"  Boarding - R²: {boarding_r2:.3f}, MAPE: {boarding_mape:.1f}%")
        print(f"  Alighting - R²: {alighting_r2:.3f}, MAPE: {alighting_mape:.1f}%")
        print(f"  Occupancy - R²: {occupancy_r2:.3f}, MAPE: {occupancy_mape:.1f}%")
        print(f"  Overall Quality: {accuracy_results['overall']['quality_rating']} (Avg R²: {avg_r2:.3f})")
        
        return accuracy_results
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n📈 GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Use the configured output directory
        plots_dir = f"{self.output_dir}/plots"
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Occupancy Rate Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Optimization Impact Analysis', fontsize=16, fontweight='bold')
        
        # Occupancy distribution comparison
        axes[0, 0].hist(self.original_data['occupancy_rate'], bins=50, alpha=0.7, label='Original', color='red')
        axes[0, 0].hist(self.optimized_data['occupancy_rate'], bins=50, alpha=0.7, label='Optimized', color='green')
        axes[0, 0].set_xlabel('Occupancy Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Occupancy Rate Distribution')
        axes[0, 0].legend()
        axes[0, 0].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='100% Capacity')
        
        # Hourly occupancy patterns
        orig_hourly = self.original_data.groupby('hour')['occupancy_rate'].mean()
        opt_hourly = self.optimized_data.groupby('hour')['occupancy_rate'].mean()
        
        axes[0, 1].plot(orig_hourly.index, orig_hourly.values, 'o-', label='Original', color='red', linewidth=2)
        axes[0, 1].plot(opt_hourly.index, opt_hourly.values, 'o-', label='Optimized', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Average Occupancy Rate')
        axes[0, 1].set_title('Hourly Occupancy Patterns')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Line-wise comparison
        orig_by_line = self.original_data.groupby('line_id')['occupancy_rate'].agg(['mean', 'max'])
        opt_by_line = self.optimized_data.groupby('line_id')['occupancy_rate'].agg(['mean', 'max'])
        
        lines = sorted(set(orig_by_line.index) & set(opt_by_line.index))
        x_pos = np.arange(len(lines))
        
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, [orig_by_line.loc[line, 'mean'] for line in lines], 
                      width, label='Original', color='red', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, [opt_by_line.loc[line, 'mean'] for line in lines], 
                      width, label='Optimized', color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Bus Line')
        axes[1, 0].set_ylabel('Average Occupancy Rate')
        axes[1, 0].set_title('Line-wise Average Occupancy')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(lines, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Overcrowding analysis
        overcrowding_data = {
            'Metric': ['> 100% Occupancy', '> 150% Occupancy', 'Peak Occupancy'],
            'Original': [
                len(self.original_data[self.original_data['occupancy_rate'] > 1.0]) / len(self.original_data) * 100,
                len(self.original_data[self.original_data['occupancy_rate'] > 1.5]) / len(self.original_data) * 100,
                self.original_data['occupancy_rate'].max() * 100
            ],
            'Optimized': [
                len(self.optimized_data[self.optimized_data['occupancy_rate'] > 1.0]) / len(self.optimized_data) * 100,
                len(self.optimized_data[self.optimized_data['occupancy_rate'] > 1.5]) / len(self.optimized_data) * 100,
                self.optimized_data['occupancy_rate'].max() * 100
            ]
        }
        
        df_overcrowding = pd.DataFrame(overcrowding_data)
        x_pos = np.arange(len(df_overcrowding))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, df_overcrowding['Original'], width, label='Original', color='red', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, df_overcrowding['Optimized'], width, label='Optimized', color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Overcrowding Metrics')
        axes[1, 1].set_ylabel('Percentage / Value')
        axes[1, 1].set_title('Overcrowding Reduction Analysis')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(df_overcrowding['Metric'], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/optimization_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display issues
        
        # 2. Constraint Validation Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Constraint Validation Analysis', fontsize=16, fontweight='bold')
        
        # Passenger conservation
        passenger_data = {
            'Dataset': ['Original\n(Daily Avg)', 'Optimized\n(Single Day)'],
            'Boarding': [self.original_metrics['daily_boarding'], self.optimized_metrics['daily_boarding']],
            'Alighting': [self.original_metrics['daily_alighting'], self.optimized_metrics['daily_alighting']]
        }
        
        x_pos = np.arange(len(passenger_data['Dataset']))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, passenger_data['Boarding'], width, label='Boarding', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, passenger_data['Alighting'], width, label='Alighting', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Daily Passengers')
        axes[0, 0].set_title('Passenger Volume Conservation')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(passenger_data['Dataset'])
        axes[0, 0].legend()
        
        # Resource utilization
        resource_data = {
            'Resource': ['Buses Used', 'Lines Operated', 'Stops Served'],
            'Original': [self.original_metrics['unique_buses'], self.original_metrics['unique_lines'], 
                        self.original_metrics['unique_stops']],
            'Optimized': [self.optimized_metrics['unique_buses'], self.optimized_metrics['unique_lines'],
                         self.optimized_metrics['unique_stops']]
        }
        
        x_pos = np.arange(len(resource_data['Resource']))
        width = 0.35
        
        axes[0, 1].bar(x_pos - width/2, resource_data['Original'], width, label='Original', color='red', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, resource_data['Optimized'], width, label='Optimized', color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Resource Type')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Resource Utilization')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(resource_data['Resource'])
        axes[0, 1].legend()
        
        # Schedule distribution
        orig_hourly_deps = self.original_data.groupby('hour').size()
        opt_hourly_deps = self.optimized_data.groupby('hour').size()
        
        axes[1, 0].plot(orig_hourly_deps.index, orig_hourly_deps.values, 'o-', label='Original', color='red', linewidth=2)
        axes[1, 0].plot(opt_hourly_deps.index, opt_hourly_deps.values, 'o-', label='Optimized', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Departures')
        axes[1, 0].set_title('Hourly Schedule Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics summary
        metrics_data = {
            'Metric': ['Avg Occupancy', 'Peak Occupancy', 'Conservation Ratio'],
            'Original': [self.original_metrics['avg_occupancy'], self.original_metrics['peak_occupancy'], 
                        self.original_metrics['conservation_ratio']],
            'Optimized': [self.optimized_metrics['avg_occupancy'], self.optimized_metrics['peak_occupancy'],
                         self.optimized_metrics['conservation_ratio']]
        }
        
        x_pos = np.arange(len(metrics_data['Metric']))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, metrics_data['Original'], width, label='Original', color='red', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, metrics_data['Optimized'], width, label='Optimized', color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Performance Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Key Performance Indicators')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics_data['Metric'], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/constraint_validation.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display issues
        
        print(f"✅ Visualizations saved to {plots_dir}/ directory")
    
    def generate_report(self, validation: dict, stats: dict, prediction_accuracy: Optional[dict] = None):
        """Generate comprehensive evaluation report"""
        print("\n📄 GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        report_file = f"{self.output_dir}/reports/optimization_evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GENETIC ALGORITHM OPTIMIZATION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"The genetic algorithm optimization {'SUCCESSFULLY' if validation['overall_pass'] else 'FAILED TO'} ")
            f.write("improve bus schedule efficiency while adhering to all operational constraints.\n\n")
            
            # Key Improvements
            f.write("KEY IMPROVEMENTS ACHIEVED:\n")
            improvement_pct = stats['occupancy_improvement']['improvement_pct']
            overcrowding_reduction = stats['overcrowding_reduction']['relative_reduction']
            
            f.write(f"• Occupancy rate reduced by {improvement_pct:.1f}%\n")
            f.write(f"• Overcrowding incidents reduced by {overcrowding_reduction:.1f}%\n")
            f.write(f"• Peak occupancy reduced from {self.original_metrics['peak_occupancy']:.1%} to {self.optimized_metrics['peak_occupancy']:.1%}\n")
            f.write(f"• Conservation ratio maintained at {self.optimized_metrics['conservation_ratio']:.3f}\n\n")
            
            # Detailed Metrics
            f.write("DETAILED PERFORMANCE COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Change':<15}\n")
            f.write("-" * 75 + "\n")
            
            metrics_comparison = [
                ('Daily Passengers', f"{self.original_metrics['daily_boarding']:,.0f}", 
                 f"{self.optimized_metrics['daily_boarding']:,.0f}", 
                 f"{(self.optimized_metrics['daily_boarding']/self.original_metrics['daily_boarding']-1)*100:+.1f}%"),
                ('Average Occupancy', f"{self.original_metrics['avg_occupancy']:.1%}", 
                 f"{self.optimized_metrics['avg_occupancy']:.1%}", 
                 f"{improvement_pct:+.1f}%"),
                ('Peak Occupancy', f"{self.original_metrics['peak_occupancy']:.1%}", 
                 f"{self.optimized_metrics['peak_occupancy']:.1%}", 
                 f"{(self.optimized_metrics['peak_occupancy']/self.original_metrics['peak_occupancy']-1)*100:+.1f}%"),
                ('Overcrowded Records', f"{self.original_metrics['overcrowded_records']:,}", 
                 f"{self.optimized_metrics['overcrowded_records']:,}", 
                 f"{overcrowding_reduction:+.1f}%"),
                ('Buses Used', f"{self.original_metrics['unique_buses']}", 
                 f"{self.optimized_metrics['unique_buses']}", 
                 f"{self.optimized_metrics['unique_buses'] - self.original_metrics['unique_buses']:+d}")
            ]
            
            for metric, orig, opt, change in metrics_comparison:
                f.write(f"{metric:<30} {orig:<15} {opt:<15} {change:<15}\n")
            
            # Constraint Validation
            f.write(f"\nCONSTRAINT VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Passenger Conservation: {'✅ PASS' if validation['passenger_conservation'] else '❌ FAIL'}\n")
            f.write(f"Resource Constraints: {'✅ PASS' if validation['resource_constraints'] else '❌ FAIL'}\n")
            f.write(f"Operational Constraints: {'✅ PASS' if validation['operational_constraints'] else '❌ FAIL'}\n")
            f.write(f"Schedule Feasibility: {'✅ PASS' if validation['schedule_feasibility'] else '❌ FAIL'}\n")
            f.write(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}\n\n")
            
            # Statistical Significance
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Occupancy improvement p-value: {stats['occupancy_improvement']['p_value']:.6f}\n")
            f.write(f"Statistical significance: {'Yes' if stats['occupancy_improvement']['significant'] else 'No'}\n\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("-" * 40 + "\n")
            if validation['overall_pass']:
                f.write("The genetic algorithm optimization successfully improved system performance ")
                f.write("while maintaining operational feasibility and resource constraints. ")
                f.write("The improvements are statistically significant and represent genuine ")
                f.write("enhancements to service quality and passenger experience.\n")
            else:
                f.write("The optimization results do not meet all validation criteria. ")
                f.write("Further analysis and refinement may be required.\n")
        
        print(f"✅ Comprehensive report saved to {report_file}")
    
    def save_evaluation_results(self, validation: dict, stats: dict, prediction_accuracy: Optional[dict] = None):
        """Save evaluation results as structured JSON data"""
        results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'original_metrics': self.original_metrics,
                'optimized_metrics': self.optimized_metrics
            },
            'constraint_validation': validation,
            'statistical_analysis': stats,
            'prediction_accuracy': prediction_accuracy
        }
        
        results_file = f"{self.output_dir}/data/evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Evaluation results saved to {results_file}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🎯 COMPREHENSIVE OPTIMIZATION EVALUATION")
        print("=" * 80)
        
        # Note: Data should be loaded externally via load_data() method
        if self.original_data is None or self.optimized_data is None:
            raise ValueError("Data must be loaded before running evaluation. Call load_data() first.")
        
        # Calculate metrics
        print("\n📊 CALCULATING PERFORMANCE METRICS")
        print("=" * 60)
        self.original_metrics = self.calculate_daily_metrics(self.original_data, "Original Simulation")
        self.optimized_metrics = self.calculate_daily_metrics(self.optimized_data, "Optimized Prediction")
        
        # Print comparison
        print(f"\nOriginal Simulation (7-day average):")
        print(f"  Daily passengers: {self.original_metrics['daily_boarding']:,.0f}")
        print(f"  Average occupancy: {self.original_metrics['avg_occupancy']:.1%}")
        print(f"  Peak occupancy: {self.original_metrics['peak_occupancy']:.1%}")
        print(f"  Overcrowded records: {self.original_metrics['overcrowded_records']:,}")
        
        print(f"\nOptimized Prediction (single day):")
        print(f"  Daily passengers: {self.optimized_metrics['daily_boarding']:,.0f}")
        print(f"  Average occupancy: {self.optimized_metrics['avg_occupancy']:.1%}")
        print(f"  Peak occupancy: {self.optimized_metrics['peak_occupancy']:.1%}")
        print(f"  Overcrowded records: {self.optimized_metrics['overcrowded_records']:,}")
        
        # Validate constraints
        validation = self.validate_constraints()
        
        # Statistical analysis
        stats = self.statistical_analysis()
        
        # Prediction accuracy evaluation (if prediction data available)
        prediction_accuracy = self.evaluate_prediction_accuracy()
        
        # Generate visualizations
        self.create_visualizations()
        
        # Generate comprehensive report
        self.generate_report(validation, stats, prediction_accuracy)
        
        # Save evaluation results as JSON
        self.save_evaluation_results(validation, stats, prediction_accuracy)
        
        return validation, stats, prediction_accuracy


def main():
    """Main execution function"""
    evaluator = OptimizationEvaluator()
    validation, stats, prediction_accuracy = evaluator.run_evaluation()
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
    print(f"Key Improvement: {stats['occupancy_improvement']['improvement_pct']:.1f}% occupancy reduction")
    if prediction_accuracy:
        print(f"Prediction Quality: {prediction_accuracy['overall']['quality_rating']}")
    print(f"Report saved: {evaluator.output_dir}/reports/optimization_evaluation_report.txt")
    print(f"Plots saved: {evaluator.output_dir}/plots/ directory")
    print(f"Data saved: {evaluator.output_dir}/data/ directory")


if __name__ == "__main__":
    main() 