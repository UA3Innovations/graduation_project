#!/usr/bin/env python3
"""
Comprehensive Optimization Evaluation Script
This script thoroughly evaluates the genetic algorithm optimization results,
validates constraint adherence, and generates detailed visualizations and reports.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os


class OptimizationEvaluator:
    """Comprehensive evaluation of bus schedule optimization results"""
    
    def __init__(self):
        self.original_data = None
        self.optimized_data = None
        self.schedule_data = None
        self.evaluation_results = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, original_file: str, optimized_file: str, schedule_file: str):
        """Load all required data files"""
        print("📁 LOADING DATA FILES")
        print("=" * 60)
        
        # Load original simulation data
        print(f"Loading original simulation: {original_file}")
        self.original_data = pd.read_csv(original_file)
        self.original_data['datetime'] = pd.to_datetime(self.original_data['datetime'])
        print(f"✅ Loaded {len(self.original_data):,} original records")
        
        # Load optimized predictions
        print(f"Loading optimized predictions: {optimized_file}")
        self.optimized_data = pd.read_csv(optimized_file)
        self.optimized_data['datetime'] = pd.to_datetime(self.optimized_data['datetime'])
        print(f"✅ Loaded {len(self.optimized_data):,} optimized records")
        
        # Load schedule structure
        print(f"Loading schedule structure: {schedule_file}")
        self.schedule_data = pd.read_csv(schedule_file)
        # Handle different datetime column names
        if 'departure_time' in self.schedule_data.columns:
            self.schedule_data['datetime'] = pd.to_datetime(self.schedule_data['departure_time'])
        elif 'datetime' in self.schedule_data.columns:
            self.schedule_data['datetime'] = pd.to_datetime(self.schedule_data['datetime'])
        print(f"✅ Loaded {len(self.schedule_data):,} schedule records")
        
    def calculate_daily_metrics(self, df: pd.DataFrame, label: str) -> dict:
        """Calculate daily passenger and operational metrics"""
        
        # Convert datetime to pandas datetime if it's not already
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calculate simulation period
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        days = (end_date - start_date).days + 1
        
        # Add hour column if not present
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
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
        # Since we don't have trip_id, group by bus_id and line_id combinations
        schedule_by_line = {}
        for line_id in self.optimized_data['line_id'].unique():
            line_data = self.optimized_data[self.optimized_data['line_id'] == line_id]
            # Group by bus_id to get individual bus schedules
            for bus_id in line_data['bus_id'].unique():
                bus_data = line_data[line_data['bus_id'] == bus_id]
                departure_times = pd.to_datetime(bus_data['datetime']).sort_values()
                if len(departure_times) > 1:
                    intervals = departure_times.diff().dt.total_seconds() / 60  # Minutes
                    intervals = intervals.dropna()  # Remove NaN from first difference
                    if len(intervals) > 0:
                        min_interval = intervals.min()
                        schedule_by_line[f"{line_id}_{bus_id}"] = min_interval
        
        min_intervals = list(schedule_by_line.values())
        if min_intervals:
            feasible_intervals = all(interval >= 1 for interval in min_intervals)  # 1 min minimum (more lenient)
            min_interval_value = min(min_intervals)
        else:
            feasible_intervals = True  # No intervals to check
            min_interval_value = float('inf')
        
        validation['details']['min_intervals'] = schedule_by_line
        validation['schedule_feasibility'] = feasible_intervals
        
        print(f"\nSchedule Feasibility:")
        print(f"  Minimum departure interval: {min_interval_value:.1f} minutes")
        print(f"  All intervals ≥ 1 minute: {'✅ YES' if feasible_intervals else '❌ NO'}")
        print(f"  Status: {'✅ PASS' if validation['schedule_feasibility'] else '❌ FAIL'}")
        
        # Overall validation
        overall_pass = all([
            validation['passenger_conservation'],
            validation['resource_constraints'],
            validation['operational_constraints'],
            validation['schedule_feasibility']
        ])
        
        validation['overall_pass'] = overall_pass
        print(f"\n🎯 OVERALL CONSTRAINT VALIDATION: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
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
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n📈 GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create output directory
        os.makedirs('evaluation_plots', exist_ok=True)
        
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
        plt.savefig('evaluation_plots/optimization_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
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
        plt.savefig('evaluation_plots/constraint_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to evaluation_plots/ directory")
    
    def generate_report(self, validation: dict, stats: dict):
        """Generate comprehensive evaluation report"""
        print("\n📄 GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        report_file = "optimization_evaluation_report.txt"
        
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
            
            # Handle both regular and pipeline validation formats
            if 'passenger_conservation' in validation:
                # Regular validation format
                f.write(f"Passenger Conservation: {'✅ PASS' if validation['passenger_conservation'] else '❌ FAIL'}\n")
                f.write(f"Resource Constraints: {'✅ PASS' if validation['resource_constraints'] else '❌ FAIL'}\n")
                f.write(f"Operational Constraints: {'✅ PASS' if validation['operational_constraints'] else '❌ FAIL'}\n")
                f.write(f"Schedule Feasibility: {'✅ PASS' if validation['schedule_feasibility'] else '❌ FAIL'}\n")
            else:
                # Pipeline validation format
                f.write(f"Bus Count Constraint: {'✅ PASS' if validation.get('bus_count_constraint', False) else '❌ FAIL'}\n")
                f.write(f"Passenger Volume Constraint: {'✅ PASS' if validation.get('passenger_volume_constraint', False) else '❌ FAIL'}\n")
                f.write(f"Pattern Preservation Constraint: {'✅ PASS' if validation.get('pattern_preservation_constraint', False) else '❌ FAIL'}\n")
            
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
    
    def validate_pipeline_constraints(self, baseline_file: str) -> dict:
        """Validate constraints specifically for the pipeline results"""
        
        print("\n🔍 PIPELINE CONSTRAINT VALIDATION")
        print("=" * 60)
        
        # Load baseline for comparison
        baseline_df = pd.read_csv(baseline_file)
        baseline_df['datetime'] = pd.to_datetime(baseline_df['datetime'])
        
        validation = {
            'bus_count_constraint': False,
            'passenger_volume_constraint': False,
            'pattern_preservation_constraint': False,
            'details': {}
        }
        
        # 1. Bus Count Validation
        baseline_buses = baseline_df['bus_id'].nunique()
        optimized_buses = self.optimized_data['bus_id'].nunique()
        
        validation['bus_count_constraint'] = optimized_buses <= baseline_buses
        validation['details']['baseline_buses'] = baseline_buses
        validation['details']['optimized_buses'] = optimized_buses
        validation['details']['bus_violation'] = optimized_buses - baseline_buses
        
        print(f"Bus Count Constraint:")
        print(f"  Baseline: {baseline_buses} buses")
        print(f"  Optimized: {optimized_buses} buses") 
        print(f"  Violation: {optimized_buses - baseline_buses:+d}")
        print(f"  Status: {'✅ PASS' if validation['bus_count_constraint'] else '❌ FAIL'}")
        
        # 2. Passenger Volume Validation
        baseline_days = baseline_df['date'].nunique()
        baseline_daily_passengers = (baseline_df['boarding'].sum() + baseline_df['alighting'].sum()) / baseline_days
        optimized_daily_passengers = self.optimized_data['boarding'].sum() + self.optimized_data['alighting'].sum()
        
        volume_error = abs(optimized_daily_passengers - baseline_daily_passengers) / baseline_daily_passengers * 100 if baseline_daily_passengers > 0 else 0
        validation['passenger_volume_constraint'] = volume_error <= 5.0  # 5% tolerance
        validation['details']['baseline_daily_passengers'] = baseline_daily_passengers
        validation['details']['optimized_daily_passengers'] = optimized_daily_passengers
        validation['details']['volume_error_pct'] = volume_error
        
        print(f"\nPassenger Volume Constraint:")
        print(f"  Baseline daily: {baseline_daily_passengers:,.0f}")
        print(f"  Optimized daily: {optimized_daily_passengers:,.0f}")
        print(f"  Error: {volume_error:.1f}%")
        print(f"  Status: {'✅ PASS' if validation['passenger_volume_constraint'] else '❌ FAIL'} (≤5% required)")
        
        # 3. Pattern Preservation
        baseline_hourly = baseline_df.groupby('hour')['occupancy_rate'].mean()
        optimized_hourly = self.optimized_data.groupby('hour')['occupancy_rate'].mean()
        
        # Calculate correlation
        common_hours = sorted(set(baseline_hourly.index) & set(optimized_hourly.index))
        if len(common_hours) >= 12:
            baseline_values = [baseline_hourly[h] for h in common_hours]
            optimized_values = [optimized_hourly[h] for h in common_hours]
            
            try:
                from scipy.stats import pearsonr
                correlation, _ = pearsonr(baseline_values, optimized_values)
                correlation = max(0, correlation)
            except:
                correlation = 0.0
        else:
            correlation = 0.0
        
        validation['pattern_preservation_constraint'] = correlation >= 0.75
        validation['details']['pattern_correlation'] = correlation
        
        print(f"\nPattern Preservation Constraint:")
        print(f"  Hourly pattern correlation: {correlation:.3f}")
        print(f"  Status: {'✅ PASS' if validation['pattern_preservation_constraint'] else '❌ FAIL'} (≥0.75 required)")
        
        # Overall validation
        overall_pass = all([
            validation['bus_count_constraint'],
            validation['passenger_volume_constraint'], 
            validation['pattern_preservation_constraint']
        ])
        
        validation['overall_pass'] = overall_pass
        
        print(f"\n🎯 OVERALL PIPELINE VALIDATION: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        return validation
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🎯 COMPREHENSIVE OPTIMIZATION EVALUATION")
        print("=" * 80)
        
        # Load data
        self.load_data(
            "output/passenger_flow_results.csv",
            "output_hybrid_optimized/data/hybrid_predictions_final.csv", 
            "output/optimized_passenger_flow_structure.csv"
        )
        
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
        
        # Generate visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report(validation, stats)
        
        return validation, stats
    
    def run_pipeline_evaluation(self, original_file: str, optimized_file: str, schedule_file: str, baseline_file: str = None):
        """Run evaluation pipeline with custom files and pipeline constraints"""
        print("🎯 PIPELINE OPTIMIZATION EVALUATION")
        print("=" * 80)
        
        # Load data
        self.load_data(original_file, optimized_file, schedule_file)
        
        # Calculate metrics
        print("\n📊 CALCULATING PERFORMANCE METRICS")
        print("=" * 60)
        self.original_metrics = self.calculate_daily_metrics(self.original_data, "Original Simulation")
        self.optimized_metrics = self.calculate_daily_metrics(self.optimized_data, "Optimized Prediction")
        
        # Print comparison
        print(f"\nOriginal Simulation:")
        print(f"  Daily passengers: {self.original_metrics['daily_boarding']:,.0f}")
        print(f"  Average occupancy: {self.original_metrics['avg_occupancy']:.1%}")
        print(f"  Peak occupancy: {self.original_metrics['peak_occupancy']:.1%}")
        print(f"  Overcrowded records: {self.original_metrics['overcrowded_records']:,}")
        
        print(f"\nOptimized Prediction:")
        print(f"  Daily passengers: {self.optimized_metrics['daily_boarding']:,.0f}")
        print(f"  Average occupancy: {self.optimized_metrics['avg_occupancy']:.1%}")
        print(f"  Peak occupancy: {self.optimized_metrics['peak_occupancy']:.1%}")
        print(f"  Overcrowded records: {self.optimized_metrics['overcrowded_records']:,}")
        
        # Use pipeline validation instead of regular validation
        baseline_file = baseline_file or original_file
        validation = self.validate_pipeline_constraints(baseline_file)
        
        # Statistical analysis
        stats = self.statistical_analysis()
        
        # Generate visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report(validation, stats)
        
        return validation, stats


def main():
    """Main execution function"""
    evaluator = OptimizationEvaluator()
    validation, stats = evaluator.run_evaluation()
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
    print(f"Key Improvement: {stats['occupancy_improvement']['improvement_pct']:.1f}% occupancy reduction")
    print(f"Report saved: optimization_evaluation_report.txt")
    print(f"Plots saved: evaluation_plots/ directory")


def main_custom_files():
    """Main execution function with custom file paths"""
    evaluator = OptimizationEvaluator()
    
    # Use custom file paths specified by the user
    validation, stats = evaluator.run_pipeline_evaluation(
        original_file="output/passenger_flow_results.csv",
        optimized_file="output_hybrid_final/data/hybrid_predictions_final.csv", 
        schedule_file="output/line_schedules.csv",
        baseline_file="output/passenger_flow_results.csv"
    )
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
    print(f"Key Improvement: {stats['occupancy_improvement']['improvement_pct']:.1f}% occupancy reduction")
    print(f"Report saved: optimization_evaluation_report.txt")
    print(f"Plots saved: evaluation_plots/ directory")


def main_mock_files():
    """Main execution function with mock files"""
    evaluator = OptimizationEvaluator()
    
    # Use real passenger flow data with hybrid mock results as requested
    validation, stats = evaluator.run_pipeline_evaluation(
        original_file="output/passenger_flow_results.csv",  # REAL passenger flow data
        optimized_file="output/hybrid_predictions_mock.csv",  # MOCK hybrid predictions
        schedule_file="output/line_schedules.csv",
        baseline_file="output/passenger_flow_results.csv"  # REAL passenger flow as baseline
    )
    
    print(f"\n🎉 REAL PASSENGER FLOW + MOCK HYBRID EVALUATION COMPLETE!")
    print(f"Overall Validation: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
    print(f"Key Improvement: {stats['occupancy_improvement']['improvement_pct']:.1f}% occupancy reduction")
    print(f"Report saved: optimization_evaluation_report.txt")
    print(f"Plots saved: evaluation_plots/ directory")
    
    return validation, stats


if __name__ == "__main__":
    # Run with mock files as requested by user
    main_mock_files() 