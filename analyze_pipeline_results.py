#!/usr/bin/env python3
"""
Comprehensive analysis of the pipeline results to understand performance issues.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_pipeline_results():
    """Analyze the complete pipeline results"""
    
    print("🔍 COMPREHENSIVE PIPELINE RESULTS ANALYSIS")
    print("=" * 80)
    
    # Latest run directory
    run_dir = "outputs/pipeline_run_quick_20250613_042324"
    
    # Load all data
    print("📄 Loading data files...")
    sim_data = pd.read_csv(f"{run_dir}/simulation_results/passenger_flow_results.csv")
    pred_data = pd.read_csv(f"{run_dir}/prediction_results/predicted_passenger_flow.csv")
    opt_schedules = pd.read_csv(f"{run_dir}/optimization_results/optimized_schedules.csv")
    
    with open(f"{run_dir}/optimization_results/optimization_metrics.json") as f:
        opt_metrics = json.load(f)
    
    print(f"✅ Simulation data: {len(sim_data):,} records")
    print(f"✅ Prediction data: {len(pred_data):,} records")
    print(f"✅ Optimization schedules: {len(opt_schedules):,} schedules")
    
    print("\n" + "="*60)
    print("1. OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Optimization success
    print(f"🎯 Lines optimized: {len(opt_metrics['lines_optimized'])}/10")
    print(f"📊 Total schedules generated: {sum(opt_metrics['schedules_per_line'].values())}")
    print(f"🏆 Average fitness score: {sum(opt_metrics['fitness_scores'].values()) / len(opt_metrics['fitness_scores']):.4f}")
    
    print("\n📈 Fitness scores by line:")
    for line_id, fitness in sorted(opt_metrics['fitness_scores'].items()):
        schedules = opt_metrics['schedules_per_line'][line_id]
        print(f"   Line {line_id}: {fitness:.4f} ({schedules} schedules)")
    
    # Identify best and worst performing lines
    fitness_scores = opt_metrics['fitness_scores']
    best_line = max(fitness_scores, key=fitness_scores.get)
    worst_line = min(fitness_scores, key=fitness_scores.get)
    
    print(f"\n🥇 Best performing line: {best_line} (fitness: {fitness_scores[best_line]:.4f})")
    print(f"🥉 Worst performing line: {worst_line} (fitness: {fitness_scores[worst_line]:.4f})")
    
    print("\n" + "="*60)
    print("2. DATA COVERAGE ANALYSIS")
    print("="*60)
    
    # Convert datetime columns
    sim_data['datetime'] = pd.to_datetime(sim_data['datetime'])
    pred_data['datetime'] = pd.to_datetime(pred_data['datetime'])
    
    # Line coverage
    sim_lines = set(sim_data['line_id'].unique())
    pred_lines = set(pred_data['line_id'].unique())
    opt_lines = set(opt_schedules['line_id'].unique())
    
    print(f"🚌 Simulation lines: {len(sim_lines)} - {sorted(sim_lines)}")
    print(f"🚌 Prediction lines: {len(pred_lines)} - {sorted(pred_lines)}")
    print(f"🚌 Optimization lines: {len(opt_lines)} - {sorted(opt_lines)}")
    print(f"✅ Line coverage: {len(pred_lines)/len(sim_lines)*100:.1f}%")
    
    # Stop coverage
    sim_stops = set(sim_data['stop_id'].unique())
    pred_stops = set(pred_data['stop_id'].unique())
    
    print(f"\n🚏 Simulation stops: {len(sim_stops)}")
    print(f"🚏 Prediction stops: {len(pred_stops)}")
    print(f"✅ Stop coverage: {len(pred_stops)/len(sim_stops)*100:.1f}%")
    
    # Time coverage
    sim_hours = set(sim_data['datetime'].dt.hour)
    pred_hours = set(pred_data['datetime'].dt.hour)
    
    print(f"\n⏰ Simulation hours: {sorted(sim_hours)}")
    print(f"⏰ Prediction hours: {sorted(pred_hours)}")
    print(f"✅ Hour coverage: {len(pred_hours)/len(sim_hours)*100:.1f}%")
    
    print("\n" + "="*60)
    print("3. PASSENGER FLOW ANALYSIS")
    print("="*60)
    
    # Passenger statistics
    sim_stats = {
        'total_boarding': sim_data['boarding'].sum(),
        'total_alighting': sim_data['alighting'].sum(),
        'avg_boarding': sim_data['boarding'].mean(),
        'avg_alighting': sim_data['alighting'].mean(),
        'avg_occupancy': sim_data['occupancy_rate'].mean()
    }
    
    pred_stats = {
        'total_boarding': pred_data['boarding'].sum(),
        'total_alighting': pred_data['alighting'].sum(),
        'avg_boarding': pred_data['boarding'].mean(),
        'avg_alighting': pred_data['alighting'].mean(),
        'avg_occupancy': pred_data['occupancy_rate'].mean()
    }
    
    print("📊 Simulation vs Prediction Comparison:")
    print(f"   Total boarding:    {sim_stats['total_boarding']:>10,} vs {pred_stats['total_boarding']:>10,} ({pred_stats['total_boarding']/sim_stats['total_boarding']*100:.1f}%)")
    print(f"   Total alighting:   {sim_stats['total_alighting']:>10,} vs {pred_stats['total_alighting']:>10,} ({pred_stats['total_alighting']/sim_stats['total_alighting']*100:.1f}%)")
    print(f"   Avg boarding:      {sim_stats['avg_boarding']:>10.1f} vs {pred_stats['avg_boarding']:>10.1f} ({pred_stats['avg_boarding']/sim_stats['avg_boarding']*100:.1f}%)")
    print(f"   Avg alighting:     {sim_stats['avg_alighting']:>10.1f} vs {pred_stats['avg_alighting']:>10.1f} ({pred_stats['avg_alighting']/sim_stats['avg_alighting']*100:.1f}%)")
    print(f"   Avg occupancy:     {sim_stats['avg_occupancy']:>10.1%} vs {pred_stats['avg_occupancy']:>10.1%}")
    
    print("\n" + "="*60)
    print("4. PREDICTION QUALITY ANALYSIS")
    print("="*60)
    
    # Check prediction model performance indicators
    if 'boarding_lstm' in pred_data.columns:
        lstm_boarding = pred_data['boarding_lstm'].mean()
        prophet_boarding = pred_data['boarding_prophet'].mean()
        final_boarding = pred_data['boarding'].mean()
        
        print(f"🤖 LSTM avg boarding prediction: {lstm_boarding:.1f}")
        print(f"📈 Prophet avg boarding prediction: {prophet_boarding:.1f}")
        print(f"🔀 Final hybrid prediction: {final_boarding:.1f}")
        
        # Check if predictions are reasonable
        if lstm_boarding > 100 or prophet_boarding > 100:
            print("⚠️  WARNING: Very high prediction values detected")
        
        # Check prediction variance
        lstm_std = pred_data['boarding_lstm'].std()
        prophet_std = pred_data['boarding_prophet'].std()
        print(f"📊 LSTM prediction variance: {lstm_std:.1f}")
        print(f"📊 Prophet prediction variance: {prophet_std:.1f}")
    
    # Occupancy analysis
    overcrowded_sim = (sim_data['occupancy_rate'] > 1.0).sum()
    overcrowded_pred = (pred_data['occupancy_rate'] > 1.0).sum()
    
    print(f"\n🚌 Overcrowding Analysis:")
    print(f"   Simulation overcrowded: {overcrowded_sim:,} ({overcrowded_sim/len(sim_data)*100:.1f}%)")
    print(f"   Prediction overcrowded: {overcrowded_pred:,} ({overcrowded_pred/len(pred_data)*100:.1f}%)")
    
    # Extreme occupancy cases
    max_sim_occupancy = sim_data['occupancy_rate'].max()
    max_pred_occupancy = pred_data['occupancy_rate'].max()
    
    print(f"   Max simulation occupancy: {max_sim_occupancy:.1%}")
    print(f"   Max prediction occupancy: {max_pred_occupancy:.1%}")
    
    if max_pred_occupancy > 5.0:  # 500%+
        print("⚠️  WARNING: Extremely high occupancy rates detected in predictions")
    
    print("\n" + "="*60)
    print("5. SCHEDULE OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Schedule timing analysis
    opt_schedules['departure_time'] = pd.to_datetime(opt_schedules['departure_time'])
    opt_schedules['hour'] = opt_schedules['departure_time'].dt.hour
    
    print("📅 Optimized schedule distribution by hour:")
    hourly_dist = opt_schedules['hour'].value_counts().sort_index()
    for hour, count in hourly_dist.items():
        print(f"   Hour {hour:2d}: {count:3d} departures")
    
    # Compare with simulation hourly patterns
    sim_hourly = sim_data.groupby(sim_data['datetime'].dt.hour)['boarding'].sum()
    opt_hourly = opt_schedules['hour'].value_counts().sort_index()
    
    print(f"\n📊 Schedule vs Demand Alignment:")
    print("   Hour | Sim Demand | Opt Schedules | Ratio")
    print("   -----|------------|---------------|------")
    for hour in sorted(set(sim_hourly.index) | set(opt_hourly.index)):
        sim_demand = sim_hourly.get(hour, 0)
        opt_supply = opt_hourly.get(hour, 0)
        ratio = opt_supply / (sim_demand/1000) if sim_demand > 0 else 0
        print(f"   {hour:2d}   | {sim_demand:>10,} | {opt_supply:>13} | {ratio:>5.2f}")
    
    print("\n" + "="*60)
    print("6. KEY INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # Coverage insights
    if len(pred_lines) == len(sim_lines):
        insights.append("✅ Full line coverage achieved (all 10 lines optimized)")
    else:
        insights.append(f"⚠️  Partial line coverage ({len(pred_lines)}/{len(sim_lines)} lines)")
    
    # Data scale insights
    data_ratio = len(pred_data) / len(sim_data)
    if data_ratio < 0.5:
        insights.append(f"⚠️  Prediction data is much smaller than simulation ({data_ratio:.1%})")
    
    # Passenger flow insights
    passenger_ratio = pred_stats['total_boarding'] / sim_stats['total_boarding']
    if passenger_ratio < 0.8:
        insights.append(f"⚠️  Significant passenger flow reduction ({passenger_ratio:.1%})")
    elif passenger_ratio > 1.2:
        insights.append(f"⚠️  Significant passenger flow increase ({passenger_ratio:.1%})")
    
    # Fitness insights
    avg_fitness = sum(opt_metrics['fitness_scores'].values()) / len(opt_metrics['fitness_scores'])
    if avg_fitness > 0.8:
        insights.append(f"✅ Good optimization fitness scores (avg: {avg_fitness:.3f})")
    elif avg_fitness > 0.7:
        insights.append(f"⚠️  Moderate optimization fitness scores (avg: {avg_fitness:.3f})")
    else:
        insights.append(f"❌ Poor optimization fitness scores (avg: {avg_fitness:.3f})")
    
    # Occupancy insights
    if max_pred_occupancy > 10.0:
        insights.append("❌ Extremely unrealistic occupancy rates (>1000%)")
    elif max_pred_occupancy > 2.0:
        insights.append("⚠️  High occupancy rates detected (>200%)")
    
    # Print insights
    for insight in insights:
        print(f"   {insight}")
    
    print("\n📋 RECOMMENDATIONS:")
    recommendations = [
        "1. Investigate why prediction occupancy rates are so high (>3000%)",
        "2. Check if OptimizedPassengerFlowGenerator is creating realistic passenger loads",
        "3. Verify that prediction model constraints are working properly",
        "4. Consider adjusting genetic algorithm fitness function weights",
        "5. Analyze why passenger flow is reduced compared to simulation"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    return {
        'sim_data': sim_data,
        'pred_data': pred_data,
        'opt_schedules': opt_schedules,
        'opt_metrics': opt_metrics,
        'insights': insights
    }

if __name__ == "__main__":
    results = analyze_pipeline_results() 