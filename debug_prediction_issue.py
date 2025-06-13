#!/usr/bin/env python3
"""
Debug script to analyze why the prediction model is performing poorly.
This will help us understand the data mismatch between training and prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_data_mismatch():
    """Analyze the mismatch between training and prediction data"""
    
    print("🔍 DEBUGGING PREDICTION MODEL PERFORMANCE")
    print("=" * 60)
    
    # Load the latest run data
    latest_run = "outputs/pipeline_run_quick_20250613_031137"
    
    # Load training data (simulation results)
    train_file = f"{latest_run}/simulation_results/passenger_flow_results.csv"
    print(f"📄 Loading training data: {train_file}")
    
    # Load prediction structure data (optimized schedule flow)
    pred_file = f"{latest_run}/optimization_results/optimized_schedules_passenger_flow.csv"
    print(f"📄 Loading prediction structure: {pred_file}")
    
    try:
        # Load data with limited rows for analysis
        train_df = pd.read_csv(train_file, nrows=10000)  # Sample for speed
        pred_df = pd.read_csv(pred_file)
        
        print(f"✅ Loaded {len(train_df):,} training records")
        print(f"✅ Loaded {len(pred_df):,} prediction structure records")
        
        # Convert datetime columns
        train_df['datetime'] = pd.to_datetime(train_df['datetime'])
        pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
        
        # Add hour columns for analysis
        train_df['hour'] = train_df['datetime'].dt.hour
        pred_df['hour'] = pred_df['datetime'].dt.hour
        
        print("\\n" + "="*50)
        print("DATA STRUCTURE COMPARISON")
        print("="*50)
        
        # Compare basic statistics
        print("\\n📊 BASIC STATISTICS:")
        print(f"Training data time range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        print(f"Prediction data time range: {pred_df['datetime'].min()} to {pred_df['datetime'].max()}")
        
        print(f"\\nTraining unique stops: {train_df['stop_id'].nunique()}")
        print(f"Prediction unique stops: {pred_df['stop_id'].nunique()}")
        
        print(f"\\nTraining unique lines: {train_df['line_id'].nunique()}")
        print(f"Prediction unique lines: {pred_df['line_id'].nunique()}")
        
        # Check stop overlap
        train_stops = set(train_df['stop_id'].unique())
        pred_stops = set(pred_df['stop_id'].unique())
        common_stops = train_stops.intersection(pred_stops)
        
        print(f"\\n🚏 STOP OVERLAP ANALYSIS:")
        print(f"Common stops: {len(common_stops)} / {len(train_stops)} training stops")
        print(f"Stop overlap percentage: {len(common_stops)/len(train_stops)*100:.1f}%")
        
        if len(common_stops) < len(train_stops):
            missing_in_pred = train_stops - pred_stops
            print(f"Stops in training but not prediction: {len(missing_in_pred)}")
            print(f"Sample missing stops: {list(missing_in_pred)[:10]}")
        
        # Check line overlap
        train_lines = set(str(x) for x in train_df['line_id'].unique())
        pred_lines = set(str(x) for x in pred_df['line_id'].unique())
        common_lines = train_lines.intersection(pred_lines)
        
        print(f"\\n🚌 LINE OVERLAP ANALYSIS:")
        print(f"Training lines: {sorted(train_lines)}")
        print(f"Prediction lines: {sorted(pred_lines)}")
        print(f"Common lines: {sorted(common_lines)}")
        
        # Analyze hourly patterns
        print("\\n⏰ HOURLY PATTERN ANALYSIS:")
        train_hourly = train_df.groupby('hour').agg({
            'boarding': ['count', 'mean', 'sum'],
            'alighting': ['count', 'mean', 'sum']
        }).round(2)
        
        pred_hourly = pred_df.groupby('hour').agg({
            'boarding': 'count'  # Only count since boarding/alighting are None
        })
        
        print("\\nTraining data hourly boarding patterns:")
        for hour in sorted(train_df['hour'].unique()):
            if hour in train_hourly.index:
                count = train_hourly.loc[hour, ('boarding', 'count')]
                mean_boarding = train_hourly.loc[hour, ('boarding', 'mean')]
                print(f"  Hour {hour:2d}: {count:4d} records, avg boarding: {mean_boarding:5.1f}")
        
        print("\\nPrediction structure hourly distribution:")
        for hour in sorted(pred_df['hour'].unique()):
            if hour in pred_hourly.index:
                count = pred_hourly.loc[hour, 'boarding']
                print(f"  Hour {hour:2d}: {count:4d} records")
        
        # Check if prediction hours match training hours
        train_hours = set(train_df['hour'].unique())
        pred_hours = set(pred_df['hour'].unique())
        common_hours = train_hours.intersection(pred_hours)
        
        print(f"\\n⏰ HOUR OVERLAP:")
        print(f"Training hours: {sorted(train_hours)}")
        print(f"Prediction hours: {sorted(pred_hours)}")
        print(f"Common hours: {sorted(common_hours)}")
        print(f"Hour overlap: {len(common_hours)}/{len(train_hours)} = {len(common_hours)/len(train_hours)*100:.1f}%")
        
        # Analyze passenger flow patterns
        print("\\n🚶 PASSENGER FLOW ANALYSIS:")
        train_stats = {
            'total_boarding': train_df['boarding'].sum(),
            'total_alighting': train_df['alighting'].sum(),
            'avg_boarding': train_df['boarding'].mean(),
            'avg_alighting': train_df['alighting'].mean(),
            'max_boarding': train_df['boarding'].max(),
            'max_alighting': train_df['alighting'].max()
        }
        
        print("Training data passenger flow:")
        for key, value in train_stats.items():
            print(f"  {key}: {value:,.1f}")
        
        # Check for data quality issues
        print("\\n🔍 DATA QUALITY CHECK:")
        
        # Training data quality
        train_null_boarding = train_df['boarding'].isnull().sum()
        train_null_alighting = train_df['alighting'].isnull().sum()
        train_zero_boarding = (train_df['boarding'] == 0).sum()
        train_zero_alighting = (train_df['alighting'] == 0).sum()
        
        print("Training data quality:")
        print(f"  Null boarding: {train_null_boarding} ({train_null_boarding/len(train_df)*100:.1f}%)")
        print(f"  Null alighting: {train_null_alighting} ({train_null_alighting/len(train_df)*100:.1f}%)")
        print(f"  Zero boarding: {train_zero_boarding} ({train_zero_boarding/len(train_df)*100:.1f}%)")
        print(f"  Zero alighting: {train_zero_alighting} ({train_zero_alighting/len(train_df)*100:.1f}%)")
        
        # Prediction data quality
        pred_null_boarding = pred_df['boarding'].isnull().sum()
        pred_null_alighting = pred_df['alighting'].isnull().sum()
        
        print("\\nPrediction structure quality:")
        print(f"  Null boarding: {pred_null_boarding} ({pred_null_boarding/len(pred_df)*100:.1f}%)")
        print(f"  Null alighting: {pred_null_alighting} ({pred_null_alighting/len(pred_df)*100:.1f}%)")
        
        # Key insights
        print("\\n" + "="*50)
        print("🎯 KEY INSIGHTS")
        print("="*50)
        
        insights = []
        
        if len(common_stops)/len(train_stops) < 0.8:
            insights.append(f"⚠️  Low stop overlap ({len(common_stops)/len(train_stops)*100:.1f}%) - model may not recognize many stops")
        
        if len(common_hours)/len(train_hours) < 0.8:
            insights.append(f"⚠️  Limited hour overlap ({len(common_hours)/len(train_hours)*100:.1f}%) - timing patterns may not match")
        
        if pred_null_boarding > 0:
            insights.append(f"⚠️  Prediction structure has {pred_null_boarding} null boarding values - model expects real values")
        
        # Check if prediction times are very different from training times
        train_time_range = (train_df['datetime'].max() - train_df['datetime'].min()).total_seconds() / 3600
        pred_time_range = (pred_df['datetime'].max() - pred_df['datetime'].min()).total_seconds() / 3600
        
        if abs(train_time_range - pred_time_range) > 2:  # More than 2 hours difference
            insights.append(f"⚠️  Time range mismatch - Training: {train_time_range:.1f}h, Prediction: {pred_time_range:.1f}h")
        
        if len(insights) == 0:
            insights.append("✅ Data structures look compatible - issue may be in model architecture or feature engineering")
        
        for insight in insights:
            print(insight)
        
        print("\\n" + "="*50)
        print("💡 RECOMMENDATIONS")
        print("="*50)
        
        recommendations = [
            "1. Check if OptimizedPassengerFlowGenerator is creating realistic time patterns",
            "2. Verify that the same stops and lines are used in both training and prediction",
            "3. Consider using pattern-based fallback for unknown stops/times",
            "4. Check if the model is actually being used vs. falling back to historical patterns"
        ]
        
        for rec in recommendations:
            print(rec)
        
        return {
            'train_df': train_df,
            'pred_df': pred_df,
            'common_stops': common_stops,
            'common_hours': common_hours,
            'insights': insights
        }
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = analyze_data_mismatch() 