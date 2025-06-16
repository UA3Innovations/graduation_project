#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

try:
    # Temporarily suppress stderr to hide plotly warning
    import sys
    import io
    from contextlib import redirect_stderr
    
    with redirect_stderr(io.StringIO()):
        from prophet import Prophet
        
except ImportError:
    print("Error: Prophet not installed. Install with: pip install prophet")
    sys.exit(1)

# Set random seeds for reproducibility
np.random.seed(42)

class ProphetModel:
    """
    Prophet model that preserves and validates against historical patterns
    with route-aware fixes, last stop logic, realistic boarding constraints, 
    and night-time logic (1-4 AM only) - similar to LSTM approach
    """
    
    def __init__(self):
        self.models = {}  # Will store Prophet models for each stop
        self.is_trained = False
        
        # Pattern preservation data (same as LSTM)
        self.historical_patterns = {}
        self.stop_patterns = {}
        self.hourly_patterns = {}
        self.training_stop_ids = set()
        
        # Route and conservation data (same as LSTM)
        self.route_patterns = {}
        self.stop_classifications = {}
        self.conservation_stats = {}
        
        # Last stop data (same as LSTM)
        self.last_stops = {}
        self.bus_stops_df = None
        
        # Prophet-specific settings
        self.prophet_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'uncertainty_samples': 100
        }
        
    def _load_last_stops(self, bus_stops_csv_path):
        """Load bus stops CSV and identify last stops for each line (same as LSTM)"""
        try:
            print(f"Loading bus stops data from {bus_stops_csv_path}")
            self.bus_stops_df = pd.read_csv(bus_stops_csv_path)
            
            # Ensure consistent column names
            if 'Line ID' in self.bus_stops_df.columns:
                self.bus_stops_df = self.bus_stops_df.rename(columns={
                    'Line ID': 'line_id',
                    'Stop ID': 'stop_id', 
                    'Stop Name': 'stop_name'
                })
            
            print(f"Loaded {len(self.bus_stops_df)} bus stop records")
            print(f"Columns: {list(self.bus_stops_df.columns)}")
            
            # Convert line_id to string to ensure consistent matching
            self.bus_stops_df['line_id'] = self.bus_stops_df['line_id'].astype(str)
            
            # Group by line and find last stop
            last_stops = {}
            unique_lines = sorted(self.bus_stops_df['line_id'].unique())
            
            print(f"\nProcessing {len(unique_lines)} unique lines...")
            
            for line_id in unique_lines:
                line_stops = self.bus_stops_df[self.bus_stops_df['line_id'] == line_id].copy()
                line_stops = line_stops.reset_index(drop=True)
                
                if len(line_stops) > 0:
                    last_stop_id = int(line_stops.iloc[-1]['stop_id'])
                    last_stop_name = line_stops.iloc[-1]['stop_name']
                    last_stops[str(line_id)] = last_stop_id
                    
                    print(f"  Line {line_id}: {len(line_stops)} stops, last stop = {last_stop_id} ({last_stop_name})")
            
            self.last_stops = last_stops
            print(f"\n=== DETECTED LAST STOPS FOR ALL LINES ===")
            
            for line_id in sorted(self.last_stops.keys()):
                stop_id = self.last_stops[line_id]
                matching_rows = self.bus_stops_df[
                    (self.bus_stops_df['line_id'] == line_id) & 
                    (self.bus_stops_df['stop_id'] == stop_id)
                ]
                stop_name = matching_rows['stop_name'].iloc[0] if len(matching_rows) > 0 else "Unknown"
                print(f"  Line '{line_id}' → Last Stop {stop_id} ({stop_name})")
            
            print(f"\nTotal lines with last stops detected: {len(self.last_stops)}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not load bus stops CSV: {e}")
            print("Last stop logic will be disabled")
            import traceback
            traceback.print_exc()
            return False
    
    def _is_last_stop(self, line_id, stop_id):
        """Check if a given stop is the last stop for its line (same as LSTM)"""
        if not self.last_stops:
            return False
        
        line_key = str(line_id)
        stop_id_int = int(stop_id)
        
        if line_key in self.last_stops:
            is_last = stop_id_int == self.last_stops[line_key]
            if is_last:
                print(f"    ✓ Last stop detected: Line '{line_key}' Stop {stop_id_int}")
            return is_last
        else:
            if line_key not in getattr(self, '_missing_lines_shown', set()):
                print(f"    ⚠️  Line '{line_key}' not found in last stops mapping")
                if not hasattr(self, '_missing_lines_shown'):
                    self._missing_lines_shown = set()
                self._missing_lines_shown.add(line_key)
        
        return False
        
    def _ensure_stop_id_consistency(self, df):
        """Ensure stop_id and line_id are consistent data types (same as LSTM)"""
        df = df.copy()
        df['stop_id'] = df['stop_id'].astype(int)
        df['line_id'] = df['line_id'].astype(str)
        return df
        
    def analyze_historical_patterns(self, df):
        """Comprehensive analysis of historical patterns (same as LSTM)"""
        print("=== ANALYZING HISTORICAL PATTERNS ===")
        
        df_temp = self._ensure_stop_id_consistency(df)
        df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
        df_temp['hour'] = df_temp['datetime'].dt.hour
        df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
        df_temp['is_weekend'] = df_temp['day_of_week'] >= 5
        
        # Store training stop IDs
        self.training_stop_ids = set(df_temp['stop_id'].unique())
        print(f"Training data contains {len(self.training_stop_ids)} unique stops")
        print(f"Stop ID range: {min(self.training_stop_ids)} - {max(self.training_stop_ids)}")
        
        # Overall hourly patterns
        weekday_data = df_temp[~df_temp['is_weekend']]
        weekend_data = df_temp[df_temp['is_weekend']]
        
        self.hourly_patterns = {
            'weekday': weekday_data.groupby('hour')[['boarding', 'alighting']].agg(['mean', 'std', 'min', 'max']),
            'weekend': weekend_data.groupby('hour')[['boarding', 'alighting']].agg(['mean', 'std', 'min', 'max']) if len(weekend_data) > 0 else None
        }
        
        # Stop-specific patterns
        for stop_id, stop_data in df_temp.groupby('stop_id'):
            stop_weekday = stop_data[~stop_data['is_weekend']]
            if len(stop_weekday) > 24:
                hourly_pattern = stop_weekday.groupby('hour')[['boarding', 'alighting']].mean()
                self.stop_patterns[stop_id] = {
                    'hourly_mean': hourly_pattern,
                    'daily_mean': stop_data[['boarding', 'alighting']].mean(),
                    'daily_std': stop_data[['boarding', 'alighting']].std(),
                    'peak_hours': hourly_pattern['boarding'].nlargest(3).index.tolist(),
                    'data_count': len(stop_data)
                }
        
        print(f"Analyzed patterns for {len(self.stop_patterns)} stops")
        print(f"Weekday hourly patterns: {len(self.hourly_patterns['weekday'])} hours")
        
        # Display key patterns
        if len(self.hourly_patterns['weekday']) > 0:
            weekday_boarding = self.hourly_patterns['weekday']['boarding']['mean']
            peak_hours = weekday_boarding.nlargest(5).index.tolist()
            print(f"Peak hours (weekday): {peak_hours}")
            print(f"Peak hour avg boarding: {weekday_boarding[peak_hours].mean():.1f}")
            print(f"Off-peak avg boarding: {weekday_boarding.drop(peak_hours).mean():.1f}")
        
        # Analyze route patterns and last stop patterns (same as LSTM)
        self._analyze_route_patterns(df_temp)
        
        if self.last_stops:
            self._analyze_last_stop_patterns(df_temp)
    
    def _analyze_last_stop_patterns(self, df):
        """Analyze patterns at last stops (same as LSTM)"""
        print("\n=== ANALYZING LAST STOP PATTERNS ===")
        
        last_stop_data = []
        for idx, row in df.iterrows():
            if self._is_last_stop(row['line_id'], row['stop_id']):
                last_stop_data.append(row)
        
        if last_stop_data:
            last_stop_df = pd.DataFrame(last_stop_data)
            print(f"Found {len(last_stop_df)} records at last stops")
            
            avg_boarding = last_stop_df['boarding'].mean()
            avg_alighting = last_stop_df['alighting'].mean()
            zero_boarding_pct = (last_stop_df['boarding'] == 0).mean() * 100
            
            print(f"Last stop averages:")
            print(f"  Boarding: {avg_boarding:.2f} (should be ~0)")
            print(f"  Alighting: {avg_alighting:.2f}")
            print(f"  Zero boarding rate: {zero_boarding_pct:.1f}%")
            
            if avg_boarding > 2:
                print("⚠️  Note: High boarding at last stops detected - will be corrected")
        else:
            print("No last stop data found in training set")
    
    def _analyze_route_patterns(self, df):
        """Analyze route patterns (same as LSTM)"""
        print("\n=== ANALYZING ROUTE PATTERNS FOR CONSERVATION ===")
        
        df = df.copy()
        df = df.sort_values(['line_id', 'bus_id', 'datetime']).reset_index(drop=True)
        
        route_analysis = {}
        
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id].copy()
            
            stop_sequences = []
            for bus_id in line_data['bus_id'].unique():
                bus_data = line_data[line_data['bus_id'] == bus_id].sort_values('datetime')
                stop_sequence = bus_data['stop_id'].tolist()
                if len(stop_sequence) > 3:
                    stop_sequences.append(stop_sequence)
            
            if stop_sequences:
                all_stops = []
                for seq in stop_sequences:
                    all_stops.extend(seq)
                
                unique_stops = list(set(all_stops))
                stop_positions = {}
                
                for stop in unique_stops:
                    positions = []
                    for seq in stop_sequences:
                        if stop in seq:
                            pos_idx = seq.index(stop)
                            position_pct = pos_idx / (len(seq) - 1) if len(seq) > 1 else 0.5
                            positions.append(position_pct)
                    
                    if positions:
                        stop_positions[stop] = {
                            'avg_position': np.mean(positions),
                            'position_std': np.std(positions),
                            'frequency': len(positions)
                        }
                
                route_analysis[line_id] = {
                    'stops': unique_stops,
                    'stop_positions': stop_positions,
                    'total_sequences': len(stop_sequences)
                }
        
        self.route_patterns = route_analysis
        self._classify_stops_by_position()
        print(f"Analyzed {len(route_analysis)} routes for conservation")
    
    def _classify_stops_by_position(self):
        """Classify stops by route position (same as LSTM)"""
        for line_id, route_data in self.route_patterns.items():
            for stop_id, pos_data in route_data['stop_positions'].items():
                avg_pos = pos_data['avg_position']
                
                if avg_pos < 0.3:
                    classification = 'ORIGIN'
                elif avg_pos > 0.7:
                    classification = 'DESTINATION'
                else:
                    classification = 'MIDDLE'
                
                if stop_id not in self.stop_classifications:
                    self.stop_classifications[stop_id] = {}
                
                self.stop_classifications[stop_id][line_id] = {
                    'type': classification,
                    'position': avg_pos,
                    'confidence': 1.0 / (pos_data['position_std'] + 0.1)
                }
        
        # Aggregate classifications
        final_classifications = {}
        for stop_id, line_data in self.stop_classifications.items():
            type_scores = {'ORIGIN': 0, 'MIDDLE': 0, 'DESTINATION': 0}
            
            for line_id, class_data in line_data.items():
                stop_type = class_data['type']
                confidence = class_data['confidence']
                type_scores[stop_type] += confidence
            
            final_type = max(type_scores, key=type_scores.get)
            final_classifications[stop_id] = {
                'primary_type': final_type,
                'confidence': type_scores[final_type] / sum(type_scores.values()),
                'scores': type_scores
            }
        
        self.stop_classifications = final_classifications
        
        type_counts = {}
        for stop_data in self.stop_classifications.values():
            stop_type = stop_data['primary_type']
            type_counts[stop_type] = type_counts.get(stop_type, 0) + 1
        
        print(f"Stop classifications: {type_counts}")
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet training"""
        print("Preparing data for Prophet training...")
        
        data = self._ensure_stop_id_consistency(df)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values(['stop_id', 'datetime']).reset_index(drop=True)
        
        # Add time features for Prophet
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_peak_hour'] = data['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        data['is_morning_peak'] = data['hour'].isin([7, 8, 9]).astype(int)
        data['is_evening_peak'] = data['hour'].isin([17, 18, 19]).astype(int)
        data['is_night_time'] = data['hour'].isin([1, 2, 3, 4]).astype(int)
        
        # Last stop indicator
        data['is_last_stop'] = 0
        if self.last_stops:
            for idx, row in data.iterrows():
                if self._is_last_stop(row['line_id'], row['stop_id']):
                    data.loc[idx, 'is_last_stop'] = 1
        
        print(f"Prepared data with Prophet features for {len(data)} records")
        return data
    
    def train_prophet_models(self, df, target_columns=['boarding', 'alighting']):
        """Train Prophet models for each stop and target variable"""
        print("Training Prophet models...")
        
        prepared_data = self.prepare_prophet_data(df)
        
        # Train separate models for each stop and target
        model_count = 0
        for stop_id in self.training_stop_ids:
            stop_data = prepared_data[prepared_data['stop_id'] == stop_id].copy()
            
            if len(stop_data) < 50:  # Need minimum data for Prophet
                print(f"  Skipping stop {stop_id}: insufficient data ({len(stop_data)} records)")
                continue
            
            self.models[stop_id] = {}
            
            for target in target_columns:
                # Prepare Prophet format
                prophet_df = pd.DataFrame({
                    'ds': stop_data['datetime'],
                    'y': stop_data[target]
                })
                
                # Add regressors
                prophet_df['hour'] = stop_data['hour']
                prophet_df['is_weekend'] = stop_data['is_weekend']
                prophet_df['is_peak_hour'] = stop_data['is_peak_hour']
                prophet_df['is_morning_peak'] = stop_data['is_morning_peak']
                prophet_df['is_evening_peak'] = stop_data['is_evening_peak']
                prophet_df['is_night_time'] = stop_data['is_night_time']
                prophet_df['is_last_stop'] = stop_data['is_last_stop']
                
                # Create and configure Prophet model
                model = Prophet(**self.prophet_params)
                
                # Add regressors
                model.add_regressor('hour')
                model.add_regressor('is_weekend')
                model.add_regressor('is_peak_hour')
                model.add_regressor('is_morning_peak')
                model.add_regressor('is_evening_peak')
                model.add_regressor('is_night_time')
                model.add_regressor('is_last_stop')
                
                # Train model
                try:
                    model.fit(prophet_df)
                    self.models[stop_id][target] = model
                    model_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to train {target} model for stop {stop_id}: {e}")
        
        print(f"Successfully trained {model_count} Prophet models")
        self.is_trained = True
    
    def _predict_from_patterns(self, future_row, stop_id=None):
        """Predict using historical patterns when Prophet fails (same as LSTM)"""
        future_dt = pd.to_datetime(future_row['datetime'])
        hour = future_dt.hour
        is_weekend = future_dt.dayofweek >= 5
        
        # Use stop-specific patterns if available
        if stop_id and stop_id in self.stop_patterns:
            stop_pattern = self.stop_patterns[stop_id]
            if hour in stop_pattern['hourly_mean'].index:
                boarding = stop_pattern['hourly_mean'].loc[hour, 'boarding']
                alighting = stop_pattern['hourly_mean'].loc[hour, 'alighting']
            else:
                boarding = stop_pattern['daily_mean']['boarding']
                alighting = stop_pattern['daily_mean']['alighting']
        else:
            # Use global hourly patterns
            pattern_key = 'weekend' if is_weekend and self.hourly_patterns.get('weekend') is not None else 'weekday'
            if pattern_key in self.hourly_patterns and hour in self.hourly_patterns[pattern_key].index:
                boarding = self.hourly_patterns[pattern_key].loc[hour, ('boarding', 'mean')]
                alighting = self.hourly_patterns[pattern_key].loc[hour, ('alighting', 'mean')]
            else:
                boarding = 8.0
                alighting = 7.0
        
        return max(0, round(boarding)), max(0, round(alighting))
    
    # Include all the constraint methods from LSTM (same implementation)
    def _get_night_time_multiplier(self, hour):
        """Get realistic night-time multipliers (same as LSTM)"""
        if hour == 1:
            return 0.3
        elif hour == 2:
            return 0.2
        elif hour in [3, 4]:
            return 0.1
        else:
            return 1.0
    
    def _apply_night_time_logic(self, current_load, predicted_boarding, predicted_alighting, hour, capacity):
        """Apply night-time specific logic (same as LSTM)"""
        time_multiplier = self._get_night_time_multiplier(hour)
        
        adjusted_boarding = max(0, round(predicted_boarding * time_multiplier))
        adjusted_alighting = max(0, round(predicted_alighting * time_multiplier))
        
        if 2 <= hour <= 4:
            print(f"    Applying night-time constraints for hour {hour}")
            
            if adjusted_boarding > 5:
                adjusted_boarding = np.random.choice([2, 3, 4])
            
            max_reasonable_night_load = 15
            if current_load > max_reasonable_night_load:
                excess_load = current_load - max_reasonable_night_load
                adjusted_alighting = min(current_load, adjusted_alighting + excess_load)
                print(f"    Night load reduction: Excess {excess_load} passengers, increasing alighting to {adjusted_alighting}")
        
        return adjusted_boarding, adjusted_alighting, time_multiplier
    
    def _apply_realistic_boarding_constraints(self, current_load, predicted_boarding, capacity, max_occupancy=None):
        """Apply realistic boarding constraints with occupancy cap (same as LSTM)"""
        if capacity <= 0:
            return 0
        
        # Generate random max occupancy between 1.5-1.6 if not provided
        if max_occupancy is None:
            max_occupancy = np.random.uniform(1.5, 1.6)
            
        current_occupancy = current_load / capacity
        
        # Hard cap: Stop all boarding if we've reached the random maximum occupancy
        if current_occupancy >= max_occupancy:
            if predicted_boarding > 0:
                print(f"    Maximum occupancy cap reached: {current_occupancy:.2f} >= {max_occupancy:.2f}, stopping all boarding")
            return 0
        
        if current_occupancy <= 0.7:
            boarding_multiplier = 1.0
            constraint_reason = "normal"
        elif current_occupancy <= 0.85:
            boarding_multiplier = 0.9
            constraint_reason = "slight_hesitation"
        elif current_occupancy <= 1.0:
            boarding_multiplier = 0.7
            constraint_reason = "near_capacity"
        elif current_occupancy <= 1.2:
            boarding_multiplier = 0.4
            constraint_reason = "driver_discretion"
        elif current_occupancy <= 1.4:
            boarding_multiplier = 0.15
            constraint_reason = "driver_stops_boarding"
        else:
            # Very overcrowded - minimal boarding allowed until max cap
            boarding_multiplier = 0.05
            constraint_reason = "minimal_boarding_only"
        
        constrained_boarding = max(0, round(predicted_boarding * boarding_multiplier))
        
        # Additional check: Don't allow boarding that would exceed max occupancy
        potential_new_load = current_load + constrained_boarding
        if potential_new_load / capacity > max_occupancy:
            # Reduce boarding to stay under max occupancy
            max_allowed_boarding = max(0, int((max_occupancy * capacity) - current_load))
            constrained_boarding = min(constrained_boarding, max_allowed_boarding)
            if constrained_boarding < predicted_boarding:
                print(f"    Occupancy cap limiting boarding: {predicted_boarding}→{constrained_boarding} to stay under {max_occupancy:.2f}")
        
        if boarding_multiplier < 0.8 and predicted_boarding > 0:
            reduction = predicted_boarding - constrained_boarding
            print(f"    Boarding constraint applied: {predicted_boarding}→{constrained_boarding} "
                  f"(occupancy: {current_occupancy:.2f}, max: {max_occupancy:.2f}, reason: {constraint_reason})")
        
        return constrained_boarding
    
    def _calculate_loads_with_realistic_and_night_constraints(self, df):
        """Calculate loads with all constraints (same as LSTM)"""
        print("Calculating loads with logical, last stop, realistic boarding, and night-time constraints (1-4 AM only)...")
        
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        
        logical_fixes = 0
        last_stop_fixes = 0
        boarding_constraints_applied = 0
        night_constraints_applied = 0
        total_boarding_reduction = 0
        total_night_reduction = 0
        
        df = df.sort_values(['bus_id', 'datetime']).reset_index(drop=True)
        
        # Store max occupancy per bus to maintain consistency
        bus_max_occupancy = {}
        
        for bus_id in df['bus_id'].unique():
            bus_data = df[df['bus_id'] == bus_id].copy()
            current_load = 0
            
            # Generate random max occupancy for this bus (between 1.5-1.6)
            bus_max_occupancy[bus_id] = np.random.uniform(1.5, 1.6)
            
            for idx in bus_data.index:
                capacity = df.loc[idx, 'capacity']
                predicted_boarding = df.loc[idx, 'boarding']
                predicted_alighting = df.loc[idx, 'alighting'] 
                line_id = df.loc[idx, 'line_id']
                stop_id = df.loc[idx, 'stop_id']
                hour = df.loc[idx, 'hour']
                
                df.loc[idx, 'current_load'] = current_load
                
                # APPLY LAST STOP LOGIC FIRST
                is_last = self._is_last_stop(line_id, stop_id)
                if is_last:
                    original_boarding = predicted_boarding
                    original_alighting = predicted_alighting
                    
                    boarding = 0
                    alighting = current_load
                    
                    if original_boarding != boarding or original_alighting != alighting:
                        print(f"    Last stop fix: Bus {bus_id} Line {line_id} Stop {stop_id} - "
                              f"boarding {original_boarding}→{boarding}, "
                              f"alighting {original_alighting}→{alighting}")
                        last_stop_fixes += 1
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                
                else:
                    # Apply night-time logic
                    night_adjusted_boarding, night_adjusted_alighting, time_multiplier = self._apply_night_time_logic(
                        current_load, predicted_boarding, predicted_alighting, hour, capacity
                    )
                    
                    if time_multiplier < 1.0:
                        night_constraints_applied += 1
                        total_night_reduction += (predicted_boarding - night_adjusted_boarding)
                    
                    # Apply realistic constraints
                    final_boarding = self._apply_realistic_boarding_constraints(
                        current_load, night_adjusted_boarding, capacity, bus_max_occupancy[bus_id]
                    )
                    
                    if final_boarding != night_adjusted_boarding:
                        boarding_constraints_applied += 1
                        total_boarding_reduction += (night_adjusted_boarding - final_boarding)
                    
                    boarding = final_boarding
                    alighting = night_adjusted_alighting
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                    
                    # Logical constraint
                    if alighting > current_load:
                        print(f"    Logical fix: Bus {bus_id} - "
                              f"alighting {alighting} > current_load {current_load}, "
                              f"setting alighting to {current_load}")
                        df.loc[idx, 'alighting'] = current_load
                        alighting = current_load
                        logical_fixes += 1
                
                # Calculate final load
                final_boarding = df.loc[idx, 'boarding']
                final_alighting = df.loc[idx, 'alighting']
                new_load = max(0, current_load + final_boarding - final_alighting)
                
                df.loc[idx, 'new_load'] = new_load
                df.loc[idx, 'occupancy_rate'] = new_load / capacity if capacity > 0 else 0
                
                current_load = new_load
        
        print(f"\nENHANCED CONSTRAINT APPLICATION SUMMARY:")
        print(f"  Last stop logic applied: {last_stop_fixes} records")
        print(f"  Logical constraints applied: {logical_fixes} records") 
        print(f"  Boarding constraints applied: {boarding_constraints_applied} records")
        print(f"  Night-time constraints applied (1-4 AM): {night_constraints_applied} records")
        print(f"  Total boarding reduction: {total_boarding_reduction} passengers")
        print(f"  Night-time reduction: {total_night_reduction:.0f} passengers")
        
        # Report statistics
        night_records = df[df['hour'].isin([1, 2, 3, 4])]
        if len(night_records) > 0:
            night_avg_load = night_records['new_load'].mean()
            night_max_load = night_records['new_load'].max()
            night_boarding_total = night_records['boarding'].sum()
            
            print(f"\nNIGHT-TIME STATISTICS (1-4 AM ONLY):")
            print(f"  Night records: {len(night_records)}")
            print(f"  Average night load: {night_avg_load:.1f} passengers")
            print(f"  Maximum night load: {night_max_load} passengers")
            print(f"  Total night boarding: {night_boarding_total} passengers")
            
            if night_avg_load <= 8 and night_max_load <= 20:
                print("  ✅ Night loads are realistic!")
            else:
                print("  ⚠️  Night loads may still be high")
        
        # Final statistics
        overcrowded = (df['occupancy_rate'] > 1.0).sum()
        severely_overcrowded = (df['occupancy_rate'] > 1.5).sum()
        max_occupancy = df['occupancy_rate'].max()
        avg_occupancy = df['occupancy_rate'].mean()
        
        print(f"\nFINAL OCCUPANCY STATISTICS WITH OCCUPANCY CAPS (1.5-1.6):")
        print(f"  >100% capacity: {overcrowded} records ({overcrowded/len(df)*100:.1f}%)")
        print(f"  >150% capacity: {severely_overcrowded} records ({severely_overcrowded/len(df)*100:.1f}%)")
        print(f"  Maximum occupancy: {max_occupancy:.2f} ({max_occupancy*100:.0f}%)")
        print(f"  Average occupancy: {avg_occupancy:.2f} ({avg_occupancy*100:.0f}%)")
        
        # Show occupancy cap effectiveness
        over_150_pct = (df['occupancy_rate'] > 1.5).sum()
        over_160_pct = (df['occupancy_rate'] > 1.6).sum()
        print(f"\nOCCUPANCY CAP EFFECTIVENESS:")
        print(f"  >150% capacity: {over_150_pct} records ({over_150_pct/len(df)*100:.1f}%)")
        print(f"  >160% capacity: {over_160_pct} records ({over_160_pct/len(df)*100:.1f}%)")
        if over_160_pct == 0:
            print("  ✅ Perfect occupancy cap enforcement!")
        elif over_160_pct < 10:
            print("  📊 Excellent occupancy cap enforcement")
        else:
            print("  ⚠️  Some buses still exceed 160% capacity")
        
        return df
    
    def predict_with_patterns(self, historical_df, future_df, apply_route_adjustments=True, apply_realistic_constraints=True, apply_night_constraints=True):
        """Make predictions with Prophet and apply same constraints as LSTM"""
        print("Making pattern-aware predictions with Prophet + constraints...")
        
        predictions_df = future_df.copy()
        predictions_df = self._ensure_stop_id_consistency(predictions_df)
        
        future_stop_ids = set(predictions_df['stop_id'].unique())
        print(f"Future data contains {len(future_stop_ids)} unique stops")
        
        # Initialize prediction columns
        predictions_df['boarding_predicted'] = 0
        predictions_df['alighting_predicted'] = 0
        
        last_stop_applications = 0
        rush_hour_boosts = 0
        total_rush_hour_increase = 0
        
        # Define rush hours
        rush_hours = [7, 8, 9, 17, 18, 19]
        
        # Process predictions
        print(f"\nProcessing {len(predictions_df)} prediction records...")
        
        # Prepare future data for Prophet
        prepared_future = self.prepare_prophet_data(predictions_df)
        
        for i, (idx, row) in enumerate(predictions_df.iterrows()):
            if i % 100 == 0:
                print(f"  Processing record {i+1}/{len(predictions_df)}")
                
            stop_id = row['stop_id']
            line_id = str(row['line_id'])
            hour = pd.to_datetime(row['datetime']).hour
            
            try:
                # Use Prophet if model exists for this stop
                if stop_id in self.models:
                    pred_boarding = self._predict_with_prophet(prepared_future.iloc[i], stop_id, 'boarding')
                    pred_alighting = self._predict_with_prophet(prepared_future.iloc[i], stop_id, 'alighting')
                else:
                    # Fallback to pattern-based prediction
                    pred_boarding, pred_alighting = self._predict_from_patterns(row, stop_id)
                
                # Apply rush hour boarding boost BEFORE last stop logic
                if hour in rush_hours and pred_boarding > 0:
                    original_boarding = pred_boarding
                    pred_boarding = round(pred_boarding * 1.2)
                    rush_hour_boosts += 1
                    total_rush_hour_increase += (pred_boarding - original_boarding)
                    if i < 10:  # Show first few examples
                        print(f"    Rush hour boost (Hour {hour}): Boarding {original_boarding}→{pred_boarding}")
                
                # Apply last stop logic AFTER rush hour boost
                if self._is_last_stop(line_id, stop_id):
                    original_boarding = pred_boarding
                    pred_boarding = 0
                    last_stop_applications += 1
                    print(f"    Last stop fix applied: Line {line_id}, Stop {stop_id}, Boarding {original_boarding}→0")
                
                predictions_df.loc[idx, 'boarding_predicted'] = pred_boarding
                predictions_df.loc[idx, 'alighting_predicted'] = pred_alighting
                predictions_df.loc[idx, 'boarding'] = pred_boarding
                predictions_df.loc[idx, 'alighting'] = pred_alighting
                
            except Exception as e:
                print(f"Error predicting for stop {stop_id}: {e}")
                pred_boarding, pred_alighting = self._predict_from_patterns(row)
                
                # Apply rush hour boost to fallback predictions too
                if hour in rush_hours and pred_boarding > 0:
                    original_boarding = pred_boarding
                    pred_boarding = round(pred_boarding * 1.2)
                    rush_hour_boosts += 1
                    total_rush_hour_increase += (pred_boarding - original_boarding)
                
                if self._is_last_stop(line_id, stop_id):
                    pred_boarding = 0
                    last_stop_applications += 1
                
                predictions_df.loc[idx, 'boarding_predicted'] = pred_boarding
                predictions_df.loc[idx, 'alighting_predicted'] = pred_alighting
                predictions_df.loc[idx, 'boarding'] = pred_boarding
                predictions_df.loc[idx, 'alighting'] = pred_alighting
        
        print(f"\nRush hour boarding boost applied to {rush_hour_boosts} records")
        print(f"Total rush hour boarding increase: {total_rush_hour_increase} passengers")
        print(f"Last stop logic applied to {last_stop_applications} records")
        
        # Apply constraints (same as LSTM)
        if apply_realistic_constraints and apply_night_constraints:
            print("\n=== APPLYING ENHANCED REALISTIC CONSTRAINTS WITH NIGHT-TIME LOGIC ===")
            predictions_df = self._calculate_loads_with_realistic_and_night_constraints(predictions_df)
        
        print(f"Completed Prophet predictions with constraints for {len(predictions_df)} records")
        return predictions_df
    
    def _predict_with_prophet(self, future_row, stop_id, target):
        """Make single prediction using Prophet model"""
        if stop_id not in self.models or target not in self.models[stop_id]:
            return None
        
        model = self.models[stop_id][target]
        
        # Create future dataframe for Prophet
        future_df = pd.DataFrame({
            'ds': [future_row['datetime']],
            'hour': [future_row['hour']],
            'is_weekend': [future_row['is_weekend']],
            'is_peak_hour': [future_row['is_peak_hour']],
            'is_morning_peak': [future_row['is_morning_peak']],
            'is_evening_peak': [future_row['is_evening_peak']],
            'is_night_time': [future_row['is_night_time']],
            'is_last_stop': [future_row['is_last_stop']]
        })
        
        # Make prediction
        forecast = model.predict(future_df)
        prediction = max(0, round(forecast['yhat'].iloc[0]))
        
        return prediction
    
    def generate_conservation_report(self, original_df, fixed_df):
        """Generate comprehensive report (same as LSTM)"""
        print("\n" + "="*60)
        print("COMPREHENSIVE PROPHET FIX REPORT WITH CONSTRAINTS")
        print("="*60)
        
        orig_boarding = original_df['boarding'].sum()
        orig_alighting = original_df['alighting'].sum()
        orig_imbalance = orig_boarding - orig_alighting
        
        fixed_boarding = fixed_df['boarding'].sum()
        fixed_alighting = fixed_df['alighting'].sum() 
        fixed_imbalance = fixed_boarding - fixed_alighting
        
        print(f"\n🚌 PASSENGER TOTALS:")
        print(f"  Original boarding: {orig_boarding:,.0f}")
        print(f"  Fixed boarding:    {fixed_boarding:,.0f}")
        print(f"  Original alighting: {orig_alighting:,.0f}")
        print(f"  Fixed alighting:    {fixed_alighting:,.0f}")
        print(f"  Original imbalance: {orig_imbalance:,.0f}")
        print(f"  Final imbalance:    {fixed_imbalance:,.0f}")
        
        # Last stop analysis
        if self.last_stops:
            orig_last_stop_boarding = 0
            fixed_last_stop_boarding = 0
            last_stop_count = 0
            
            for idx, row in original_df.iterrows():
                if self._is_last_stop(row['line_id'], row['stop_id']):
                    orig_last_stop_boarding += row['boarding']
                    last_stop_count += 1
            
            for idx, row in fixed_df.iterrows():
                if self._is_last_stop(row['line_id'], row['stop_id']):
                    fixed_last_stop_boarding += row['boarding']
            
            print(f"\n🏁 LAST STOP LOGIC:")
            print(f"  Last stop records: {last_stop_count}")
            print(f"  Original boarding at last stops: {orig_last_stop_boarding}")
            print(f"  Fixed boarding at last stops: {fixed_last_stop_boarding}")
            print(f"  Status: {'✅ PASS' if fixed_last_stop_boarding == 0 else '⚠️  NEEDS ATTENTION'}")
        
        # Overcrowding comparison
        orig_overcrowded = (original_df['occupancy_rate'] > 1.0).sum()
        fixed_overcrowded = (fixed_df['occupancy_rate'] > 1.0).sum()
        
        print(f"\n🚌 OVERCROWDING ANALYSIS:")
        print(f"  Original >100%: {orig_overcrowded} ({orig_overcrowded/len(original_df)*100:.1f}%)")
        print(f"  Fixed >100%:    {fixed_overcrowded} ({fixed_overcrowded/len(fixed_df)*100:.1f}%)")
        print(f"  Improvement:    {orig_overcrowded - fixed_overcrowded} records")


def compare_with_historical(historical_df, predictions_df, save_dir=None):
    """Compare predictions with historical data (same as LSTM)"""
    hist_temp = historical_df.copy()
    hist_temp['datetime'] = pd.to_datetime(hist_temp['datetime'])
    hist_temp['hour'] = hist_temp['datetime'].dt.hour
    hist_temp['is_weekend'] = hist_temp['datetime'].dt.dayofweek >= 5
    
    pred_temp = predictions_df.copy()
    pred_temp['datetime'] = pd.to_datetime(pred_temp['datetime'])
    pred_temp['hour'] = pred_temp['datetime'].dt.hour
    pred_temp['is_weekend'] = pred_temp['datetime'].dt.dayofweek >= 5
    
    # Weekday comparison
    hist_weekday = hist_temp[~hist_temp['is_weekend']].groupby('hour')[['boarding', 'alighting']].mean()
    pred_weekday = pred_temp[~pred_temp['is_weekend']].groupby('hour')[['boarding', 'alighting']].mean()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Boarding comparison
    hours = range(24)
    hist_boarding = [hist_weekday.loc[h, 'boarding'] if h in hist_weekday.index else 0 for h in hours]
    pred_boarding = [pred_weekday.loc[h, 'boarding'] if h in pred_weekday.index else 0 for h in hours]
    
    ax1.plot(hours, hist_boarding, 'b-', linewidth=2, label='Historical', marker='o')
    ax1.plot(hours, pred_boarding, 'r--', linewidth=2, label='Prophet Predicted', marker='s')
    ax1.fill_between([7, 9], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='yellow', label='Morning Rush')
    ax1.fill_between([17, 19], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='orange', label='Evening Rush')
    ax1.fill_between([1, 4], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='purple', label='Night Constraints')
    ax1.set_title('Weekday Boarding Patterns Comparison (Prophet)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Average Passengers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Alighting comparison
    hist_alighting = [hist_weekday.loc[h, 'alighting'] if h in hist_weekday.index else 0 for h in hours]
    pred_alighting = [pred_weekday.loc[h, 'alighting'] if h in pred_weekday.index else 0 for h in hours]
    
    ax2.plot(hours, hist_alighting, 'b-', linewidth=2, label='Historical', marker='o')
    ax2.plot(hours, pred_alighting, 'r--', linewidth=2, label='Prophet Predicted', marker='s')
    ax2.fill_between([7, 9], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='yellow')
    ax2.fill_between([17, 19], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='orange')
    ax2.fill_between([1, 4], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='purple')
    ax2.set_title('Weekday Alighting Patterns Comparison (Prophet)')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Average Passengers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot - correlation
    ax3.scatter(hist_boarding, pred_boarding, alpha=0.6, label='Boarding')
    ax3.scatter(hist_alighting, pred_alighting, alpha=0.6, label='Alighting')
    max_val = max(max(hist_boarding), max(pred_boarding), max(hist_alighting), max(pred_alighting))
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Match')
    ax3.set_xlabel('Historical Average')
    ax3.set_ylabel('Prophet Predicted Average')
    ax3.set_title('Pattern Correlation (Prophet)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax4.hist(historical_df['boarding'], bins=30, alpha=0.5, label='Historical Boarding', color='blue')
    ax4.hist(predictions_df['boarding'], bins=30, alpha=0.5, label='Prophet Predicted Boarding', color='red')
    ax4.set_xlabel('Passenger Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution Comparison (Prophet)')
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'historical_vs_prophet_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Calculate correlation
    correlation_boarding = np.corrcoef(hist_boarding, pred_boarding)[0, 1] if len(hist_boarding) > 1 else 0
    correlation_alighting = np.corrcoef(hist_alighting, pred_alighting)[0, 1] if len(hist_alighting) > 1 else 0
    
    print(f"\nProphet Pattern Correlation:")
    print(f"  Boarding: {correlation_boarding:.3f}")
    print(f"  Alighting: {correlation_alighting:.3f}")


def main():
    """Main function - same structure as LSTM"""
    parser = argparse.ArgumentParser(description='Prophet Model with LSTM-like Constraints')
    parser.add_argument('--historical', required=True, help='Historical data CSV')
    parser.add_argument('--future', help='Future data CSV')
    parser.add_argument('--bus-stops', default='ankara_bus_stops.csv', help='Bus stops CSV file')
    parser.add_argument('--output', default='output_prophet_constraints', help='Output directory')
    parser.add_argument('--sequence', type=int, default=48, help='Sequence length (not used in Prophet but kept for compatibility)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs (not used in Prophet but kept for compatibility)')
    parser.add_argument('--disable-route-adjustments', action='store_true', 
                       help='Disable route-aware adjustments')
    parser.add_argument('--disable-realistic-constraints', action='store_true',
                       help='Disable realistic boarding constraints')
    parser.add_argument('--disable-night-constraints', action='store_true',
                       help='Disable night-time constraints (no 1-4 AM logic)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    plots_dir = os.path.join(args.output, 'plots')
    data_dir = os.path.join(args.output, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Load data
        print(f"Loading historical data from {args.historical}")
        historical_df = pd.read_csv(args.historical)
        print(f"Loaded {len(historical_df)} historical records")
        
        # Create model
        model = ProphetModel()
        
        # Load bus stops data
        if args.bus_stops:
            model._load_last_stops(args.bus_stops)
        else:
            print("Warning: No bus stops CSV provided. Last stop logic will be disabled.")
        
        # Analyze historical patterns
        model.analyze_historical_patterns(historical_df)
        
        # Train Prophet models
        model.train_prophet_models(historical_df)
        
        if model.is_trained:
            # Make predictions if future data provided
            if args.future:
                print(f"\nLoading future data from {args.future}")
                future_df = pd.read_csv(args.future)
                
                original_future_df = future_df.copy()
                
                # Apply constraints based on flags
                use_route_adjustments = not args.disable_route_adjustments
                use_realistic_constraints = not args.disable_realistic_constraints
                use_night_constraints = not args.disable_night_constraints
                
                print(f"Configuration:")
                print(f"  Route adjustments: {'ON' if use_route_adjustments else 'OFF'}")
                print(f"  Realistic constraints: {'ON' if use_realistic_constraints else 'OFF'}")
                print(f"  Night-time constraints (1-4 AM): {'ON' if use_night_constraints else 'OFF'}")
                print(f"  Occupancy caps: Random 1.5-1.6 per bus (150-160% capacity max)")
                print(f"  Rush hour boost: 1.2x boarding during hours 7-9 AM, 5-7 PM")
                
                predictions_df = model.predict_with_patterns(
                    historical_df, 
                    future_df, 
                    apply_route_adjustments=use_route_adjustments,
                    apply_realistic_constraints=use_realistic_constraints,
                    apply_night_constraints=use_night_constraints
                )
                
                # Generate report
                model.generate_conservation_report(original_future_df, predictions_df)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(data_dir, f"prophet_predictions_{timestamp}.csv")
                predictions_df.to_csv(output_file, index=False)
                print(f"\nProphet predictions saved to {output_file}")
                
                # Compare with historical patterns
                compare_with_historical(historical_df, predictions_df, plots_dir)
                
                print("Prophet model with LSTM-like constraints completed successfully!")
            
        else:
            print("Failed to train Prophet models")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()