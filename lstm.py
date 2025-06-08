#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTM:
    """
    LSTM model that preserves and validates against historical patterns
    with route-aware fixes, last stop logic, realistic boarding constraints, 
    and night-time logic (1-4 AM only)
    """
    
    def __init__(self, sequence_length=48):
        self.model = None
        self.sequence_length = sequence_length
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
        # Pattern preservation data
        self.historical_patterns = {}
        self.stop_patterns = {}
        self.hourly_patterns = {}
        self.feature_columns = []
        self.stop_mapping = {}
        self.training_stop_ids = set()  # Track training stop IDs
        
        # Route and conservation data
        self.route_patterns = {}
        self.stop_classifications = {}
        self.conservation_stats = {}
        
        # Last stop data
        self.last_stops = {}  # line_id -> last_stop_id mapping
        self.bus_stops_df = None
        
    def _load_last_stops(self, bus_stops_csv_path):
        """Load bus stops CSV and identify last stops for each line"""
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
            
            # Group by line and find last stop (assuming stops are in order in CSV)
            last_stops = {}
            unique_lines = sorted(self.bus_stops_df['line_id'].unique())
            
            print(f"\nProcessing {len(unique_lines)} unique lines...")
            
            for line_id in unique_lines:
                line_stops = self.bus_stops_df[self.bus_stops_df['line_id'] == line_id].copy()
                line_stops = line_stops.reset_index(drop=True)
                
                if len(line_stops) > 0:
                    # Last stop is the last one in the CSV for this line
                    last_stop_id = int(line_stops.iloc[-1]['stop_id'])
                    last_stop_name = line_stops.iloc[-1]['stop_name']
                    last_stops[str(line_id)] = last_stop_id
                    
                    print(f"  Line {line_id}: {len(line_stops)} stops, last stop = {last_stop_id} ({last_stop_name})")
            
            self.last_stops = last_stops
            print(f"\n=== DETECTED LAST STOPS FOR ALL LINES ===")
            
            # Show ALL detected last stops for debugging
            for line_id in sorted(self.last_stops.keys()):
                stop_id = self.last_stops[line_id]
                # Find stop name
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
        """Check if a given stop is the last stop for its line"""
        if not self.last_stops:
            return False
        
        # Ensure consistent string matching
        line_key = str(line_id)
        stop_id_int = int(stop_id)
        
        if line_key in self.last_stops:
            is_last = stop_id_int == self.last_stops[line_key]
            # Debug output for verification
            if is_last:
                print(f"    ✓ Last stop detected: Line '{line_key}' Stop {stop_id_int}")
            return is_last
        else:
            # Debug: show when line is not found
            if line_key not in getattr(self, '_missing_lines_shown', set()):
                print(f"    ⚠️  Line '{line_key}' not found in last stops mapping")
                if not hasattr(self, '_missing_lines_shown'):
                    self._missing_lines_shown = set()
                self._missing_lines_shown.add(line_key)
        
        return False
        
    def _ensure_stop_id_consistency(self, df):
        """Ensure stop_id and line_id are consistent data types"""
        # Convert stop_id to int and line_id to string to ensure consistency
        df = df.copy()
        df['stop_id'] = df['stop_id'].astype(int)
        df['line_id'] = df['line_id'].astype(str)
        return df
        
    def analyze_historical_patterns(self, df):
        """Comprehensive analysis of historical patterns"""
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
            if len(stop_weekday) > 24:  # Enough data for reliable patterns
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
        
        # Analyze route patterns for conservation fixes
        self._analyze_route_patterns(df_temp)
        
        # Analyze last stop patterns if data available
        if self.last_stops:
            self._analyze_last_stop_patterns(df_temp)
    
    def _analyze_last_stop_patterns(self, df):
        """Analyze patterns at last stops"""
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
        """Analyze route patterns to understand stop positions and flow"""
        print("\n=== ANALYZING ROUTE PATTERNS FOR CONSERVATION ===")
        
        df = df.copy()
        df = df.sort_values(['line_id', 'bus_id', 'datetime']).reset_index(drop=True)
        
        route_analysis = {}
        
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id].copy()
            
            # Analyze stop sequences per bus
            stop_sequences = []
            for bus_id in line_data['bus_id'].unique():
                bus_data = line_data[line_data['bus_id'] == bus_id].sort_values('datetime')
                stop_sequence = bus_data['stop_id'].tolist()
                if len(stop_sequence) > 3:  # Only meaningful sequences
                    stop_sequences.append(stop_sequence)
            
            if stop_sequences:
                # Find common stop patterns
                all_stops = []
                for seq in stop_sequences:
                    all_stops.extend(seq)
                
                # Get unique stops and their frequency positions
                unique_stops = list(set(all_stops))
                stop_positions = {}
                
                for stop in unique_stops:
                    positions = []
                    for seq in stop_sequences:
                        if stop in seq:
                            # Position as percentage of route completion
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
        
        # Classify stops by route position
        self._classify_stops_by_position()
        
        print(f"Analyzed {len(route_analysis)} routes for conservation")
    
    def _classify_stops_by_position(self):
        """Classify stops as ORIGIN, DESTINATION, or MIDDLE based on route position"""
        
        for line_id, route_data in self.route_patterns.items():
            for stop_id, pos_data in route_data['stop_positions'].items():
                avg_pos = pos_data['avg_position']
                
                # Classify based on position in route
                if avg_pos < 0.3:
                    classification = 'ORIGIN'  # Early in route
                elif avg_pos > 0.7:
                    classification = 'DESTINATION'  # Late in route
                else:
                    classification = 'MIDDLE'  # Middle of route
                
                # Store classification
                if stop_id not in self.stop_classifications:
                    self.stop_classifications[stop_id] = {}
                
                self.stop_classifications[stop_id][line_id] = {
                    'type': classification,
                    'position': avg_pos,
                    'confidence': 1.0 / (pos_data['position_std'] + 0.1)  # Higher confidence for consistent positions
                }
        
        # Aggregate classifications across lines for each stop
        final_classifications = {}
        for stop_id, line_data in self.stop_classifications.items():
            # Weight by confidence and frequency
            type_scores = {'ORIGIN': 0, 'MIDDLE': 0, 'DESTINATION': 0}
            
            for line_id, class_data in line_data.items():
                stop_type = class_data['type']
                confidence = class_data['confidence']
                type_scores[stop_type] += confidence
            
            # Choose type with highest score
            final_type = max(type_scores, key=type_scores.get)
            final_classifications[stop_id] = {
                'primary_type': final_type,
                'confidence': type_scores[final_type] / sum(type_scores.values()),
                'scores': type_scores
            }
        
        self.stop_classifications = final_classifications
        
        # Print classification summary
        type_counts = {}
        for stop_data in self.stop_classifications.values():
            stop_type = stop_data['primary_type']
            type_counts[stop_type] = type_counts.get(stop_type, 0) + 1
        
        print(f"Stop classifications: {type_counts}")
    
    def prepare_enhanced_features(self, df, create_mapping=True):
        """Create features that preserve patterns"""
        data = self._ensure_stop_id_consistency(df)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values(['stop_id', 'datetime']).reset_index(drop=True)
        
        # Basic time features
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Enhanced cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Last stop indicator
        data['is_last_stop'] = 0
        if self.last_stops:
            for idx, row in data.iterrows():
                if self._is_last_stop(row['line_id'], row['stop_id']):
                    data.loc[idx, 'is_last_stop'] = 1
        
        # Rush hour indicators based on historical analysis
        if len(self.hourly_patterns) > 0 and 'weekday' in self.hourly_patterns:
            weekday_boarding = self.hourly_patterns['weekday']['boarding']['mean']
            if len(weekday_boarding) > 0:
                peak_threshold = weekday_boarding.quantile(0.7)
                peak_hours = weekday_boarding[weekday_boarding >= peak_threshold].index.tolist()
            else:
                peak_hours = [7, 8, 9, 17, 18, 19]  # Default
        else:
            peak_hours = [7, 8, 9, 17, 18, 19]  # Default
        
        data['is_peak_hour'] = data['hour'].isin(peak_hours).astype(int)
        data['is_morning_peak'] = data['hour'].isin([7, 8, 9]).astype(int)
        data['is_evening_peak'] = data['hour'].isin([17, 18, 19]).astype(int)
        
        # Stop encoding - ONLY create new mapping during training
        if create_mapping:
            unique_stops = sorted(data['stop_id'].unique())
            self.stop_mapping = {stop: idx for idx, stop in enumerate(unique_stops)}
            print(f"Created stop mapping for {len(unique_stops)} stops")
        
        # Apply stop encoding (use existing mapping for prediction)
        data['stop_encoded'] = data['stop_id'].map(self.stop_mapping)
        # For unknown stops, use a default encoding
        max_encoded = max(self.stop_mapping.values()) if self.stop_mapping else 0
        data['stop_encoded'] = data['stop_encoded'].fillna(max_encoded + 1)
        data['stop_encoded'] = data['stop_encoded'] / (max_encoded + 2)  # Normalize
        
        # Historical pattern features
        data['hour_historical_boarding'] = 0.0
        data['hour_historical_alighting'] = 0.0
        data['stop_relative_boarding'] = 1.0
        data['stop_relative_alighting'] = 1.0
        
        # Add historical pattern information
        for idx, row in data.iterrows():
            hour = row['hour']
            stop_id = row['stop_id']
            is_weekend = row['is_weekend']
            
            # Get historical hourly average
            pattern_key = 'weekend' if is_weekend and self.hourly_patterns.get('weekend') is not None else 'weekday'
            if pattern_key in self.hourly_patterns and hour in self.hourly_patterns[pattern_key].index:
                data.loc[idx, 'hour_historical_boarding'] = self.hourly_patterns[pattern_key].loc[hour, ('boarding', 'mean')]
                data.loc[idx, 'hour_historical_alighting'] = self.hourly_patterns[pattern_key].loc[hour, ('alighting', 'mean')]
            
            # Get stop-specific relative patterns
            if stop_id in self.stop_patterns:
                stop_pattern = self.stop_patterns[stop_id]
                daily_mean_boarding = stop_pattern['daily_mean']['boarding']
                daily_mean_alighting = stop_pattern['daily_mean']['alighting']
                
                if 'boarding' in row and pd.notna(row['boarding']):
                    if daily_mean_boarding > 0:
                        data.loc[idx, 'stop_relative_boarding'] = row['boarding'] / daily_mean_boarding
                if 'alighting' in row and pd.notna(row['alighting']):
                    if daily_mean_alighting > 0:
                        data.loc[idx, 'stop_relative_alighting'] = row['alighting'] / daily_mean_alighting
        
        # Lag features with pattern context (only if boarding/alighting exist)
        if 'boarding' in data.columns and 'alighting' in data.columns:
            for col in ['boarding', 'alighting']:
                # Recent lags
                data[f'{col}_lag_1'] = data.groupby('stop_id')[col].shift(1)
                data[f'{col}_lag_6'] = data.groupby('stop_id')[col].shift(6)
                data[f'{col}_lag_24'] = data.groupby('stop_id')[col].shift(24)
                
                # Pattern-based lags (same hour yesterday)
                data[f'{col}_same_hour_yesterday'] = data.groupby('stop_id')[col].shift(24)
                
                # Fill NaN with pattern-based defaults
                for lag_col in [f'{col}_lag_1', f'{col}_lag_6', f'{col}_lag_24', f'{col}_same_hour_yesterday']:
                    # Use stop-specific means first
                    stop_means = data.groupby('stop_id')[col].transform('mean')
                    data[lag_col] = data[lag_col].fillna(stop_means)
                    
                    # Then global means
                    data[lag_col] = data[lag_col].fillna(data[col].mean())
        else:
            # For prediction data without targets, initialize with zeros
            for col in ['boarding', 'alighting']:
                for lag_suffix in ['_lag_1', '_lag_6', '_lag_24', '_same_hour_yesterday']:
                    data[col + lag_suffix] = 0.0
        
        # Define feature columns (including last stop indicator)
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_peak_hour', 'is_morning_peak', 'is_evening_peak',
            'is_last_stop',  # NEW: Last stop indicator
            'stop_encoded',
            'hour_historical_boarding', 'hour_historical_alighting',
            'stop_relative_boarding', 'stop_relative_alighting',
            'boarding_lag_1', 'alighting_lag_1',
            'boarding_lag_6', 'alighting_lag_6', 
            'boarding_lag_24', 'alighting_lag_24',
            'boarding_same_hour_yesterday', 'alighting_same_hour_yesterday'
        ]
        
        print(f"Created {len(self.feature_columns)} pattern-preserving features (including last stop logic)")
        
        # Clean data
        for col in self.feature_columns:
            if col in data.columns:
                data[col] = data[col].fillna(0)
                data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def create_pattern_sequences(self, data, validation_split=0.2):
        """Create sequences that preserve temporal patterns"""
        print("Creating pattern-preserving sequences...")
        
        # Prepare data
        feature_data = data[self.feature_columns].values
        target_data = data[['boarding', 'alighting']].values
        
        # Fit scalers
        self.feature_scaler.fit(feature_data)
        self.target_scaler.fit(target_data)
        
        # Scale data
        feature_data_scaled = self.feature_scaler.transform(feature_data)
        target_data_scaled = self.target_scaler.transform(target_data)
        
        # Create sequences with emphasis on pattern preservation
        X, y, metadata = [], [], []
        
        for stop_id, stop_data in data.groupby('stop_id'):
            if len(stop_data) < self.sequence_length + 1:
                continue
            
            stop_data = stop_data.sort_values('datetime').reset_index(drop=True)
            stop_indices = stop_data.index
            stop_features = feature_data_scaled[stop_indices]
            stop_targets = target_data_scaled[stop_indices]
            
            # Create overlapping sequences for better pattern learning
            step_size = max(1, self.sequence_length // 4)  # More overlap for better patterns
            
            for i in range(0, len(stop_data) - self.sequence_length, step_size):
                seq_x = stop_features[i:i + self.sequence_length]
                seq_y = stop_targets[i + self.sequence_length]
                
                if not (np.isnan(seq_x).any() or np.isnan(seq_y).any()):
                    X.append(seq_x)
                    y.append(seq_y)
                    metadata.append({
                        'stop_id': stop_id,
                        'datetime': stop_data['datetime'].iloc[i + self.sequence_length],
                        'hour': stop_data['hour'].iloc[i + self.sequence_length]
                    })
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences with pattern preservation")
        
        # Split temporally
        split_idx = int(len(X) * (1 - validation_split))
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], metadata
    
    def build_pattern_aware_model(self, input_shape, output_dim):
        """Build model that can learn and preserve patterns"""
        model = Sequential([
            # First LSTM layer - capture short-term patterns
            LSTM(80, return_sequences=True, input_shape=input_shape,
                 dropout=0.15, recurrent_dropout=0.15),
            BatchNormalization(),
            
            # Second LSTM layer - capture long-term patterns  
            LSTM(40, return_sequences=False,
                 dropout=0.15, recurrent_dropout=0.15),
            BatchNormalization(),
            
            # Dense layers for pattern integration
            Dense(64, activation='relu'),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.15),
            
            Dense(16, activation='relu'),
            
            # Output layer
            Dense(output_dim, activation='sigmoid')
        ])
        
        # Optimizer with appropriate clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_with_pattern_validation(self, train_X, train_y, val_X, val_y, epochs=60):
        """Train model with pattern validation"""
        print("Training pattern-preserving LSTM...")
        
        self.model = self.build_pattern_aware_model(
            input_shape=(train_X.shape[1], train_X.shape[2]),
            output_dim=train_y.shape[1]
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=32,
            shuffle=False,  # Preserve temporal order
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Validate against historical patterns
        self.validate_model_patterns(val_X, val_y)
        
        return history.history
    
    def validate_model_patterns(self, val_X, val_y):
        """Validate that model learned historical patterns"""
        print("\n=== PATTERN VALIDATION ===")
        
        # Make predictions on validation set
        val_pred_scaled = self.model.predict(val_X, verbose=0)
        val_pred = self.target_scaler.inverse_transform(val_pred_scaled)
        val_true = self.target_scaler.inverse_transform(val_y)
        
        # Calculate pattern metrics
        pred_boarding = val_pred[:, 0]
        true_boarding = val_true[:, 0]
        
        # Overall correlation
        correlation = np.corrcoef(pred_boarding, true_boarding)[0, 1]
        print(f"Overall correlation: {correlation:.3f}")
        
        # Pattern preservation metrics
        pred_mean = pred_boarding.mean()
        true_mean = true_boarding.mean()
        pred_std = pred_boarding.std()
        true_std = true_boarding.std()
        
        print(f"Mean preservation: True={true_mean:.2f}, Pred={pred_mean:.2f}, Ratio={pred_mean/true_mean:.3f}")
        print(f"Std preservation: True={true_std:.2f}, Pred={pred_std:.2f}, Ratio={pred_std/true_std:.3f}")
        
        if correlation < 0.5:
            print("⚠️  Warning: Low correlation with historical data!")
        elif correlation > 0.7:
            print("✅ Good pattern preservation!")
        else:
            print("📊 Moderate pattern preservation")
    
    def _get_night_time_multiplier(self, hour):
        """
        Get realistic night-time multipliers based on actual transit patterns
        """
        if hour == 1:
            return 0.3    # 70% reduction - still some nightlife activity
        elif hour == 2:
            return 0.2    # 80% reduction - post-bar closure
        elif hour in [3, 4]:
            return 0.1   # 85% reduction - minimal essential travel
        else:
            return 1.0    # Normal ridership
    
    def _apply_night_time_logic(self, current_load, predicted_boarding, predicted_alighting, hour, capacity):
        """
        Apply night-time specific logic ONLY for 1-4 AM hours
        """
        
        time_multiplier = self._get_night_time_multiplier(hour)
        
        # Apply time-based reduction to predictions
        adjusted_boarding = max(0, round(predicted_boarding * time_multiplier))
        adjusted_alighting = max(0, round(predicted_alighting * time_multiplier))
        
        # Night-time specific constraints ONLY for 1-4 AM
        if 2 <= hour <= 4:
            print(f"    Applying night-time constraints for hour {hour}")
            
            # 1. Extremely low boarding at night
            if adjusted_boarding > 5:
                adjusted_boarding = np.random.choice([2, 3, 4]) # Random realistic night boarding
             #   print(f"    Night boarding cap: reduced to {adjusted_boarding}")
            
            # 2. Prevent unrealistic load accumulation
            max_reasonable_night_load = 15  # Busiest night route max
            if current_load > max_reasonable_night_load:
                # Force higher alighting to reduce unrealistic loads
                excess_load = current_load - max_reasonable_night_load
                adjusted_alighting = min(current_load, adjusted_alighting + excess_load)
                print(f"    Night load reduction: Excess {excess_load} passengers, increasing alighting to {adjusted_alighting}")
            
        
        return adjusted_boarding, adjusted_alighting, time_multiplier
    
    def _apply_realistic_boarding_constraints(self, current_load, predicted_boarding, capacity):
        """
        Apply realistic boarding constraints based on current occupancy
        
        Models:
        1. Driver discretion - stops allowing boarding when visibly full
        2. Passenger behavior - people wait for next bus when crowded
        3. Physical space limitations
        """
        
        if capacity <= 0:
            return 0
            
        current_occupancy = current_load / capacity
        
        # Define occupancy thresholds and corresponding boarding multipliers
        if current_occupancy <= 0.7:
            # Comfortable - normal boarding
            boarding_multiplier = 1.0
            constraint_reason = "normal"
            
        elif current_occupancy <= 0.85:
            # Getting crowded - slight passenger hesitation
            boarding_multiplier = 0.9
            constraint_reason = "slight_hesitation"
            
        elif current_occupancy <= 1.0:
            # Near capacity - significant passenger hesitation + driver awareness
            boarding_multiplier = 0.7
            constraint_reason = "near_capacity"
            
        elif current_occupancy <= 1.2:
            # Over capacity but still possible - major hesitation + driver discretion
            boarding_multiplier = 0.4
            constraint_reason = "driver_discretion"
            
        elif current_occupancy <= 1.4:
            # Severely overcrowded - driver likely stops most boarding
            boarding_multiplier = 0.15
            constraint_reason = "driver_stops_boarding"
            
        else:
            # Impossibly overcrowded - driver stops all boarding
            boarding_multiplier = 0.0
            constraint_reason = "no_boarding_allowed"
        
        # Apply the multiplier
        constrained_boarding = max(0, round(predicted_boarding * boarding_multiplier))
        
        # Log significant reductions for analysis
        if boarding_multiplier < 0.8 and predicted_boarding > 0:
            reduction = predicted_boarding - constrained_boarding
            print(f"    Boarding constraint applied: {predicted_boarding}→{constrained_boarding} "
                  f"(occupancy: {current_occupancy:.2f}, reason: {constraint_reason})")
        
        return constrained_boarding
    
    def _calculate_loads_with_realistic_and_night_constraints(self, df):
        """Calculate loads with logical, last stop, realistic boarding, AND night-time constraints (1-4 AM only)"""
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
        
        # Sort by bus and time to process sequentially
        df = df.sort_values(['bus_id', 'datetime']).reset_index(drop=True)
        
        for bus_id in df['bus_id'].unique():
            bus_data = df[df['bus_id'] == bus_id].copy()
            current_load = 0
            
            for idx in bus_data.index:
                capacity = df.loc[idx, 'capacity']
                predicted_boarding = df.loc[idx, 'boarding']
                predicted_alighting = df.loc[idx, 'alighting'] 
                line_id = df.loc[idx, 'line_id']
                stop_id = df.loc[idx, 'stop_id']
                hour = df.loc[idx, 'hour']
                
                # Set current load before any adjustments
                df.loc[idx, 'current_load'] = current_load
                
                # APPLY LAST STOP LOGIC FIRST
                is_last = self._is_last_stop(line_id, stop_id)
                if is_last:
                    # At last stop: boarding = 0, alighting = current_load
                    original_boarding = predicted_boarding
                    original_alighting = predicted_alighting
                    
                    boarding = 0  # No one boards at last stop
                    alighting = current_load  # Everyone gets off
                    
                    if original_boarding != boarding or original_alighting != alighting:
                        print(f"    Last stop fix: Bus {bus_id} Line {line_id} Stop {stop_id} - "
                              f"boarding {original_boarding}→{boarding}, "
                              f"alighting {original_alighting}→{alighting}")
                        last_stop_fixes += 1
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                
                else:
                    # Step 1: Apply night-time logic first (1-4 AM only)
                    night_adjusted_boarding, night_adjusted_alighting, time_multiplier = self._apply_night_time_logic(
                        current_load, predicted_boarding, predicted_alighting, hour, capacity
                    )
                    
                    # Track night constraint applications
                    if time_multiplier < 1.0:
                        night_constraints_applied += 1
                        total_night_reduction += (predicted_boarding - night_adjusted_boarding)
                    
                    # Step 2: Apply occupancy-based realistic constraints
                    final_boarding = self._apply_realistic_boarding_constraints(
                        current_load, night_adjusted_boarding, capacity
                    )
                    
                    # Track boarding constraint applications
                    if final_boarding != night_adjusted_boarding:
                        boarding_constraints_applied += 1
                        total_boarding_reduction += (night_adjusted_boarding - final_boarding)
                    
                    boarding = final_boarding
                    alighting = night_adjusted_alighting
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                    
                    # LOGICAL CONSTRAINT: alighting cannot exceed current load
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
                
                # Update current load for next iteration
                current_load = new_load
        
        print(f"\nENHANCED CONSTRAINT APPLICATION SUMMARY:")
        print(f"  Last stop logic applied: {last_stop_fixes} records")
        print(f"  Logical constraints applied: {logical_fixes} records") 
        print(f"  Boarding constraints applied: {boarding_constraints_applied} records")
        print(f"  Night-time constraints applied (1-4 AM): {night_constraints_applied} records")
        print(f"  Total boarding reduction: {total_boarding_reduction} passengers")
        print(f"  Night-time reduction: {total_night_reduction:.0f} passengers")
        
        # Report night-specific statistics (1-4 AM only)
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
        else:
            print("\nNo night-time records (1-4 AM) found in data")
        
        # Report final realistic statistics
        impossible_alighting = sum(1 for idx, row in df.iterrows() 
                                  if row['alighting'] > row['current_load'] and row['current_load'] >= 0)
        
        overcrowded = (df['occupancy_rate'] > 1.0).sum()
        severely_overcrowded = (df['occupancy_rate'] > 1.5).sum()
        max_occupancy = df['occupancy_rate'].max()
        avg_occupancy = df['occupancy_rate'].mean()
        
        print(f"\nFINAL OCCUPANCY STATISTICS WITH NIGHT-TIME AWARENESS:")
        print(f"  >100% capacity: {overcrowded} records ({overcrowded/len(df)*100:.1f}%)")
        print(f"  >150% capacity: {severely_overcrowded} records ({severely_overcrowded/len(df)*100:.1f}%)")
        print(f"  Maximum occupancy: {max_occupancy:.2f} ({max_occupancy*100:.0f}%)")
        print(f"  Average occupancy: {avg_occupancy:.2f} ({avg_occupancy*100:.0f}%)")
        print(f"  Impossible alighting cases: {impossible_alighting}")
        
        return df
    
    def _calculate_loads_with_realistic_constraints(self, df):
        """Calculate loads with logical, last stop, AND realistic boarding constraints"""
        print("Calculating loads with logical, last stop, and realistic boarding constraints...")
        
        df = df.copy()
        logical_fixes = 0
        last_stop_fixes = 0
        boarding_constraints_applied = 0
        total_boarding_reduction = 0
        
        # Sort by bus and time to process sequentially
        df = df.sort_values(['bus_id', 'datetime']).reset_index(drop=True)
        
        for bus_id in df['bus_id'].unique():
            bus_data = df[df['bus_id'] == bus_id].copy()
            current_load = 0
            
            for idx in bus_data.index:
                capacity = df.loc[idx, 'capacity']
                predicted_boarding = df.loc[idx, 'boarding']
                alighting = df.loc[idx, 'alighting']
                line_id = df.loc[idx, 'line_id']
                stop_id = df.loc[idx, 'stop_id']
                
                # Set current load before any adjustments
                df.loc[idx, 'current_load'] = current_load
                
                # APPLY LAST STOP LOGIC FIRST
                is_last = self._is_last_stop(line_id, stop_id)
                if is_last:
                    # At last stop: boarding = 0, alighting = current_load
                    original_boarding = predicted_boarding
                    original_alighting = alighting
                    
                    boarding = 0  # No one boards at last stop
                    alighting = current_load  # Everyone gets off
                    
                    if original_boarding != boarding or original_alighting != alighting:
                        print(f"    Last stop fix: Bus {bus_id} Line {line_id} Stop {stop_id} - "
                              f"boarding {original_boarding}→{boarding}, "
                              f"alighting {original_alighting}→{alighting}")
                        last_stop_fixes += 1
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                
                else:
                    # APPLY REALISTIC BOARDING CONSTRAINTS
                    constrained_boarding = self._apply_realistic_boarding_constraints(
                        current_load, predicted_boarding, capacity
                    )
                    
                    if constrained_boarding != predicted_boarding:
                        boarding_constraints_applied += 1
                        total_boarding_reduction += (predicted_boarding - constrained_boarding)
                    
                    boarding = constrained_boarding
                    df.loc[idx, 'boarding'] = boarding
                    
                    # LOGICAL CONSTRAINT: alighting cannot exceed current load
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
                
                # Update current load for next iteration
                current_load = new_load
        
        print(f"\nCONSTRAINT APPLICATION SUMMARY:")
        print(f"  Last stop logic applied: {last_stop_fixes} records")
        print(f"  Logical constraints applied: {logical_fixes} records") 
        print(f"  Realistic boarding constraints applied: {boarding_constraints_applied} records")
        print(f"  Total boarding reduction: {total_boarding_reduction} passengers")
        
        # Report final realistic statistics
        impossible_alighting = sum(1 for idx, row in df.iterrows() 
                                  if row['alighting'] > row['current_load'] and row['current_load'] >= 0)
        
        overcrowded = (df['occupancy_rate'] > 1.0).sum()
        severely_overcrowded = (df['occupancy_rate'] > 1.5).sum()
        extremely_overcrowded = (df['occupancy_rate'] > 2.0).sum()
        max_occupancy = df['occupancy_rate'].max()
        avg_occupancy = df['occupancy_rate'].mean()
        
        print(f"\nREALISTIC OCCUPANCY STATISTICS:")
        print(f"  >100% capacity: {overcrowded} records ({overcrowded/len(df)*100:.1f}%)")
        print(f"  >150% capacity: {severely_overcrowded} records ({severely_overcrowded/len(df)*100:.1f}%)")
        print(f"  >200% capacity: {extremely_overcrowded} records ({extremely_overcrowded/len(df)*100:.1f}%)")
        print(f"  Maximum occupancy: {max_occupancy:.2f} ({max_occupancy*100:.0f}%)")
        print(f"  Average occupancy: {avg_occupancy:.2f} ({avg_occupancy*100:.0f}%)")
        print(f"  Impossible alighting cases: {impossible_alighting}")
        
        return df
    
    def _calculate_loads_with_last_stop_logic(self, df):
        """Calculate loads with logical and last stop constraints (NO CAPACITY HARD CAP)"""
        print("Calculating loads with logical and last stop constraints (no capacity limits)...")
        
        df = df.copy()
        logical_fixes = 0
        last_stop_fixes = 0
        
        # Sort by bus and time to process sequentially
        df = df.sort_values(['bus_id', 'datetime']).reset_index(drop=True)
        
        for bus_id in df['bus_id'].unique():
            bus_data = df[df['bus_id'] == bus_id].copy()
            current_load = 0
            
            for idx in bus_data.index:
                capacity = df.loc[idx, 'capacity']
                boarding = df.loc[idx, 'boarding']
                alighting = df.loc[idx, 'alighting']
                line_id = df.loc[idx, 'line_id']
                stop_id = df.loc[idx, 'stop_id']
                
                # Set current load before any adjustments
                df.loc[idx, 'current_load'] = current_load
                
                # APPLY LAST STOP LOGIC FIRST
                is_last = self._is_last_stop(line_id, stop_id)
                if is_last:
                    # At last stop: boarding = 0, alighting = current_load
                    original_boarding = boarding
                    original_alighting = alighting
                    
                    boarding = 0  # No one boards at last stop
                    alighting = current_load  # Everyone gets off
                    
                    if original_boarding != boarding or original_alighting != alighting:
                        print(f"    Last stop load fix: Bus {bus_id} Line {line_id} Stop {stop_id} - "
                              f"boarding {original_boarding}→{boarding}, "
                              f"alighting {original_alighting}→{alighting} (current_load={current_load})")
                        last_stop_fixes += 1
                    
                    df.loc[idx, 'boarding'] = boarding
                    df.loc[idx, 'alighting'] = alighting
                
                # CRITICAL FIX: Enforce logical constraint - alighting cannot exceed current load
                elif alighting > current_load:
                    print(f"  Logical fix: Bus {bus_id} at {df.loc[idx, 'datetime']} - "
                          f"alighting {alighting} > current_load {current_load}, "
                          f"setting alighting to {current_load}")
                    df.loc[idx, 'alighting'] = current_load
                    alighting = current_load
                    logical_fixes += 1
                
                # Recalculate final load with corrected values
                final_boarding = df.loc[idx, 'boarding']
                final_alighting = df.loc[idx, 'alighting']
                new_load = max(0, current_load + final_boarding - final_alighting)
                
                df.loc[idx, 'new_load'] = new_load
                df.loc[idx, 'occupancy_rate'] = new_load / capacity if capacity > 0 else 0
                
                # Update current load for next iteration
                current_load = new_load
        
        print(f"Applied last stop logic to {last_stop_fixes} records")
        print(f"Applied logical constraints to {logical_fixes} records")
        print("⚠️  NO CAPACITY CONSTRAINTS APPLIED - Natural occupancy rates preserved")
        
        # Report final statistics
        impossible_alighting = 0
        for idx, row in df.iterrows():
            if row['alighting'] > row['current_load'] and row['current_load'] >= 0:
                impossible_alighting += 1
        
        if impossible_alighting == 0:
            print("✅ All alighting values are logically consistent!")
        else:
            print(f"⚠️  Still {impossible_alighting} records with impossible alighting")
        
        # Report natural overcrowding statistics without intervention
        overcrowded = (df['occupancy_rate'] > 1.0).sum()
        severely_overcrowded = (df['occupancy_rate'] > 1.5).sum()
        extremely_overcrowded = (df['occupancy_rate'] > 2.0).sum()
        max_occupancy = df['occupancy_rate'].max()
        avg_occupancy = df['occupancy_rate'].mean()
        
        print(f"\nNATURAL OCCUPANCY STATISTICS (no caps applied):")
        print(f"  >100% capacity: {overcrowded} records ({overcrowded/len(df)*100:.1f}%)")
        print(f"  >150% capacity: {severely_overcrowded} records ({severely_overcrowded/len(df)*100:.1f}%)")
        print(f"  >200% capacity: {extremely_overcrowded} records ({extremely_overcrowded/len(df)*100:.1f}%)")
        print(f"  Maximum occupancy: {max_occupancy:.2f} ({max_occupancy*100:.0f}%)")
        print(f"  Average occupancy: {avg_occupancy:.2f} ({avg_occupancy*100:.0f}%)")
        
        # Validate last stop logic
        last_stop_violations = 0
        for idx, row in df.iterrows():
            if self._is_last_stop(row['line_id'], row['stop_id']):
                if row['boarding'] != 0:
                    last_stop_violations += 1
        
        if last_stop_violations == 0:
            print("✅ All last stops have zero boarding!")
        else:
            print(f"⚠️  {last_stop_violations} last stops still have non-zero boarding")
        
        return df
    
    def generate_boarding_constraint_analysis(self, df_before, df_after):
        """Analyze the impact of realistic boarding constraints"""
        print("\n" + "="*60)
        print("REALISTIC BOARDING CONSTRAINT ANALYSIS WITH NIGHT-TIME LOGIC")
        print("="*60)
        
        # Boarding comparison
        before_boarding = df_before['boarding'].sum()
        after_boarding = df_after['boarding'].sum()
        boarding_reduction = before_boarding - after_boarding
        
        print(f"\n🚌 BOARDING IMPACT:")
        print(f"  Before constraints: {before_boarding:,.0f} total boardings")
        print(f"  After constraints:  {after_boarding:,.0f} total boardings")
        print(f"  Total reduction:    {boarding_reduction:,.0f} boardings ({boarding_reduction/before_boarding*100:.1f}%)")
        
        # Night-time specific analysis (1-4 AM only)
        night_before = df_before[df_before['hour'].isin([1, 2, 3, 4])]
        night_after = df_after[df_after['hour'].isin([1, 2, 3, 4])]
        
        if len(night_before) > 0 and len(night_after) > 0:
            night_boarding_before = night_before['boarding'].sum()
            night_boarding_after = night_after['boarding'].sum()
            night_reduction = night_boarding_before - night_boarding_after
            
            night_load_before = night_before['new_load'].mean()
            night_load_after = night_after['new_load'].mean()
            
            print(f"\n🌙 NIGHT-TIME IMPACT (1-4 AM ONLY):")
            print(f"  Night boarding before: {night_boarding_before}")
            print(f"  Night boarding after:  {night_boarding_after}")
            print(f"  Night reduction:       {night_reduction} ({night_reduction/night_boarding_before*100:.1f}%)")
            print(f"  Average load before:   {night_load_before:.1f} passengers")
            print(f"  Average load after:    {night_load_after:.1f} passengers")
        
        # Occupancy comparison
        before_overcrowded = (df_before['occupancy_rate'] > 1.0).sum()
        after_overcrowded = (df_after['occupancy_rate'] > 1.0).sum()
        
        before_severe = (df_before['occupancy_rate'] > 1.5).sum()
        after_severe = (df_after['occupancy_rate'] > 1.5).sum()
        
        print(f"\n📊 OVERCROWDING REDUCTION:")
        print(f"  >100% Before: {before_overcrowded} ({before_overcrowded/len(df_before)*100:.1f}%)")
        print(f"  >100% After:  {after_overcrowded} ({after_overcrowded/len(df_after)*100:.1f}%)")
        print(f"  Improvement:  {before_overcrowded - after_overcrowded} records")
        
        print(f"  >150% Before: {before_severe} ({before_severe/len(df_before)*100:.1f}%)")
        print(f"  >150% After:  {after_severe} ({after_severe/len(df_after)*100:.1f}%)")
        print(f"  Improvement:  {before_severe - after_severe} records")
        
        # Compare to historical target (1.8% overcrowded)
        historical_target = 1.8
        current_overcrowded_pct = after_overcrowded/len(df_after)*100
        
        print(f"\n🎯 HISTORICAL COMPARISON:")
        print(f"  Historical overcrowding: ~{historical_target}%")
        print(f"  Current overcrowding:    {current_overcrowded_pct:.1f}%")
        
        if current_overcrowded_pct <= historical_target * 1.5:  # Within 50% of historical
            print("  ✅ EXCELLENT: Very close to realistic historical levels!")
        elif current_overcrowded_pct <= historical_target * 3:  # Within 3x of historical  
            print("  📊 GOOD: Significant improvement toward realistic levels")
        else:
            print("  ⚠️  Still above realistic levels - may need stricter constraints")
        
        return {
            'boarding_reduction_pct': boarding_reduction/before_boarding*100,
            'overcrowding_improvement': before_overcrowded - after_overcrowded,
            'final_overcrowding_pct': current_overcrowded_pct
        }
    
    def predict_with_patterns(self, historical_df, future_df, apply_route_adjustments=True, apply_realistic_constraints=True, apply_night_constraints=True):
        """Make predictions with route-aware fixes, realistic constraints, night-time logic (1-4 AM), and last stop logic"""
        print("Making pattern-aware predictions with night-time realistic boarding constraints (1-4 AM only)...")
        
        # Start with original future_df to preserve all columns
        predictions_df = future_df.copy()
        predictions_df = self._ensure_stop_id_consistency(predictions_df)
        
        # Debug: Check stop ID overlap
        future_stop_ids = set(predictions_df['stop_id'].unique())
        future_line_ids = set(predictions_df['line_id'].astype(str).unique())
        print(f"Future data contains {len(future_stop_ids)} unique stops and {len(future_line_ids)} unique lines")
        
        # Initialize prediction columns
        predictions_df['boarding_predicted'] = 0
        predictions_df['alighting_predicted'] = 0
        
        # Track last stop applications
        last_stop_applications = 0
        
        # Process each row individually to preserve all data
        print(f"\nProcessing {len(predictions_df)} prediction records...")
        for i, (idx, row) in enumerate(predictions_df.iterrows()):
            if i % 100 == 0:  # Progress indicator
                print(f"  Processing record {i+1}/{len(predictions_df)}")
                
            stop_id = row['stop_id']
            line_id = str(row['line_id'])  # Ensure string type
            
            try:
                if stop_id not in self.training_stop_ids:
                    # Use historical patterns for unknown stops
                    pred_boarding, pred_alighting = self._predict_from_patterns(row)
                else:
                    # Use pattern-based prediction for known stops
                    pred_boarding, pred_alighting = self._predict_from_patterns(row, stop_id)
                
                # Apply last stop logic EARLY - before other adjustments
                if self._is_last_stop(line_id, stop_id):
                    original_boarding = pred_boarding
                    pred_boarding = 0  # No one boards at last stop
                    last_stop_applications += 1
                    print(f"    Last stop fix applied: Line {line_id}, Stop {stop_id}, Boarding {original_boarding}→0")
                    # Keep pred_alighting as predicted - will be corrected in load calculation
                
                # Update predictions in the original dataframe
                predictions_df.loc[idx, 'boarding_predicted'] = pred_boarding
                predictions_df.loc[idx, 'alighting_predicted'] = pred_alighting
                
                # Replace original boarding/alighting with predictions
                predictions_df.loc[idx, 'boarding'] = pred_boarding
                predictions_df.loc[idx, 'alighting'] = pred_alighting
                
            except Exception as e:
                print(f"Error predicting for stop {stop_id} at {row['datetime']}: {e}")
                # Use fallback predictions
                pred_boarding, pred_alighting = self._predict_from_patterns(row)
                
                # Apply last stop logic for fallback too
                if self._is_last_stop(line_id, stop_id):
                    pred_boarding = 0
                    last_stop_applications += 1
                
                predictions_df.loc[idx, 'boarding_predicted'] = pred_boarding
                predictions_df.loc[idx, 'alighting_predicted'] = pred_alighting
                predictions_df.loc[idx, 'boarding'] = pred_boarding
                predictions_df.loc[idx, 'alighting'] = pred_alighting
        
        print(f"\nLast stop logic applied to {last_stop_applications} records")
        
        # Store original predictions for comparison
        predictions_before_constraints = predictions_df.copy()
        
        # Apply route-aware fixes (conservation enforcement removed)
        print("\n=== APPLYING ROUTE-AWARE FIXES ===")
        
        if apply_route_adjustments:
            predictions_df = self._apply_route_aware_adjustments(predictions_df)
        else:
            print("Skipping route-aware adjustments (disabled)")
        
        print("Conservation enforcement skipped - using natural predictions")
        
        # Apply enhanced realistic constraints with night-time awareness (1-4 AM only)
        if apply_realistic_constraints and apply_night_constraints:
            print("\n=== APPLYING ENHANCED REALISTIC CONSTRAINTS WITH NIGHT-TIME LOGIC (1-4 AM ONLY) ===")
            predictions_df = self._calculate_loads_with_realistic_and_night_constraints(predictions_df)
            
            # Generate detailed constraint analysis
            self.generate_boarding_constraint_analysis(predictions_before_constraints, predictions_df)
            
        elif apply_realistic_constraints:
            print("\n=== APPLYING REALISTIC BOARDING CONSTRAINTS (NO NIGHT-TIME LOGIC) ===")
            predictions_df = self._calculate_loads_with_realistic_constraints(predictions_df)
            
            # Generate detailed constraint analysis
            self.generate_boarding_constraint_analysis(predictions_before_constraints, predictions_df)
        else:
            print("Using original load calculation without realistic constraints")
            predictions_df = self._calculate_loads_with_last_stop_logic(predictions_df)
        
        # Validate predictions against historical patterns
        self._validate_prediction_patterns(predictions_df)
        
        print(f"Completed predictions with night-time realistic constraints (1-4 AM only) for {len(predictions_df)} records")
        return predictions_df
    
    def _apply_route_aware_adjustments(self, df):
        """Apply route-aware boarding/alighting adjustments based on stop types"""
        print("Applying route-aware adjustments...")
        
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Define adjustment multipliers by stop type and time (MINIMAL INTENSITY)
        adjustments = {
            'ORIGIN': {
                'rush_hour': {'boarding_mult': 1.05, 'alighting_mult': 0.95},    # Very gentle: 5% changes
                'off_peak': {'boarding_mult': 1.03, 'alighting_mult': 0.97}      # Very gentle: 3% changes
            },
            'DESTINATION': {
                'rush_hour': {'boarding_mult': 0.95, 'alighting_mult': 1.05},    # Very gentle: 5% changes
                'off_peak': {'boarding_mult': 0.97, 'alighting_mult': 1.03}      # Very gentle: 3% changes
            },
            'MIDDLE': {
                'rush_hour': {'boarding_mult': 1.0, 'alighting_mult': 1.0},
                'off_peak': {'boarding_mult': 1.0, 'alighting_mult': 1.0}
            }
        }
        
        adjustments_applied = 0
        total_boarding_change = 0
        total_alighting_change = 0
        last_stop_skipped = 0
        
        for idx, row in df.iterrows():
            stop_id = row['stop_id']
            line_id = row['line_id']
            is_rush = row['is_rush_hour']
            
            # Skip adjustments for last stops - they have special logic
            if self._is_last_stop(line_id, stop_id):
                last_stop_skipped += 1
                continue
            
            # Get stop classification
            if stop_id in self.stop_classifications:
                stop_type = self.stop_classifications[stop_id]['primary_type']
                confidence = self.stop_classifications[stop_id]['confidence']
                
                # Only apply adjustments for high-confidence classifications, but reduce intensity
                if confidence > 0.7:  # Increased confidence threshold from 0.6 to 0.7
                    time_key = 'rush_hour' if is_rush else 'off_peak'
                    mult_data = adjustments[stop_type][time_key]
                    
                    # Apply adjustments with reduced confidence weighting
                    confidence_factor = min(confidence, 0.8)  # Cap confidence impact
                    boarding_mult = 1.0 + (mult_data['boarding_mult'] - 1.0) * confidence_factor
                    alighting_mult = 1.0 + (mult_data['alighting_mult'] - 1.0) * confidence_factor
                    
                    # Apply multipliers
                    original_boarding = df.loc[idx, 'boarding']
                    original_alighting = df.loc[idx, 'alighting']
                    
                    new_boarding = max(0, round(original_boarding * boarding_mult))
                    new_alighting = max(0, round(original_alighting * alighting_mult))
                    
                    df.loc[idx, 'boarding'] = new_boarding
                    df.loc[idx, 'alighting'] = new_alighting
                    
                    # Track changes
                    total_boarding_change += (new_boarding - original_boarding)
                    total_alighting_change += (new_alighting - original_alighting)
                    
                    adjustments_applied += 1
        
        print(f"Applied route-aware adjustments to {adjustments_applied} records")
        print(f"Skipped {last_stop_skipped} last stops (have special logic)")
        print(f"  Net boarding change: {total_boarding_change:+.0f}")
        print(f"  Net alighting change: {total_alighting_change:+.0f}")
        print(f"  Net route adjustment imbalance: {total_boarding_change - total_alighting_change:+.0f}")
        
        return df
    
    def _predict_from_patterns(self, future_row, stop_id=None):
        """Predict using historical patterns when model fails"""
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
                # Default fallback
                boarding = 8.0
                alighting = 7.0
        
        return max(0, round(boarding)), max(0, round(alighting))
    
    def _validate_prediction_patterns(self, predictions_df):
        """Validate predictions against historical patterns"""
        print("\n=== PREDICTION PATTERN VALIDATION ===")
        
        pred_temp = predictions_df.copy()
        pred_temp['datetime'] = pd.to_datetime(pred_temp['datetime'])
        pred_temp['hour'] = pred_temp['datetime'].dt.hour
        pred_temp['is_weekend'] = pred_temp['datetime'].dt.dayofweek >= 5
        
        # Compare hourly patterns
        pred_hourly = pred_temp.groupby('hour')[['boarding', 'alighting']].mean()
        
        # Check if rush hours are preserved
        rush_hours = [7, 8, 9, 17, 18, 19]
        off_peak_hours = [10, 11, 12, 13, 14, 15, 16]
        
        if len(pred_hourly) > 0:
            rush_avg = pred_hourly.loc[pred_hourly.index.intersection(rush_hours), 'boarding'].mean()
            off_peak_avg = pred_hourly.loc[pred_hourly.index.intersection(off_peak_hours), 'boarding'].mean()
            
            if off_peak_avg > 0:
                ratio = rush_avg / off_peak_avg
                print(f"Rush hour ratio: {ratio:.2f} (should be 2.0+)")
                if ratio > 2.0:
                    print("✅ Good rush hour pattern preservation!")
                elif ratio > 1.5:
                    print("📊 Moderate rush hour pattern preservation")
                else:
                    print("⚠️  Weak rush hour patterns")
            
            print(f"Predicted rush hour avg: {rush_avg:.1f}")
            print(f"Predicted off-peak avg: {off_peak_avg:.1f}")
        
        # Validate last stop logic
        if self.last_stops:
            last_stop_boarding_sum = 0
            last_stop_count = 0
            
            for idx, row in pred_temp.iterrows():
                if self._is_last_stop(row['line_id'], row['stop_id']):
                    last_stop_boarding_sum += row['boarding']
                    last_stop_count += 1
            
            print(f"\nLast stop validation:")
            print(f"  Last stop records: {last_stop_count}")
            print(f"  Total boarding at last stops: {last_stop_boarding_sum}")
            if last_stop_boarding_sum == 0:
                print("  ✅ Perfect last stop logic!")
            else:
                print(f"  ⚠️  {last_stop_boarding_sum} boarding at last stops (should be 0)")
        
        # Validate night-time logic (1-4 AM only)
        night_data = pred_temp[pred_temp['hour'].isin([1, 2, 3, 4])]
        if len(night_data) > 0:
            night_avg_load = night_data['new_load'].mean()
            night_max_load = night_data['new_load'].max()
            night_boarding_total = night_data['boarding'].sum()
            
            print(f"\nNight-time validation (1-4 AM only):")
            print(f"  Night records: {len(night_data)}")
            print(f"  Average load: {night_avg_load:.1f} passengers")
            print(f"  Maximum load: {night_max_load} passengers")
            print(f"  Total boarding: {night_boarding_total} passengers")
            
            if night_avg_load <= 8 and night_max_load <= 20:
                print("  ✅ Realistic night-time loads!")
            else:
                print("  ⚠️  Night loads may still be high")
    
    def generate_conservation_report(self, original_df, fixed_df):
        """Generate a comprehensive report of the fixes applied (without conservation enforcement)"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FIX REPORT WITH LAST STOP LOGIC, REALISTIC CONSTRAINTS, AND NIGHT-TIME LOGIC (1-4 AM)")
        print("="*60)
        
        # Basic comparison without conservation enforcement
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
        
        # Night-time analysis (1-4 AM only)
        orig_night = original_df[original_df['hour'].isin([1, 2, 3, 4])]
        fixed_night = fixed_df[fixed_df['hour'].isin([1, 2, 3, 4])]
        
        if len(orig_night) > 0 and len(fixed_night) > 0:
            orig_night_avg = orig_night['alighting'].mean() if 'alighting' in orig_night.columns else 0
            fixed_night_avg = fixed_night['new_load'].mean() if 'new_load' in fixed_night.columns else 0
            
            print(f"\n🌙 NIGHT-TIME LOGIC (1-4 AM ONLY):")
            print(f"  Night records: {len(fixed_night)}")
            print(f"  Original avg alighting: {orig_night_avg:.1f}")
            print(f"  Fixed avg load: {fixed_night_avg:.1f}")
            print(f"  Status: {'✅ REALISTIC' if fixed_night_avg <= 8 else '⚠️  STILL HIGH'}")
        
        # Overcrowding comparison
        orig_overcrowded = (original_df['occupancy_rate'] > 1.0).sum()
        orig_severe = (original_df['occupancy_rate'] > 1.5).sum()
        
        fixed_overcrowded = (fixed_df['occupancy_rate'] > 1.0).sum()
        fixed_severe = (fixed_df['occupancy_rate'] > 1.5).sum()
        
        print(f"\n🚌 OVERCROWDING ANALYSIS:")
        print(f"  Original >100%: {orig_overcrowded} ({orig_overcrowded/len(original_df)*100:.1f}%)")
        print(f"  Fixed >100%:    {fixed_overcrowded} ({fixed_overcrowded/len(fixed_df)*100:.1f}%)")
        print(f"  Original >150%: {orig_severe} ({orig_severe/len(original_df)*100:.1f}%)")
        print(f"  Fixed >150%:    {fixed_severe} ({fixed_severe/len(fixed_df)*100:.1f}%)")
        
        # Max occupancy
        orig_max_occ = original_df['occupancy_rate'].max()
        fixed_max_occ = fixed_df['occupancy_rate'].max()
        
        print(f"\n📊 MAXIMUM OCCUPANCY:")
        print(f"  Original: {orig_max_occ:.2f} ({orig_max_occ*100:.0f}%)")
        print(f"  Fixed:    {fixed_max_occ:.2f} ({fixed_max_occ*100:.0f}%)")
        
        # Store stats for future reference
        self.conservation_stats = {
            'original_max_occupancy': orig_max_occ,
            'fixed_max_occupancy': fixed_max_occ,
            'original_overcrowded_pct': orig_overcrowded/len(original_df)*100,
            'fixed_overcrowded_pct': fixed_overcrowded/len(fixed_df)*100,
            'last_stop_records': last_stop_count if self.last_stops else 0,
            'fixed_last_stop_boarding': fixed_last_stop_boarding if self.last_stops else 0,
            'original_imbalance': orig_imbalance,
            'final_imbalance': fixed_imbalance
        }


def compare_with_historical(historical_df, predictions_df, save_dir=None):
    """Compare predictions with historical data patterns"""
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
    ax1.plot(hours, pred_boarding, 'r--', linewidth=2, label='Predicted', marker='s')
    ax1.fill_between([7, 9], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='yellow', label='Morning Rush')
    ax1.fill_between([17, 19], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='orange', label='Evening Rush')
    ax1.fill_between([1, 4], 0, max(max(hist_boarding), max(pred_boarding)) * 1.1, alpha=0.2, color='purple', label='Night Constraints')
    ax1.set_title('Weekday Boarding Patterns Comparison')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Average Passengers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Alighting comparison
    hist_alighting = [hist_weekday.loc[h, 'alighting'] if h in hist_weekday.index else 0 for h in hours]
    pred_alighting = [pred_weekday.loc[h, 'alighting'] if h in pred_weekday.index else 0 for h in hours]
    
    ax2.plot(hours, hist_alighting, 'b-', linewidth=2, label='Historical', marker='o')
    ax2.plot(hours, pred_alighting, 'r--', linewidth=2, label='Predicted', marker='s')
    ax2.fill_between([7, 9], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='yellow')
    ax2.fill_between([17, 19], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='orange')
    ax2.fill_between([1, 4], 0, max(max(hist_alighting), max(pred_alighting)) * 1.1, alpha=0.2, color='purple')
    ax2.set_title('Weekday Alighting Patterns Comparison')
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
    ax3.set_ylabel('Predicted Average')
    ax3.set_title('Pattern Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax4.hist(historical_df['boarding'], bins=30, alpha=0.5, label='Historical Boarding', color='blue')
    ax4.hist(predictions_df['boarding'], bins=30, alpha=0.5, label='Predicted Boarding', color='red')
    ax4.set_xlabel('Passenger Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution Comparison')
    ax4.legend()
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'historical_vs_predicted_with_night_constraints.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Calculate correlation
    correlation_boarding = np.corrcoef(hist_boarding, pred_boarding)[0, 1] if len(hist_boarding) > 1 else 0
    correlation_alighting = np.corrcoef(hist_alighting, pred_alighting)[0, 1] if len(hist_alighting) > 1 else 0
    
    print(f"\nPattern Correlation:")
    print(f"  Boarding: {correlation_boarding:.3f}")
    print(f"  Alighting: {correlation_alighting:.3f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pattern-Preserving LSTM with Night-Time Constraints (1-4 AM Only)')
    parser.add_argument('--historical', required=True, help='Historical data CSV')
    parser.add_argument('--future', help='Future data CSV')
    parser.add_argument('--bus-stops', default='ankara_bus_stops.csv', help='Bus stops CSV file')
    parser.add_argument('--output', default='output_night_constraints_1_4am', help='Output directory')
    parser.add_argument('--sequence', type=int, default=48, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--disable-route-adjustments', action='store_true', 
                       help='Disable route-aware adjustments')
    parser.add_argument('--disable-realistic-constraints', action='store_true',
                       help='Disable realistic boarding constraints (allow unlimited overcrowding)')
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
        model = LSTM(sequence_length=args.sequence)
        
        # Load bus stops data if provided
        if args.bus_stops:
            model._load_last_stops(args.bus_stops)
        else:
            print("Warning: No bus stops CSV provided. Last stop logic will be disabled.")
            print("Use --bus-stops ankara_bus_stops.csv to enable last stop logic.")
        
        # Analyze historical patterns first
        model.analyze_historical_patterns(historical_df)
        
        # Prepare features (creates stop mapping)
        prepared_data = model.prepare_enhanced_features(historical_df, create_mapping=True)
        
        # Create sequences and train
        train_X, train_y, val_X, val_y, metadata = model.create_pattern_sequences(prepared_data)
        
        if train_X is not None:
            # Train model
            history = model.train_with_pattern_validation(train_X, train_y, val_X, val_y, epochs=args.epochs)
            
            # Make predictions if future data provided
            if args.future:
                print(f"\nLoading future data from {args.future}")
                future_df = pd.read_csv(args.future)
                
                # Store original future data for comparison
                original_future_df = future_df.copy()
                
                # Apply constraints based on flags
                use_route_adjustments = not args.disable_route_adjustments
                use_realistic_constraints = not args.disable_realistic_constraints
                use_night_constraints = not args.disable_night_constraints
                
                print(f"Configuration:")
                print(f"  Route adjustments: {'ON' if use_route_adjustments else 'OFF'}")
                print(f"  Realistic constraints: {'ON' if use_realistic_constraints else 'OFF'}")
                print(f"  Night-time constraints (1-4 AM): {'ON' if use_night_constraints else 'OFF'}")
                
                predictions_df = model.predict_with_patterns(
                    historical_df, 
                    future_df, 
                    apply_route_adjustments=use_route_adjustments,
                    apply_realistic_constraints=use_realistic_constraints,
                    apply_night_constraints=use_night_constraints
                )
                
                # Generate report (no conservation balancing)
                model.generate_conservation_report(original_future_df, predictions_df)
                
                # Save results with appropriate naming
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                output_file = os.path.join(data_dir, f"future_predictions_{timestamp}.csv")
                predictions_df.to_csv(output_file, index=False)
                print(f"\nPredictions saved to {output_file}")
                
                # Compare with historical patterns
                compare_with_historical(historical_df, predictions_df, plots_dir)
                
                print("Pattern-preserving model with night-time constraints (1-4 AM only) completed successfully!")
            
        else:
            print("Failed to create training sequences")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()