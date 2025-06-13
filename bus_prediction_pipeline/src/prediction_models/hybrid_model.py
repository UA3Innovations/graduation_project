#!/usr/bin/env python3
"""
Hybrid LSTM-Prophet Model for Transportation Passenger Flow Prediction

Weighting Logic:
- Normal days: 0.7 LSTM + 0.3 Prophet  
- Special days: 0.7 Prophet + 0.3 LSTM

Special days include:
- Turkish national holidays
- Islamic holidays 

Features:
- Trains and saves both LSTM and Prophet models
- Applies same constraints as individual models
- Intelligent date-based weighting
- Comprehensive prediction and reporting
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import LSTM and Prophet models
from .lstm_model import lstm_model
from .prophet_model import ProphetModel

# Set random seeds for reproducibility
np.random.seed(42)

class HybridModel:
    """
    Hybrid model combining LSTM and Prophet with intelligent weighting
    """
    
    def __init__(self, sequence_length=48):
        self.lstm_model = lstm_model(sequence_length=sequence_length)
        self.prophet_model = ProphetModel()
        self.is_trained = False
        
        # Hybrid-specific settings
        self.normal_day_weights = {'lstm': 0.7, 'prophet': 0.3}
        self.special_day_weights = {'lstm': 0.3, 'prophet': 0.7}
        
        # Special dates for 2025 (pre-calculated)
        self.special_dates_2025 = self._define_special_dates_2025()
        
        # Model saving paths
        self.model_save_dir = None
        
    def _define_special_dates_2025(self):
        """Define all special dates for 2025"""
        special_dates = set()
        
        # Turkish National Holidays 2025
        national_holidays = [
            '2025-01-01',  # New Year's Day
            '2025-04-23',  # National Sovereignty and Children's Day
            '2025-05-01',  # Labor Day
            '2025-05-19',  # Commemoration of Atatürk, Youth and Sports Day
            '2025-08-30',  # Victory Day
            '2025-10-29'   # Republic Day
        ]
        
        # Islamic Holidays 2025 (pre-calculated dates)
        islamic_holidays = {
            
            # Eid al-Fitr (Ramazan Bayramı) - 3 days
            'eid_fitr': [
                '2025-03-30', '2025-03-31', '2025-04-01'
            ],
            
            # Eid al-Adha (Kurban Bayramı) - 4 days  
            'eid_adha': [
                '2025-06-06', '2025-06-07', '2025-06-08', '2025-06-09'
            ]
        }
        
        # Add national holidays
        for date_str in national_holidays:
            special_dates.add(pd.to_datetime(date_str).date())
        
        # Add Islamic holidays
        for holiday_type, dates in islamic_holidays.items():
            if holiday_type == 'ramadan_month':
                # Add entire Ramadan month
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[1])
                current_date = start_date
                while current_date <= end_date:
                    special_dates.add(current_date.date())
                    current_date += timedelta(days=1)
            else:
                # Add specific holiday dates
                for date_str in dates:
                    special_dates.add(pd.to_datetime(date_str).date())
        
        print(f"Defined {len(special_dates)} special dates for 2025")
        print("Special date types:")
        print("  - Turkish National Holidays:", len(national_holidays))
        print("  - Ramadan Month: 30 days")
        print("  - Eid al-Fitr: 3 days")
        print("  - Eid al-Adha: 4 days")
        
        return special_dates
    
    def is_special_date(self, date):
        """Check if a given date is a special date"""
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        
        return date in self.special_dates_2025
    
    def get_prediction_weights(self, date):
        """Get weights for LSTM and Prophet based on date"""
        if self.is_special_date(date):
            return self.special_day_weights.copy()
        else:
            return self.normal_day_weights.copy()
    
    def train_hybrid_models(self, historical_df, save_models=True, model_save_dir='hybrid_models'):
        """Train both LSTM and Prophet models"""
        print("="*60)
        print("TRAINING HYBRID MODEL (LSTM + PROPHET)")
        print("="*60)
        
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Train LSTM model
        print("\n" + "="*40)
        print("TRAINING LSTM MODEL")
        print("="*40)
        
        # Analyze patterns and prepare data for LSTM
        self.lstm_model.analyze_historical_patterns(historical_df)
        prepared_data = self.lstm_model.prepare_enhanced_features(historical_df, create_mapping=True)
        train_X, train_y, val_X, val_y, metadata = self.lstm_model.create_pattern_sequences(prepared_data)
        
        if train_X is not None:
            lstm_history = self.lstm_model.train_with_pattern_validation(train_X, train_y, val_X, val_y)
            print("✅ LSTM model training completed successfully")
        else:
            print("❌ LSTM model training failed - no sequences created")
            return False
        
        # Train Prophet model
        print("\n" + "="*40)
        print("TRAINING PROPHET MODEL")
        print("="*40)
        
        # Analyze patterns for Prophet (reuse LSTM patterns)
        self.prophet_model.analyze_historical_patterns(historical_df)
        self.prophet_model.train_prophet_models(historical_df)
        
        if self.prophet_model.is_trained:
            print("✅ Prophet model training completed successfully")
        else:
            print("❌ Prophet model training failed")
            return False
        
        # Save models if requested
        if save_models:
            self._save_models()
        
        self.is_trained = True
        print("\n" + "="*60)
        print("HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return True
    
    def _save_models(self):
        """Save both trained models"""
        print(f"\nSaving models to {self.model_save_dir}...")
        
        try:
            # Save LSTM model
            lstm_save_path = os.path.join(self.model_save_dir, 'lstm_model.h5')
            if self.lstm_model.model is not None:
                self.lstm_model.model.save(lstm_save_path)
                print(f"  ✅ LSTM model saved to {lstm_save_path}")
            
            # Save LSTM metadata
            lstm_metadata = {
                'feature_scaler': self.lstm_model.feature_scaler,
                'target_scaler': self.lstm_model.target_scaler,
                'stop_mapping': self.lstm_model.stop_mapping,
                'training_stop_ids': self.lstm_model.training_stop_ids,
                'historical_patterns': self.lstm_model.historical_patterns,
                'stop_patterns': self.lstm_model.stop_patterns,
                'hourly_patterns': self.lstm_model.hourly_patterns,
                'route_patterns': self.lstm_model.route_patterns,
                'stop_classifications': self.lstm_model.stop_classifications,
                'last_stops': self.lstm_model.last_stops,
                'feature_columns': self.lstm_model.feature_columns,
                'sequence_length': self.lstm_model.sequence_length
            }
            
            with open(os.path.join(self.model_save_dir, 'lstm_metadata.pkl'), 'wb') as f:
                pickle.dump(lstm_metadata, f)
            print(f"  ✅ LSTM metadata saved")
            
            # Save Prophet models
            prophet_models_path = os.path.join(self.model_save_dir, 'prophet_models.pkl')
            with open(prophet_models_path, 'wb') as f:
                pickle.dump(self.prophet_model.models, f)
            print(f"  ✅ Prophet models saved to {prophet_models_path}")
            
            # Save Prophet metadata
            prophet_metadata = {
                'training_stop_ids': self.prophet_model.training_stop_ids,
                'historical_patterns': self.prophet_model.historical_patterns,
                'stop_patterns': self.prophet_model.stop_patterns,
                'hourly_patterns': self.prophet_model.hourly_patterns,
                'route_patterns': self.prophet_model.route_patterns,
                'stop_classifications': self.prophet_model.stop_classifications,
                'last_stops': self.prophet_model.last_stops,
                'prophet_params': self.prophet_model.prophet_params
            }
            
            with open(os.path.join(self.model_save_dir, 'prophet_metadata.pkl'), 'wb') as f:
                pickle.dump(prophet_metadata, f)
            print(f"  ✅ Prophet metadata saved")
            
            # Save hybrid model metadata
            hybrid_metadata = {
                'normal_day_weights': self.normal_day_weights,
                'special_day_weights': self.special_day_weights,
                'special_dates_2025': self.special_dates_2025,
                'is_trained': self.is_trained
            }
            
            with open(os.path.join(self.model_save_dir, 'hybrid_metadata.pkl'), 'wb') as f:
                pickle.dump(hybrid_metadata, f)
            print(f"  ✅ Hybrid metadata saved")
            
        except Exception as e:
            print(f"  ❌ Error saving models: {e}")
    
    def load_models(self, model_save_dir='hybrid_models'):
        """Load previously trained models"""
        print(f"Loading models from {model_save_dir}...")
        
        try:
            # Load LSTM model
            lstm_model_path = os.path.join(model_save_dir, 'lstm_model.h5')
            if os.path.exists(lstm_model_path):
                from tensorflow.keras.models import load_model
                self.lstm_model.model = load_model(lstm_model_path)
                print(f"  ✅ LSTM model loaded from {lstm_model_path}")
            
            # Load LSTM metadata
            with open(os.path.join(model_save_dir, 'lstm_metadata.pkl'), 'rb') as f:
                lstm_metadata = pickle.load(f)
            
            for key, value in lstm_metadata.items():
                setattr(self.lstm_model, key, value)
            
            self.lstm_model.is_trained = True
            print(f"  ✅ LSTM metadata loaded")
            
            # Load Prophet models
            with open(os.path.join(model_save_dir, 'prophet_models.pkl'), 'rb') as f:
                self.prophet_model.models = pickle.load(f)
            print(f"  ✅ Prophet models loaded")
            
            # Load Prophet metadata
            with open(os.path.join(model_save_dir, 'prophet_metadata.pkl'), 'rb') as f:
                prophet_metadata = pickle.load(f)
            
            for key, value in prophet_metadata.items():
                setattr(self.prophet_model, key, value)
            
            self.prophet_model.is_trained = True
            print(f"  ✅ Prophet metadata loaded")
            
            # Load hybrid metadata
            with open(os.path.join(model_save_dir, 'hybrid_metadata.pkl'), 'rb') as f:
                hybrid_metadata = pickle.load(f)
            
            for key, value in hybrid_metadata.items():
                setattr(self, key, value)
            
            print(f"  ✅ Hybrid metadata loaded")
            print("✅ All models loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_hybrid(self, historical_df, future_df, apply_route_adjustments=True, 
                      apply_realistic_constraints=True, apply_night_constraints=True):
        """Make hybrid predictions using weighted combination of LSTM and Prophet"""
        
        if not self.is_trained:
            print("❌ Models not trained! Train models first or load pre-trained models.")
            return None
        
        print("="*60)
        print("MAKING HYBRID PREDICTIONS (LSTM + PROPHET)")
        print("="*60)
        
        print("\nWeighting strategy:")
        print(f"  Normal days: {self.normal_day_weights}")
        print(f"  Special days: {self.special_day_weights}")
        
        # Prepare future data
        predictions_df = future_df.copy()
        predictions_df = self.lstm_model._ensure_stop_id_consistency(predictions_df)
        predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
        
        # Analyze dates in future data
        future_dates = predictions_df['datetime'].dt.date.unique()
        special_date_count = sum(1 for date in future_dates if self.is_special_date(date))
        normal_date_count = len(future_dates) - special_date_count
        
        print(f"\nFuture data analysis:")
        print(f"  Total unique dates: {len(future_dates)}")
        print(f"  Normal dates: {normal_date_count}")
        print(f"  Special dates: {special_date_count}")
        
        if special_date_count > 0:
            print(f"  Special dates found:")
            for date in future_dates:
                if self.is_special_date(date):
                    print(f"    {date}")
        
        # Get predictions from both models
        print("\n" + "="*40)
        print("GETTING LSTM PREDICTIONS")
        print("="*40)
        
        lstm_predictions = self.lstm_model.predict_with_patterns(
            historical_df, future_df,
            apply_route_adjustments=apply_route_adjustments,
            apply_realistic_constraints=False,  # Apply constraints later in hybrid
            apply_night_constraints=False
        )
        
        print("\n" + "="*40)
        print("GETTING PROPHET PREDICTIONS")  
        print("="*40)
        
        prophet_predictions = self.prophet_model.predict_with_patterns(
            historical_df, future_df,
            apply_route_adjustments=apply_route_adjustments,
            apply_realistic_constraints=False,  # Apply constraints later in hybrid
            apply_night_constraints=False
        )
        
        # Combine predictions using date-based weighting
        print("\n" + "="*40)
        print("COMBINING PREDICTIONS WITH INTELLIGENT WEIGHTING")
        print("="*40)
        
        predictions_df['boarding_lstm'] = lstm_predictions['boarding']
        predictions_df['alighting_lstm'] = lstm_predictions['alighting'] 
        predictions_df['boarding_prophet'] = prophet_predictions['boarding']
        predictions_df['alighting_prophet'] = prophet_predictions['alighting']
        
        # Apply intelligent weighting
        predictions_df['is_special_date'] = predictions_df['datetime'].apply(self.is_special_date)
        predictions_df['lstm_weight'] = 0.0
        predictions_df['prophet_weight'] = 0.0
        
        normal_count = 0
        special_count = 0
        
        for idx, row in predictions_df.iterrows():
            weights = self.get_prediction_weights(row['datetime'])
            predictions_df.loc[idx, 'lstm_weight'] = weights['lstm']
            predictions_df.loc[idx, 'prophet_weight'] = weights['prophet']
            
            if self.is_special_date(row['datetime']):
                special_count += 1
            else:
                normal_count += 1
        
        print(f"Applied weighting to {len(predictions_df)} records:")
        print(f"  Normal day records: {normal_count} (LSTM: 70%, Prophet: 30%)")
        print(f"  Special day records: {special_count} (LSTM: 30%, Prophet: 70%)")
        
        # Calculate weighted predictions
        predictions_df['boarding'] = (
            predictions_df['lstm_weight'] * predictions_df['boarding_lstm'] +
            predictions_df['prophet_weight'] * predictions_df['boarding_prophet']
        ).round().astype(int)
        
        predictions_df['alighting'] = (
            predictions_df['lstm_weight'] * predictions_df['alighting_lstm'] +
            predictions_df['prophet_weight'] * predictions_df['alighting_prophet']
        ).round().astype(int)
        
        # Store original for comparison
        predictions_before_constraints = predictions_df.copy()
        
        # Apply all constraints using LSTM's constraint system
        print("\n" + "="*40)
        print("APPLYING CONSTRAINTS TO HYBRID PREDICTIONS")
        print("="*40)
        
        if apply_realistic_constraints and apply_night_constraints:
            predictions_df = self.lstm_model._calculate_loads_with_realistic_and_night_constraints(predictions_df)
            self.lstm_model.generate_boarding_constraint_analysis(predictions_before_constraints, predictions_df)
        elif apply_realistic_constraints:
            predictions_df = self.lstm_model._calculate_loads_with_realistic_constraints(predictions_df)
            self.lstm_model.generate_boarding_constraint_analysis(predictions_before_constraints, predictions_df)
        else:
            predictions_df = self.lstm_model._calculate_loads_with_last_stop_logic(predictions_df)
        
        # Validate predictions
        self._validate_hybrid_predictions(predictions_df)
        
        print("\n" + "="*60)
        print("HYBRID PREDICTIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return predictions_df
    
    def _validate_hybrid_predictions(self, predictions_df):
        """Validate hybrid predictions"""
        print("\n=== HYBRID PREDICTION VALIDATION ===")
        
        pred_temp = predictions_df.copy()
        pred_temp['datetime'] = pd.to_datetime(pred_temp['datetime'])
        pred_temp['hour'] = pred_temp['datetime'].dt.hour
        
        # Overall statistics
        total_boarding = pred_temp['boarding'].sum()
        total_alighting = pred_temp['alighting'].sum()
        avg_boarding = pred_temp['boarding'].mean()
        avg_alighting = pred_temp['alighting'].mean()
        
        print(f"Overall prediction statistics:")
        print(f"  Total boarding: {total_boarding:,}")
        print(f"  Total alighting: {total_alighting:,}")
        print(f"  Average boarding per record: {avg_boarding:.2f}")
        print(f"  Average alighting per record: {avg_alighting:.2f}")
        print(f"  Imbalance: {total_boarding - total_alighting:,}")
        
        # Special vs normal day analysis
        normal_day_preds = pred_temp[~pred_temp['is_special_date']]
        special_day_preds = pred_temp[pred_temp['is_special_date']]
        
        if len(normal_day_preds) > 0:
            print(f"\nNormal days ({len(normal_day_preds)} records):")
            print(f"  Avg boarding: {normal_day_preds['boarding'].mean():.2f}")
            print(f"  Avg alighting: {normal_day_preds['alighting'].mean():.2f}")
        
        if len(special_day_preds) > 0:
            print(f"\nSpecial days ({len(special_day_preds)} records):")
            print(f"  Avg boarding: {special_day_preds['boarding'].mean():.2f}")
            print(f"  Avg alighting: {special_day_preds['alighting'].mean():.2f}")
        
        # Rush hour validation
        rush_hours = [7, 8, 9, 17, 18, 19]
        off_peak_hours = [10, 11, 12, 13, 14, 15, 16]
        
        rush_avg = pred_temp[pred_temp['hour'].isin(rush_hours)]['boarding'].mean()
        off_peak_avg = pred_temp[pred_temp['hour'].isin(off_peak_hours)]['boarding'].mean()
        
        if off_peak_avg > 0:
            ratio = rush_avg / off_peak_avg
            print(f"\nRush hour validation:")
            print(f"  Rush hour avg boarding: {rush_avg:.2f}")
            print(f"  Off-peak avg boarding: {off_peak_avg:.2f}")
            print(f"  Rush hour ratio: {ratio:.2f}")
            
            if ratio > 2.0:
                print("  ✅ Good rush hour pattern preservation!")
            elif ratio > 1.5:
                print("  📊 Moderate rush hour pattern preservation")
            else:
                print("  ⚠️  Weak rush hour patterns")
    
    def generate_hybrid_report(self, original_df, hybrid_df):
        """Generate comprehensive hybrid model report"""
        print("\n" + "="*60)
        print("HYBRID MODEL COMPREHENSIVE REPORT")
        print("="*60)
        
        # Basic statistics
        orig_boarding = original_df['boarding'].sum()
        orig_alighting = original_df['alighting'].sum()
        
        hybrid_boarding = hybrid_df['boarding'].sum()
        hybrid_alighting = hybrid_df['alighting'].sum()
        
        print(f"\n🔄 HYBRID PREDICTION SUMMARY:")
        print(f"  Original boarding: {orig_boarding:,.0f}")
        print(f"  Hybrid boarding:   {hybrid_boarding:,.0f}")
        print(f"  Original alighting: {orig_alighting:,.0f}")
        print(f"  Hybrid alighting:   {hybrid_alighting:,.0f}")
        print(f"  Boarding change:    {hybrid_boarding - orig_boarding:+,.0f}")
        print(f"  Alighting change:   {hybrid_alighting - orig_alighting:+,.0f}")
        
        # Special day analysis
        special_records = hybrid_df[hybrid_df['is_special_date'] == True]
        normal_records = hybrid_df[hybrid_df['is_special_date'] == False]
        
        print(f"\n📅 DATE-BASED ANALYSIS:")
        print(f"  Normal day records: {len(normal_records)} (LSTM-weighted)")
        print(f"  Special day records: {len(special_records)} (Prophet-weighted)")
        
        if len(special_records) > 0:
            special_boarding_avg = special_records['boarding'].mean()
            normal_boarding_avg = normal_records['boarding'].mean() if len(normal_records) > 0 else 0
            
            print(f"  Normal day avg boarding: {normal_boarding_avg:.2f}")
            print(f"  Special day avg boarding: {special_boarding_avg:.2f}")
            
            if special_boarding_avg < normal_boarding_avg:
                print("  📉 Special days show reduced ridership (expected)")
            else:
                print("  📈 Special days show increased ridership")
        
        # Model contribution analysis
        if 'lstm_weight' in hybrid_df.columns:
            avg_lstm_weight = hybrid_df['lstm_weight'].mean()
            avg_prophet_weight = hybrid_df['prophet_weight'].mean()
            
            print(f"\n⚖️  MODEL CONTRIBUTION:")
            print(f"  Average LSTM weight: {avg_lstm_weight:.3f}")
            print(f"  Average Prophet weight: {avg_prophet_weight:.3f}")
        
        # Constraint effectiveness
        overcrowded = (hybrid_df['occupancy_rate'] > 1.0).sum()
        severely_overcrowded = (hybrid_df['occupancy_rate'] > 1.5).sum()
        
        print(f"\n🚌 CONSTRAINT EFFECTIVENESS:")
        print(f"  >100% capacity: {overcrowded} records ({overcrowded/len(hybrid_df)*100:.1f}%)")
        print(f"  >150% capacity: {severely_overcrowded} records ({severely_overcrowded/len(hybrid_df)*100:.1f}%)")


def compare_all_models(historical_df, lstm_predictions, prophet_predictions, hybrid_predictions, save_dir=None):
    """Compare LSTM, Prophet, and Hybrid predictions with historical data"""
    
    # Prepare data
    hist_temp = historical_df.copy()
    hist_temp['datetime'] = pd.to_datetime(hist_temp['datetime'])
    hist_temp['hour'] = hist_temp['datetime'].dt.hour
    hist_temp['is_weekend'] = hist_temp['datetime'].dt.dayofweek >= 5
    
    models_data = {
        'Historical': hist_temp,
        'LSTM': lstm_predictions,
        'Prophet': prophet_predictions, 
        'Hybrid': hybrid_predictions
    }
    
    # Calculate hourly patterns for each model
    hourly_patterns = {}
    for model_name, data in models_data.items():
        temp_data = data.copy()
        temp_data['datetime'] = pd.to_datetime(temp_data['datetime'])
        temp_data['hour'] = temp_data['datetime'].dt.hour
        temp_data['is_weekend'] = temp_data['datetime'].dt.dayofweek >= 5
        
        weekday_data = temp_data[~temp_data['is_weekend']]
        if len(weekday_data) > 0:
            hourly_patterns[model_name] = weekday_data.groupby('hour')[['boarding', 'alighting']].mean()
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    hours = range(24)
    colors = {'Historical': 'black', 'LSTM': 'blue', 'Prophet': 'red', 'Hybrid': 'green'}
    styles = {'Historical': '-', 'LSTM': '--', 'Prophet': '-.', 'Hybrid': ':'}
    
    # Boarding comparison
    for model_name, pattern in hourly_patterns.items():
        boarding_values = [pattern.loc[h, 'boarding'] if h in pattern.index else 0 for h in hours]
        ax1.plot(hours, boarding_values, color=colors[model_name], linestyle=styles[model_name], 
                linewidth=2, label=model_name, marker='o' if model_name == 'Historical' else None)
    
    ax1.fill_between([7, 9], 0, max([max([pattern.loc[h, 'boarding'] if h in pattern.index else 0 for h in hours]) 
                                    for pattern in hourly_patterns.values()]) * 1.1, 
                    alpha=0.2, color='yellow', label='Morning Rush')
    ax1.fill_between([17, 19], 0, max([max([pattern.loc[h, 'boarding'] if h in pattern.index else 0 for h in hours]) 
                                      for pattern in hourly_patterns.values()]) * 1.1, 
                    alpha=0.2, color='orange', label='Evening Rush')
    ax1.set_title('Weekday Boarding Patterns Comparison (All Models)')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Average Passengers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Alighting comparison
    for model_name, pattern in hourly_patterns.items():
        alighting_values = [pattern.loc[h, 'alighting'] if h in pattern.index else 0 for h in hours]
        ax2.plot(hours, alighting_values, color=colors[model_name], linestyle=styles[model_name], 
                linewidth=2, label=model_name, marker='o' if model_name == 'Historical' else None)
    
    ax2.fill_between([7, 9], 0, max([max([pattern.loc[h, 'alighting'] if h in pattern.index else 0 for h in hours]) 
                                    for pattern in hourly_patterns.values()]) * 1.1, 
                    alpha=0.2, color='yellow')
    ax2.fill_between([17, 19], 0, max([max([pattern.loc[h, 'alighting'] if h in pattern.index else 0 for h in hours]) 
                                      for pattern in hourly_patterns.values()]) * 1.1, 
                    alpha=0.2, color='orange')
    ax2.set_title('Weekday Alighting Patterns Comparison (All Models)')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Average Passengers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Correlation analysis
    if 'Historical' in hourly_patterns:
        hist_boarding = [hourly_patterns['Historical'].loc[h, 'boarding'] if h in hourly_patterns['Historical'].index else 0 for h in hours]
        
        for model_name in ['LSTM', 'Prophet', 'Hybrid']:
            if model_name in hourly_patterns:
                model_boarding = [hourly_patterns[model_name].loc[h, 'boarding'] if h in hourly_patterns[model_name].index else 0 for h in hours]
                ax3.scatter(hist_boarding, model_boarding, alpha=0.7, label=f'{model_name}', color=colors[model_name])
        
        max_val = max(hist_boarding + [max([max([pattern.loc[h, 'boarding'] if h in pattern.index else 0 for h in hours]) 
                                           for pattern in hourly_patterns.values()])])
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Match')
        ax3.set_xlabel('Historical Average Boarding')
        ax3.set_ylabel('Predicted Average Boarding')
        ax3.set_title('Pattern Correlation with Historical Data')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Distribution comparison
    for model_name, data in models_data.items():
        if 'boarding' in data.columns:
            ax4.hist(data['boarding'], bins=30, alpha=0.4, label=f'{model_name}', 
                    color=colors[model_name], density=True)
    
    ax4.set_xlabel('Passenger Count')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Boarding Distribution Comparison')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'all_models_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plots saved to {save_dir}")
    else:
        plt.show()
    
    # Calculate and print correlations
    if 'Historical' in hourly_patterns:
        print(f"\nPattern Correlations with Historical Data:")
        hist_boarding = np.array([hourly_patterns['Historical'].loc[h, 'boarding'] if h in hourly_patterns['Historical'].index else 0 for h in hours])
        
        for model_name in ['LSTM', 'Prophet', 'Hybrid']:
            if model_name in hourly_patterns:
                model_boarding = np.array([hourly_patterns[model_name].loc[h, 'boarding'] if h in hourly_patterns[model_name].index else 0 for h in hours])
                correlation = np.corrcoef(hist_boarding, model_boarding)[0, 1] if len(hist_boarding) > 1 else 0
                print(f"  {model_name}: {correlation:.3f}")


def main():
    """Main function for hybrid model"""
    parser = argparse.ArgumentParser(description='Hybrid LSTM-Prophet Model for Transportation Prediction')
    parser.add_argument('--historical', required=True, help='Historical data CSV')
    parser.add_argument('--future', help='Future data CSV for predictions')
    parser.add_argument('--bus-stops', default='ankara_bus_stops.csv', help='Bus stops CSV file')
    parser.add_argument('--output', default='output_hybrid', help='Output directory')
    parser.add_argument('--sequence', type=int, default=48, help='LSTM sequence length')
    parser.add_argument('--epochs', type=int, default=60, help='LSTM training epochs')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--load-models', action='store_true', help='Load pre-trained models')
    parser.add_argument('--model-dir', default='hybrid_models', help='Model save/load directory')
    parser.add_argument('--disable-route-adjustments', action='store_true', help='Disable route-aware adjustments')
    parser.add_argument('--disable-realistic-constraints', action='store_true', help='Disable realistic boarding constraints')
    parser.add_argument('--disable-night-constraints', action='store_true', help='Disable night-time constraints')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    plots_dir = os.path.join(args.output, 'plots')
    data_dir = os.path.join(args.output, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Load historical data
        print(f"Loading historical data from {args.historical}")
        historical_df = pd.read_csv(args.historical)
        print(f"Loaded {len(historical_df)} historical records")
        
        # Create hybrid model
        hybrid_model = HybridModel(sequence_length=args.sequence)
        
        # Load bus stops data for both models
        if args.bus_stops and os.path.exists(args.bus_stops):
            hybrid_model.lstm_model._load_last_stops(args.bus_stops)
            hybrid_model.prophet_model._load_last_stops(args.bus_stops)
        else:
            print("Warning: Bus stops CSV not found. Last stop logic will be disabled.")
        
        # Load or train models
        if args.load_models:
            print(f"Attempting to load models from {args.model_dir}...")
            if hybrid_model.load_models(args.model_dir):
                print("✅ Models loaded successfully")
            else:
                print("❌ Failed to load models, will train new ones")
                args.load_models = False
        
        if not args.load_models:
            # Train hybrid models
            success = hybrid_model.train_hybrid_models(
                historical_df, 
                save_models=args.save_models,
                model_save_dir=args.model_dir
            )
            
            if not success:
                print("❌ Failed to train hybrid models")
                return
        
        # Make predictions if future data provided
        if args.future:
            print(f"\nLoading future data from {args.future}")
            future_df = pd.read_csv(args.future)
            print(f"Loaded {len(future_df)} future records")
            
            original_future_df = future_df.copy()
            
            # Configuration
            use_route_adjustments = not args.disable_route_adjustments
            use_realistic_constraints = not args.disable_realistic_constraints
            use_night_constraints = not args.disable_night_constraints
            
            print(f"\nHybrid model configuration:")
            print(f"  Route adjustments: {'ON' if use_route_adjustments else 'OFF'}")
            print(f"  Realistic constraints: {'ON' if use_realistic_constraints else 'OFF'}")
            print(f"  Night-time constraints (1-4 AM): {'ON' if use_night_constraints else 'OFF'}")
            
            # Make hybrid predictions
            hybrid_predictions = hybrid_model.predict_hybrid(
                historical_df,
                future_df,
                apply_route_adjustments=use_route_adjustments,
                apply_realistic_constraints=use_realistic_constraints,
                apply_night_constraints=use_night_constraints
            )
            
            if hybrid_predictions is not None:
                # Generate hybrid report
                hybrid_model.generate_hybrid_report(original_future_df, hybrid_predictions)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(data_dir, f"hybrid_predictions_{timestamp}.csv")
                hybrid_predictions.to_csv(output_file, index=False)
                print(f"\n💾 Hybrid predictions saved to {output_file}")
                
                # Also save individual model predictions for comparison
                if hasattr(hybrid_predictions, 'lstm_weight'):
                    lstm_only_file = os.path.join(data_dir, f"lstm_predictions_{timestamp}.csv")
                    prophet_only_file = os.path.join(data_dir, f"prophet_predictions_{timestamp}.csv")
                    
                    # Create LSTM-only predictions
                    lstm_only = hybrid_predictions.copy()
                    lstm_only['boarding'] = lstm_only['boarding_lstm']
                    lstm_only['alighting'] = lstm_only['alighting_lstm']
                    lstm_only.to_csv(lstm_only_file, index=False)
                    
                    # Create Prophet-only predictions
                    prophet_only = hybrid_predictions.copy()
                    prophet_only['boarding'] = prophet_only['boarding_prophet']
                    prophet_only['alighting'] = prophet_only['alighting_prophet']
                    prophet_only.to_csv(prophet_only_file, index=False)
                    
                    print(f"💾 Individual model predictions saved for comparison")
                    
                    # Create comprehensive comparison plot
                    compare_all_models(historical_df, lstm_only, prophet_only, hybrid_predictions, plots_dir)
                
                print("\n" + "="*60)
                print("HYBRID MODEL EXECUTION COMPLETED SUCCESSFULLY! 🎉")
                print("="*60)
            else:
                print("❌ Failed to generate hybrid predictions")
        else:
            print("\n✅ Hybrid model training completed. Use --future to make predictions.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()