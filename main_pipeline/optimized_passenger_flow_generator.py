#!/usr/bin/env python3
"""
Generate passenger flow data structure based on optimized schedule.
This creates the framework for LSTM+Prophet prediction by generating:
- Trip times and bus movements
- Stop sequences for each line
- Bus allocations 
- Empty passenger data columns (to be filled by prediction algorithm)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path


class OptimizedPassengerFlowGenerator:
    """Generate passenger flow structure based on optimized schedules"""
    
    def __init__(self, stops_file: str, project_root: Path):
        """Initialize with network data"""
        self.stops_file = stops_file
        self.project_root = project_root
        self.travel_times = {}  # Cache for travel times
        
        # Import simulation components directly from core
        core_path = str(project_root / "bus_simulation_pipeline" / "src" / "core")
        sys.path.insert(0, core_path)
        
        try:
            from data_models import BusTransitData
            from transit_network import TransitNetwork
        except ImportError as e:
            print(f"Failed to import simulation components: {e}")
            # Create minimal mock classes for testing
            class BusTransitData:
                def __init__(self):
                    self.stops = {}
                    self.lines = {}
            
            class TransitNetwork:
                def __init__(self, data):
                    self.data = data
                
                def load_network_data(self, stops_file):
                    # Load CSV and create basic network structure
                    try:
                        df = pd.read_csv(stops_file)
                        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                        
                        # Create stops
                        for _, row in df.iterrows():
                            stop_id = row['stop_id']
                            self.data.stops[stop_id] = {
                                'stop_id': stop_id,
                                'stop_name': row.get('stop_name', f'Stop {stop_id}'),
                                'line_id': row['line_id']
                            }
                        
                        # Create lines with stops
                        lines = {}
                        for _, row in df.iterrows():
                            line_id = row['line_id']
                            stop_id = row['stop_id']
                            if line_id not in lines:
                                lines[line_id] = {'stops': []}
                            if stop_id not in lines[line_id]['stops']:
                                lines[line_id]['stops'].append(stop_id)
                        
                        # Convert to simple objects
                        class Line:
                            def __init__(self, stops):
                                self.stops = sorted(stops)  # Sort stops for consistent ordering
                        
                        for line_id, line_data in lines.items():
                            self.data.lines[line_id] = Line(line_data['stops'])
                        
                        return True
                    except Exception as e:
                        print(f"Error loading network data: {e}")
                        return False
        
        self.data = BusTransitData()
        self.network = TransitNetwork(self.data)
        
        # Load network data
        if not self.network.load_network_data(stops_file):
            raise ValueError(f"Failed to load network data from {stops_file}")
            
        print(f"✅ Loaded {len(self.data.stops)} stops, {len(self.data.lines)} lines")
    
    def load_optimized_schedule(self, schedule_file: str) -> pd.DataFrame:
        """Load the optimized schedule from CSV"""
        df = pd.read_csv(schedule_file)
        df['departure_time'] = pd.to_datetime(df['departure_time'])
        return df
    
    def get_line_stops_sequence(self, line_id: str) -> List[int]:
        """Get the ordered sequence of stops for a line"""
        if line_id in self.data.lines:
            return self.data.lines[line_id].stops
        return []
    
    def estimate_travel_time(self, from_stop: int, to_stop: int, line_id: str, departure_time: datetime) -> float:
        """Estimate travel time between stops based on time of day"""
        # Use cached travel time if available
        cache_key = (from_stop, to_stop, line_id)
        if cache_key in self.travel_times:
            base_time = self.travel_times[cache_key]
        else:
            # Base travel time (2-5 minutes between stops)
            base_time = np.random.uniform(2.0, 5.0)
            self.travel_times[cache_key] = base_time
        
        # Adjust for time of day
        hour = departure_time.hour
        if 7 <= hour <= 9 or 16 <= hour <= 19:  # Rush hours
            return base_time * 1.8
        elif 22 <= hour or hour <= 5:  # Night time
            return base_time * 0.7
        else:  # Regular hours
            return base_time * 1.0
    
    def generate_trip_records(self, schedule_df: pd.DataFrame) -> List[Dict]:
        """Generate trip records for each departure following passenger flow format"""
        
        print("🚌 Generating trip records based on optimized schedule...")
        
        all_records = []
        trip_id = 1
        
        for _, departure in schedule_df.iterrows():
            line_id = departure['line_id']
            departure_time = departure['departure_time']
            fitness_score = departure.get('fitness_score', 0.0)
            
            # Generate unique bus ID for this trip
            bus_id = f"opt_bus_{line_id}_{departure_time.strftime('%H%M')}"
            capacity = 50  # Standard bus capacity
            
            # Get stops for this line
            stops_sequence = self.get_line_stops_sequence(line_id)
            if not stops_sequence:
                continue
            
            # Generate records for each stop on the route
            current_time = departure_time
            current_load = 0
            
            for stop_index, stop_id in enumerate(stops_sequence):
                # Create passenger flow record
                record = {
                    'datetime': current_time,
                    'hour': current_time.hour,
                    'minute': current_time.minute,
                    'date': current_time.date(),
                    'line_id': line_id,
                    'stop_id': stop_id,
                    'bus_id': bus_id,
                    
                    # Passenger data columns (empty for LSTM+Prophet prediction)
                    'boarding': 0,  # To be predicted
                    'alighting': 0,  # To be predicted  
                    'current_load': current_load,  # Will be updated by prediction
                    'new_load': current_load,  # Will be updated by prediction
                    'capacity': capacity,  # Bus capacity - needed by prediction model
                    'occupancy_rate': current_load / capacity,  # Will be updated by prediction
                    'actual_stop': True,  # All generated stops are actual stops
                    'status': '',  # Empty status
                    
                    # Trip metadata (for internal use)
                    'trip_id': trip_id,
                    'stop_sequence': stop_index + 1,
                    'total_stops': len(stops_sequence),
                    'fitness_score': fitness_score,
                    'source': 'optimized_schedule'
                }
                
                all_records.append(record)
                
                # Calculate time to next stop
                if stop_index < len(stops_sequence) - 1:
                    next_stop = stops_sequence[stop_index + 1]
                    travel_time = self.estimate_travel_time(stop_id, next_stop, line_id, current_time)
                    
                    # Add stop time (1-2 minutes for passenger exchange)
                    stop_time = 1.5
                    current_time += timedelta(minutes=travel_time + stop_time)
            
            trip_id += 1
            
            if trip_id % 50 == 0:
                print(f"   Generated {trip_id-1} trips...")
        
        print(f"✅ Generated {len(all_records)} trip records for {trip_id-1} trips")
        return all_records
    
    def generate_optimized_passenger_flow(self, 
                                        schedule_file: str,
                                        output_file: str):
        """Main function to generate optimized passenger flow structure"""
        
        print("🎯 GENERATING OPTIMIZED PASSENGER FLOW STRUCTURE")
        print("=" * 60)
        
        try:
            # Load data
            print("📄 Loading optimized schedule...")
            schedule_df = self.load_optimized_schedule(schedule_file)
            
            print(f"   Optimized departures: {len(schedule_df)}")
            print(f"   Lines: {schedule_df['line_id'].nunique()}")
            
            # Generate trip records
            trip_records = self.generate_trip_records(schedule_df)
            
            # Create DataFrame
            print("📊 Creating passenger flow structure DataFrame...")
            flow_df = pd.DataFrame(trip_records)
            
            # Sort by datetime and line
            flow_df = flow_df.sort_values(['datetime', 'line_id', 'stop_sequence'])
            
            # Save to CSV
            print(f"💾 Saving to {output_file}...")
            flow_df.to_csv(output_file, index=False)
            
            # Generate summary
            print(f"\n📈 GENERATION COMPLETE!")
            print("=" * 60)
            print(f"📄 Output file: {output_file}")
            print(f"📊 Total records: {len(flow_df):,}")
            print(f"🚌 Total trips: {flow_df['trip_id'].nunique():,}")
            print(f"🚏 Unique stops: {flow_df['stop_id'].nunique()}")
            print(f"🚌 Unique buses: {flow_df['bus_id'].nunique()}")
            
            # Time range
            start_time = flow_df['datetime'].min()
            end_time = flow_df['datetime'].max()
            print(f"⏰ Time range: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
            
            # Records by line
            print(f"\n📋 Records by line:")
            line_counts = flow_df['line_id'].value_counts().sort_index()
            for line_id, count in line_counts.items():
                print(f"   Line {line_id}: {count:,} records")
            
            # Sample records
            print(f"\n🔍 Sample records (first 5):")
            sample_cols = ['datetime', 'line_id', 'stop_id', 'bus_id', 'trip_id']
            for i, (_, row) in enumerate(flow_df[sample_cols].head().iterrows()):
                print(f"   {i+1}. {row['datetime'].strftime('%H:%M')} | Line {row['line_id']} | Stop {row['stop_id']} | Bus {row['bus_id']} | Trip {row['trip_id']}")
            
            print(f"\n✅ Ready for LSTM+Prophet passenger prediction!")
            print("   The following columns are empty and ready for prediction:")
            print("   • boarding, alighting, current_load, new_load, occupancy_rate")
            
            return flow_df
            
        except Exception as e:
            print(f"❌ Error generating passenger flow structure: {e}")
            import traceback
            traceback.print_exc()
            return None 