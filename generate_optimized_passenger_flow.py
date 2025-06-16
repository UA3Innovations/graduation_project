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

# Import simulation components
from data_models import BusTransitData, Line, Stop, Bus
from transit_network import TransitNetwork


class OptimizedPassengerFlowGenerator:
    """Generate passenger flow structure based on optimized schedules"""
    
    def __init__(self, stops_file: str = "ankara_bus_stops_10.csv"):
        """Initialize with network data"""
        self.stops_file = stops_file
        self.data = BusTransitData()
        self.network = TransitNetwork(self.data)
        self.travel_times = {}  # Cache for travel times
        self.baseline_metrics = None
        
        # Load network data
        if not self.network.load_network_data(stops_file):
            raise ValueError(f"Failed to load network data from {stops_file}")
            
        print(f"✅ Loaded {len(self.data.stops)} stops, {len(self.data.lines)} lines")
    
    def load_optimized_schedule(self, schedule_file: str) -> pd.DataFrame:
        """Load the optimized schedule from CSV"""
        df = pd.read_csv(schedule_file)
        df['departure_time'] = pd.to_datetime(df['departure_time'])
        return df
    
    def load_bus_assignments(self, buses_file: str) -> pd.DataFrame:
        """Load bus data for assignments"""
        return pd.read_csv(buses_file)
    
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
    
    def load_baseline_metrics(self, simulation_results_file: str):
        """Load baseline metrics from Stage 1 simulation"""
        import pandas as pd
        
        print(f"📊 Loading baseline metrics from: {simulation_results_file}")
        df = pd.read_csv(simulation_results_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calculate daily averages
        total_days = df['date'].nunique()
        
        self.baseline_metrics = {
            'daily_boarding': df['boarding'].sum() / total_days,
            'daily_alighting': df['alighting'].sum() / total_days,
            'hourly_boarding_dist': df.groupby('hour')['boarding'].sum().to_dict(),
            'hourly_alighting_dist': df.groupby('hour')['alighting'].sum().to_dict(),
            'total_days': total_days,
            'avg_occupancy_by_hour': df.groupby('hour')['occupancy_rate'].mean().to_dict()
        }
        
        boarding_total = sum(self.baseline_metrics['hourly_boarding_dist'].values())
        alighting_total = sum(self.baseline_metrics['hourly_alighting_dist'].values())
        
        # Normalize to ratios
        self.baseline_metrics['hourly_boarding_ratios'] = {
            h: count/boarding_total if boarding_total > 0 else 0 
            for h, count in self.baseline_metrics['hourly_boarding_dist'].items()
        }
        self.baseline_metrics['hourly_alighting_ratios'] = {
            h: count/alighting_total if alighting_total > 0 else 0
            for h, count in self.baseline_metrics['hourly_alighting_dist'].items()
        }
        
        print(f"✅ Baseline metrics loaded:")
        print(f"   Daily boarding: {self.baseline_metrics['daily_boarding']:,.0f}")
        print(f"   Daily alighting: {self.baseline_metrics['daily_alighting']:,.0f}")
        
    def generate_optimized_passenger_flow_with_constraints(self, 
                                                         schedule_file: str,
                                                         buses_file: str,
                                                         baseline_file: str,
                                                         output_file: str):
        """Generate passenger flow structure preserving baseline volumes"""
        
        print("🎯 GENERATING OPTIMIZED PASSENGER FLOW WITH BASELINE PRESERVATION")
        print("=" * 70)
        
        # Load baseline constraints
        self.load_baseline_metrics(baseline_file)
        
        # Load schedule and buses (keep your existing methods)
        schedule_df = self.load_optimized_schedule(schedule_file)
        buses_df = self.load_bus_assignments(buses_file)
        
        print(f"   Optimized departures: {len(schedule_df)}")
        print(f"   Available buses: {len(buses_df)}")
        print(f"   Lines: {schedule_df['line_id'].nunique()}")
        
        # Generate initial structure
        trip_records = self.generate_trip_records(schedule_df, buses_df)
        flow_df = pd.DataFrame(trip_records)
        
        # CRITICAL: Apply baseline preservation before hybrid model
        flow_df = self._apply_baseline_preservation(flow_df)
        
        # Sort by datetime and line
        flow_df = flow_df.sort_values(['datetime', 'line_id', 'stop_sequence'])
        
        # Save for hybrid model
        flow_df.to_csv(output_file, index=False)
        
        # Validation
        self._validate_baseline_preservation(flow_df)
        
        print(f"✅ Generated {len(flow_df)} records with baseline preservation")
        return flow_df
    
    def _apply_baseline_preservation(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """Apply baseline preservation to maintain historical volumes"""
        
        print("🔄 Applying baseline preservation...")
        
        # For initial structure, set placeholder values that maintain ratios
        target_daily_boarding = self.baseline_metrics['daily_boarding']
        target_daily_alighting = self.baseline_metrics['daily_alighting']
        
        # Distribute boarding/alighting based on historical hourly patterns
        for hour in range(24):
            hour_mask = flow_df['hour'] == hour
            hour_records = flow_df[hour_mask]
            
            if len(hour_records) == 0:
                continue
            
            # Calculate target for this hour
            boarding_ratio = self.baseline_metrics['hourly_boarding_ratios'].get(hour, 0)
            alighting_ratio = self.baseline_metrics['hourly_alighting_ratios'].get(hour, 0)
            
            hour_target_boarding = target_daily_boarding * boarding_ratio
            hour_target_alighting = target_daily_alighting * alighting_ratio
            
            # Distribute across stops in this hour
            if len(hour_records) > 0:
                avg_boarding_per_stop = hour_target_boarding / len(hour_records)
                avg_alighting_per_stop = hour_target_alighting / len(hour_records)
                
                # Add some variation based on stop type using numpy random
                import numpy as np
                flow_df.loc[hour_mask, 'boarding'] = np.random.poisson(
                    max(0.1, avg_boarding_per_stop), len(hour_records)
                )
                flow_df.loc[hour_mask, 'alighting'] = np.random.poisson(
                    max(0.1, avg_alighting_per_stop), len(hour_records)
                )
        
        # Ensure non-negative values
        flow_df['boarding'] = flow_df['boarding'].fillna(0).clip(lower=0)
        flow_df['alighting'] = flow_df['alighting'].fillna(0).clip(lower=0)
        
        # Calculate initial loads
        flow_df = self._calculate_loads_and_occupancy(flow_df)
        
        return flow_df
    
    def _calculate_loads_and_occupancy(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate passenger loads and occupancy rates"""
        flow_df = flow_df.sort_values(['trip_id', 'stop_sequence'])
        
        for trip_id in flow_df['trip_id'].unique():
            trip_mask = flow_df['trip_id'] == trip_id
            trip_data = flow_df[trip_mask].sort_values('stop_sequence')
            
            current_load = 0
            for idx in trip_data.index:
                # Passengers alighting
                alighting = flow_df.loc[idx, 'alighting']
                current_load = max(0, current_load - alighting)
                flow_df.loc[idx, 'current_load'] = current_load
                
                # Passengers boarding
                boarding = flow_df.loc[idx, 'boarding']
                current_load += boarding
                flow_df.loc[idx, 'new_load'] = current_load
                
                # Occupancy rate
                capacity = flow_df.loc[idx, 'capacity']
                if capacity > 0:
                    flow_df.loc[idx, 'occupancy_rate'] = current_load / capacity
                else:
                    flow_df.loc[idx, 'occupancy_rate'] = 0.0
        
        return flow_df
    
    def _validate_baseline_preservation(self, flow_df: pd.DataFrame):
        """Validate that baseline metrics are preserved"""
        
        actual_boarding = flow_df['boarding'].sum()
        actual_alighting = flow_df['alighting'].sum()
        
        target_boarding = self.baseline_metrics['daily_boarding']
        target_alighting = self.baseline_metrics['daily_alighting']
        
        boarding_error = abs(actual_boarding - target_boarding) / target_boarding * 100 if target_boarding > 0 else 0
        alighting_error = abs(actual_alighting - target_alighting) / target_alighting * 100 if target_alighting > 0 else 0
        
        print(f"\n📊 BASELINE PRESERVATION VALIDATION:")
        print(f"   Target boarding: {target_boarding:,.0f}")
        print(f"   Actual boarding: {actual_boarding:,.0f}")
        print(f"   Boarding error: {boarding_error:.1f}%")
        print(f"   Target alighting: {target_alighting:,.0f}")
        print(f"   Actual alighting: {actual_alighting:,.0f}")
        print(f"   Alighting error: {alighting_error:.1f}%")
        
        if boarding_error < 10 and alighting_error < 10:
            print("   ✅ Baseline preservation ACCEPTABLE (<10% error)")
        else:
            print("   ⚠️ Baseline preservation needs improvement (>10% error)")
            
    
    def assign_buses_to_departures(self, schedule_df: pd.DataFrame, buses_df: pd.DataFrame) -> Dict[str, str]:
        """Create bus assignments for each departure"""
        # Group buses by their assigned lines
        line_buses = {}
        for _, bus in buses_df.iterrows():
            line_id = bus['current_line']
            if line_id not in line_buses:
                line_buses[line_id] = []
            line_buses[line_id].append({
                'bus_id': bus['bus_id'],
                'capacity': bus['capacity'],
                'bus_type': bus['bus_type']
            })
        
        # Assign buses to departures using round-robin
        departure_bus_map = {}
        line_bus_counters = {line_id: 0 for line_id in line_buses.keys()}
        
        for _, departure in schedule_df.iterrows():
            line_id = departure['line_id']
            departure_key = f"{line_id}_{departure['departure_time']}"
            
            if line_id in line_buses and line_buses[line_id]:
                # Round-robin assignment
                bus_index = line_bus_counters[line_id] % len(line_buses[line_id])
                assigned_bus = line_buses[line_id][bus_index]
                departure_bus_map[departure_key] = assigned_bus
                line_bus_counters[line_id] += 1
            else:
                print(f"⚠️ No buses available for line {line_id}")
        
        return departure_bus_map
    
    def generate_trip_records(self, schedule_df: pd.DataFrame, buses_df: pd.DataFrame) -> List[Dict]:
        """Generate trip records for each departure following passenger flow format"""
        
        print("🚌 Generating trip records based on optimized schedule...")
        
        # Get bus assignments
        bus_assignments = self.assign_buses_to_departures(schedule_df, buses_df)
        
        all_records = []
        trip_id = 1
        
        for _, departure in schedule_df.iterrows():
            line_id = departure['line_id']
            departure_time = departure['departure_time']
            departure_key = f"{line_id}_{departure_time}"
            
            # Get assigned bus
            if departure_key not in bus_assignments:
                continue
                
            bus_info = bus_assignments[departure_key]
            bus_id = bus_info['bus_id']
            capacity = bus_info['capacity']
            
            # Get stops for this line
            stops_sequence = self.get_line_stops_sequence(line_id)
            if not stops_sequence:
                continue
            
            # Generate records for each stop on the route
            current_time = departure_time
            
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
                    'boarding': None,  # To be predicted
                    'alighting': None,  # To be predicted  
                    'current_load': None,  # To be predicted
                    'new_load': None,  # To be predicted
                    'capacity': capacity,
                    'occupancy_rate': None,  # To be predicted
                    
                    # Trip metadata
                    'actual_stop': True,
                    'status': '',
                    'trip_id': trip_id,
                    'stop_sequence': stop_index + 1,
                    'total_stops': len(stops_sequence),
                    'fitness_score': departure['fitness_score']
                }
                
                all_records.append(record)
                
                # Calculate time to next stop
                if stop_index < len(stops_sequence) - 1:
                    next_stop = stops_sequence[stop_index + 1]
                    travel_time = self.estimate_travel_time(stop_id, next_stop, line_id, current_time)
                    current_time += timedelta(minutes=travel_time)
            
            trip_id += 1
            
            if trip_id % 50 == 0:
                print(f"   Generated {trip_id-1} trips...")
        
        print(f"✅ Generated {len(all_records)} trip records for {trip_id-1} trips")
        return all_records
    
    def generate_optimized_passenger_flow(self, 
                                        schedule_file: str = "../output/all_lines_optimized_schedule.csv",
                                        buses_file: str = "../output/buses.csv",
                                        output_file: str = "../output/optimized_passenger_flow_structure.csv"):
        """Main function to generate optimized passenger flow structure"""
        
        print("🎯 GENERATING OPTIMIZED PASSENGER FLOW STRUCTURE")
        print("=" * 60)
        
        try:
            # Load data
            print("📄 Loading optimized schedule and bus data...")
            schedule_df = self.load_optimized_schedule(schedule_file)
            buses_df = self.load_bus_assignments(buses_file)
            
            print(f"   Optimized departures: {len(schedule_df)}")
            print(f"   Available buses: {len(buses_df)}")
            print(f"   Lines: {schedule_df['line_id'].nunique()}")
            
            # Generate trip records
            trip_records = self.generate_trip_records(schedule_df, buses_df)
            
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
            sample_cols = ['datetime', 'line_id', 'stop_id', 'bus_id', 'capacity', 'trip_id']
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


def main():
    """Main execution function"""
    
    # Create generator
    generator = OptimizedPassengerFlowGenerator("ankara_bus_stops_10.csv")
    
    # Generate optimized passenger flow structure
    result = generator.generate_optimized_passenger_flow()
    
    if result is not None:
        print("\n🎉 Optimized passenger flow structure generated successfully!")
        return result
    else:
        print("\n❌ Failed to generate optimized passenger flow structure")
        return None


if __name__ == "__main__":
    main() 