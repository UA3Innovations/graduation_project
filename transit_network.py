"""
Network topology modeling for the bus transit system.
Handles stops, lines, and travel times between stops.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from data_models import Stop, Line, TravelSegment, BusTransitData


class TransitNetwork:
    """
    Manages the transit network topology including stops, lines, and travel times.
    """
    
    # Constants for special locations and conditions - UPDATED to match actual data
    SPECIAL_LOCATIONS = [
        "Kızılay", "AŞTİ", "Aşti", "Ulus", "Sıhhiye", "Opera", "Gölbaşı",
        "Ankara Üniversitesi", "Atılım Üniversitesi", "Ufuk Üniversitesi", 
        "Gazi Hastanesi", "Teknokent", "Avm", "Taurus", "Samanyolu",
        "Emek", "Bahçelievler", "Dikmen", "İncek", "Balgat", 
        "Jandarma", "Belediye", "Kaymakamlık", "Patalya", "Aquapark",
        "Sayıştay", "Milli Kütüphane", "Karayolları", "Güvenpark"
    ]
    
    # Line classifications for passenger demand patterns - UPDATED to match actual data
    CROWDED_LINES = ['101', '101-1', '105-1', '105-2']  # Major lines with high demand
    UNDERUTILIZED_LINES = ['102-1', '102-2']  # Lines with lower demand
    UNSTABLE_LINES = ['103-1', '103-2', '104-1', '104-2']  # Lines with unpredictable patterns
    
    # Time periods - EXTENDED RUSH HOURS for overcrowding simulation
    RUSH_HOURS_MORNING = [(6, 10)]  # EXTENDED morning rush: 6-10 AM
    RUSH_HOURS_EVENING = [(16, 20)]  # EXTENDED evening rush: 4-8 PM
    OPERATIONAL_HOURS = (5, 24)  # 5:00-24:00 (0:00)
    
    def __init__(self, data: BusTransitData):
        """Initialize the transit network."""
        self.data = data
        
        # Register constants with the data container
        self.data.special_locations = set(self.SPECIAL_LOCATIONS)
        self.data.crowded_lines = set(self.CROWDED_LINES)
        self.data.underutilized_lines = set(self.UNDERUTILIZED_LINES)
        self.data.unstable_lines = set(self.UNSTABLE_LINES)
    
    def load_network_data(self, stops_file: str) -> None:
        """
        Load transit network data from files.
        
        Parameters:
        -----------
        stops_file : str
            Path to CSV file with stop and line information.
        """
        try:
            # Load the CSV data
            df = pd.read_csv(stops_file)
            
            # Check required columns
            required_columns = ['Line ID', 'Stop ID', 'Stop Name']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV.")
            
            # Convert line IDs to strings for consistency
            df['Line ID'] = df['Line ID'].astype(str)
            
            # Process stops
            self._process_stops(df)
            
            # Process lines
            self._process_lines(df)
            
            # Generate travel times between stops
            self._generate_travel_times()
            
            # Initialize passenger profiles
            self._initialize_passenger_profiles()
            
            return True
            
        except Exception as e:
            print(f"Error loading network data: {str(e)}")
            return False
    
    def _process_stops(self, df: pd.DataFrame) -> None:
        """Process stop information from the dataframe."""
        # Extract unique stops
        stops_df = df[['Stop ID', 'Stop Name']].drop_duplicates().reset_index(drop=True)
        
        self.data.stops = {}
        for _, row in stops_df.iterrows():
            stop_id = row['Stop ID']
            stop_name = row['Stop Name']
            
            is_special = any(location in stop_name for location in self.SPECIAL_LOCATIONS)
            
            # Create stop object
            self.data.stops[stop_id] = Stop(
                stop_id=stop_id,
                stop_name=stop_name,
                is_special_location=is_special
            )
    
    def _process_lines(self, df: pd.DataFrame) -> None:
        """Process line information from the dataframe."""
        # Get unique line IDs
        line_ids = df['Line ID'].unique()
        
        self.data.lines = {}
        for line_id in line_ids:
            # Get stops for this line in order
            line_stops = df[df['Line ID'] == line_id][['Stop ID']].values.flatten().tolist()
            
            # Determine line classification and popularity factor - INCREASED for overcrowding
            if line_id in self.CROWDED_LINES:
                is_crowded = True
                popularity = random.uniform(2.0, 3.0)  # INCREASED from 1.2-1.5 for heavy overcrowding
            elif line_id in self.UNDERUTILIZED_LINES:
                is_crowded = False
                popularity = random.uniform(0.8, 1.2)  # INCREASED from 0.6-0.8 for more balanced simulation
            elif line_id in self.UNSTABLE_LINES:
                is_crowded = True if random.random() < 0.6 else False  # More often crowded
                popularity = random.uniform(1.3, 1.8)  # INCREASED from 0.9-1.2 for more demand
            else:
                is_crowded = False
                popularity = random.uniform(1.1, 1.4)  # INCREASED from 0.9-1.1 for higher baseline
            
            # Create line object
            self.data.lines[line_id] = Line(
                line_id=line_id,
                stops=line_stops,
                is_crowded=is_crowded,
                popularity_factor=popularity
            )
    
    def _generate_travel_times(self) -> None:
        """Generate travel times between consecutive stops for each line."""
        self.data.travel_segments = {}
        
        for line_id, line in self.data.lines.items():
            stops = line.stops
            self.data.travel_segments[line_id] = {}
            
            # Determine if this is a major/popular line
            is_crowded_line = line.is_crowded
            
            for i in range(len(stops) - 1):
                from_stop_id = stops[i]
                to_stop_id = stops[i + 1]
                
                from_stop_name = self.data.stops[from_stop_id].stop_name
                to_stop_name = self.data.stops[to_stop_id].stop_name
                
                # Special case for high-traffic areas
                high_traffic_area = any(loc in from_stop_name or loc in to_stop_name 
                                       for loc in self.SPECIAL_LOCATIONS)
                
                # Base travel time: 2-6 minutes between stops
                # Longer for crowded lines and high traffic areas
                base_multiplier = 1.2 if is_crowded_line else 1.0
                base_multiplier *= 1.3 if high_traffic_area else 1.0
                
                base_time = random.randint(2, 5) * base_multiplier
                
                # Create travel segment
                self.data.travel_segments[line_id][(from_stop_id, to_stop_id)] = TravelSegment(
                    from_stop_id=from_stop_id,
                    to_stop_id=to_stop_id,
                    line_id=line_id,
                    base_time=base_time,
                    rush_hour_factor=1.8,
                    weekend_factor=0.8,
                    night_factor=0.7,
                    holiday_factor=0.8,
                    bad_weather_factor=2.0
                )
    
    def _initialize_passenger_profiles(self) -> None:
        """Initialize passenger flow characteristics for each stop."""
        for stop_id, stop in self.data.stops.items():
            # Count how many lines serve this stop
            lines_serving_stop = 0
            for line in self.data.lines.values():
                if stop_id in line.stops:
                    lines_serving_stop += 1
            
            # Base boarding rate (passengers per hour) - MASSIVELY INCREASED for overcrowding testing
            # Extreme values to create heavy overcrowding conditions
            if stop.is_special_location:
                # Important locations like malls, transit hubs, universities
                if any(loc in stop.stop_name for loc in ["Kızılay", "AŞTİ", "Aşti", "Opera"]):
                    # Major transit hubs - DRAMATICALLY INCREASED for overcrowding
                    base_boarding = random.randint(80, 150)  # INCREASED from 25-45 for heavy overcrowding
                    # Higher transfer rates at major transit hubs
                    transfer_likelihood = random.uniform(0.5, 0.7)
                elif any(loc in stop.stop_name for loc in ["Gölbaşı", "Ankara Üniversitesi", "Atılım Üniversitesi", "Ufuk Üniversitesi"]):
                    # Universities - INCREASED for overcrowding
                    base_boarding = random.randint(60, 120)  # INCREASED from 18-35 for overcrowding
                    transfer_likelihood = random.uniform(0.3, 0.45)
                elif any(loc in stop.stop_name for loc in ["Gazi Hastanesi", "Teknokent", "Avm", "Taurus"]):
                    # Malls and facilities - INCREASED for overcrowding
                    base_boarding = random.randint(50, 100)  # INCREASED from 15-30 for overcrowding
                    transfer_likelihood = random.uniform(0.35, 0.55)
                else:
                    # Other special locations - INCREASED for overcrowding
                    base_boarding = random.randint(40, 80)  # INCREASED from 12-25 for overcrowding
                    transfer_likelihood = random.uniform(0.35, 0.5)
            else:
                # Regular stops - INCREASED for overcrowding
                if lines_serving_stop > 1:
                    # Stops served by multiple lines
                    base_boarding = random.randint(30, 60)  # INCREASED from 10-20 for overcrowding
                    transfer_likelihood = random.uniform(0.2, 0.35)
                else:
                    # Single line stops - INCREASED for overcrowding
                    base_boarding = random.randint(20, 40)   # INCREASED from 8-18 for overcrowding
                    transfer_likelihood = random.uniform(0.05, 0.15)
            
            # Base alighting rate as a percentage of boarding
            base_alighting_ratio = random.uniform(0.8, 1.2)  # Can be more or less than boarding
            
            # Transfer probability bonus for multiple lines
            if lines_serving_stop > 1:
                # More lines means better connectivity and more transfers
                transfer_bonus = 0.05 * min(lines_serving_stop, 4)  # Up to 20% bonus for 4+ lines
                transfer_likelihood += transfer_bonus
            
            # Add special case transfer probabilities
            if any(loc in stop.stop_name for loc in ["Kızılay", "Ulus", "Opera", "Sıhhiye"]):
                # Central transit hubs have extremely high transfer rates
                transfer_likelihood = max(transfer_likelihood, random.uniform(0.65, 0.8))
            elif lines_serving_stop == 2 and not stop.is_special_location:
                # Two regular lines crossing has moderate transfer potential
                transfer_likelihood = min(transfer_likelihood, random.uniform(0.2, 0.3))
            
            # Update stop with these characteristics
            stop.base_boarding_rate = base_boarding
            stop.base_alighting_ratio = base_alighting_ratio
            stop.transfer_likelihood = min(transfer_likelihood, 0.8)  # Cap at 80%
    
    def calculate_travel_time(self, line_id: str, from_stop_id: int, to_stop_id: int, 
                             dt: datetime, include_random: bool = True) -> float:
        """
        Calculate travel time between two consecutive stops based on conditions.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        from_stop_id : int
            ID of the departure stop.
        to_stop_id : int
            ID of the arrival stop.
        dt : datetime
            The datetime to calculate travel time for.
        include_random : bool
            Whether to include random variations in travel time.
            
        Returns:
        --------
        float
            Travel time in minutes.
        """
        # Get the travel segment
        segment = self.data.travel_segments.get(line_id, {}).get((from_stop_id, to_stop_id))
        
        if segment is None:
            # If not found, estimate a reasonable travel time
            return 5.0  # Default 5 minutes
        
        # Get time conditions
        hour = dt.hour
        minute = dt.minute
        weekday = dt.weekday()
        date_str = f"{dt.month:02d}-{dt.day:02d}"
        is_holiday = date_str in self.data.holidays
        
        # Determine base time-of-day factor
        if is_holiday:
            factor = segment.holiday_factor
        elif weekday >= 5:  # Weekend
            factor = segment.weekend_factor
        elif any(start <= hour < end for start, end in self.RUSH_HOURS_MORNING) or \
             any(start <= hour < end for start, end in self.RUSH_HOURS_EVENING):
            if weekday < 5:  # Weekday rush hour
                # Add more granular rush hour handling
                if 7 <= hour < 8:
                    # Early rush - building up
                    factor = segment.rush_hour_factor * 0.9
                elif 8 <= hour < 9:
                    # Peak rush - worst traffic
                    factor = segment.rush_hour_factor * 1.1
                    # Even more delay at peak morning rush minute (8:30)
                    if 25 <= minute <= 35:
                        factor *= 1.1
                elif 17 <= hour < 18:
                    # Peak evening rush
                    factor = segment.rush_hour_factor * 1.05
                    # Worst at 17:45
                    if 40 <= minute <= 50:
                        factor *= 1.08
                else:
                    factor = segment.rush_hour_factor
            else:
                factor = 1.0  # Normal on weekend
        elif hour < 6 or hour >= 22:  # Late night
            factor = segment.night_factor
        elif 12 <= hour < 13:  # Lunch hour
            # Slight increase during lunch hour
            factor = 1.1
        else:
            factor = 1.0  # Normal
            
        # Apply day-of-week adjustments for non-weekend days
        if weekday < 5 and not is_holiday:
            if weekday == 0:  # Monday
                factor *= 1.05  # Slightly worse traffic on Mondays
            elif weekday == 4:  # Friday
                if hour >= 15:  # Friday afternoon/evening
                    factor *= 1.1  # Worse traffic Friday afternoons
        
        # Apply random weather effects occasionally - use config value
        weather_effect_probability = getattr(self.data, 'weather_effects_probability', 0.15)
        if include_random and random.random() < weather_effect_probability:  # Use config value instead of hardcoded 0.15
            # Blend with bad weather effect
            base_effect = segment.base_time * factor
            weather_effect = segment.base_time * segment.bad_weather_factor
            travel_time = (base_effect + weather_effect) / 2
        else:
            travel_time = segment.base_time * factor
        
        # Add randomness for traffic variation if requested
        if include_random:
            # More variability during rush hours and on crowded lines
            if is_holiday:
                # Less variability on holidays
                travel_time *= random.uniform(0.92, 1.15)
            elif factor > 1.2 and line_id in self.CROWDED_LINES:
                # High variability for crowded lines in rush hour
                travel_time *= random.uniform(0.85, 1.3)
            elif factor > 1.2:
                # Moderate variability for regular lines in rush hour
                travel_time *= random.uniform(0.88, 1.25)
            elif weekday >= 5:
                # Less variability on weekends
                travel_time *= random.uniform(0.92, 1.15)
            else:
                # Normal variability
                travel_time *= random.uniform(0.9, 1.2)
                
            # Add occasional traffic incidents (accidents, construction, etc.)
            if random.random() < 0.03:  # 3% chance of traffic incident
                severity = random.uniform(1.2, 2.0)  # Severity multiplier
                travel_time *= severity
        
        return max(1.0, travel_time)  # Minimum 1 minute
    
    def calculate_dwell_time(self, stop_id: int, dt: datetime, 
                           boarding: int, alighting: int) -> float:
        """
        Calculate dwell time at a stop based on conditions and passenger activity.
        
        Parameters:
        -----------
        stop_id : int
            ID of the stop.
        dt : datetime
            Current datetime.
        boarding : int
            Number of boarding passengers.
        alighting : int
            Number of alighting passengers.
            
        Returns:
        --------
        float
            Dwell time in minutes.
        """
        stop = self.data.stops.get(stop_id)
        if not stop:
            return 0.5  # Default minimum dwell time
        
        # Base time for door operations (opening/closing) - 20 seconds
        base_time = 0.33  # in minutes
        
        # If no passenger activity, use minimum dwell time
        if boarding == 0 and alighting == 0:
            # Very brief stop - simply opening doors and closing them
            if stop.is_special_location:
                return max(0.33, base_time)  # At least 20 seconds at major stops
            else:
                return max(0.25, base_time)  # At least 15 seconds at regular stops
        
        # Time for passenger boarding/alighting
        # Boarding takes longer than alighting (3 seconds vs 2 seconds per passenger)
        passenger_time = (boarding * 3 + alighting * 2) / 60.0  # Convert seconds to minutes
        
        # Account for crowding effects during rush hours
        hour = dt.hour
        weekday = dt.weekday()
        is_rush_hour = ((7 <= hour < 9) or (16 <= hour < 19)) and weekday < 5
        
        # Apply congestion factor based on stop type and time of day
        if stop.is_special_location and is_rush_hour:
            congestion_factor = 1.3  # 30% longer at major stops during rush hour
        elif stop.is_special_location:
            congestion_factor = 1.15  # 15% longer at major stops during off-peak
        elif is_rush_hour:
            congestion_factor = 1.2  # 20% longer at regular stops during rush hour
        else:
            congestion_factor = 1.0  # No adjustment during off-peak at regular stops
            
        # Total dwell time calculation
        dwell_time = (base_time + passenger_time) * congestion_factor
        
        # Additional time for very high passenger volume (reflects bottlenecks)
        # This accounts for crowding effects when many people board/alight simultaneously
        if boarding + alighting > 40:
            dwell_time += 0.5  # Add 30 seconds for very busy stops
        elif boarding + alighting > 20:
            dwell_time += 0.25  # Add 15 seconds for moderately busy stops
            
        # Special case for wheelchair/accessibility needs (random occurrence)
        # This simulates the occasional need for ramp deployment
        if random.random() < 0.05:  # 5% chance of wheelchair boarding/alighting
            dwell_time += 1.0  # Add 1 minute for accessibility accommodation
            
        # Ensure minimum dwell time
        return max(0.33, dwell_time)  # At least 20 seconds at any stop
    
    def get_route_travel_time(self, line_id: str, dt: datetime, 
                             include_dwell: bool = True) -> float:
        """
        Calculate the total travel time for a complete route.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        dt : datetime
            Departure time.
        include_dwell : bool
            Whether to include dwell time at stops.
            
        Returns:
        --------
        float
            Estimated travel time in minutes.
        """
        line = self.data.lines.get(line_id)
        if not line:
            return 60.0  # Default 1 hour if line not found
        
        stops = line.stops
        total_time = 0.0
        current_time = dt
        
        for i in range(len(stops) - 1):
            from_stop = stops[i]
            to_stop = stops[i + 1]
            
            # Calculate travel time for this segment
            segment_time = self.calculate_travel_time(line_id, from_stop, to_stop, current_time)
            total_time += segment_time
            
            # Update current time for next segment
            current_time += timedelta(minutes=segment_time)
            
            # Add dwell time at destination stop if requested
            if include_dwell:
                # Estimate passenger activity (this is simplified)
                stop_obj = self.data.stops.get(to_stop)
                is_rush = any(start <= current_time.hour < end for start, end in self.RUSH_HOURS_MORNING) or \
                         any(start <= current_time.hour < end for start, end in self.RUSH_HOURS_EVENING)
                
                est_boarding = 10 if is_rush else 5
                est_alighting = 10 if is_rush else 5
                
                if stop_obj and stop_obj.is_special_location:
                    est_boarding *= 2
                    est_alighting *= 2
                
                dwell_time = self.calculate_dwell_time(to_stop, current_time, est_boarding, est_alighting)
                total_time += dwell_time
                current_time += timedelta(minutes=dwell_time)
        
        # For one-way routes, double the time for return journey
        return total_time * 2
