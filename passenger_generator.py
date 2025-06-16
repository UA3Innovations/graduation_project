"""
Passenger demand modeling module for generating realistic passenger flows.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from data_models import BusTransitData, PassengerGroup, Stop, Bus
from transit_network import TransitNetwork


class PassengerGenerator:
    """
    Generates realistic passenger demand patterns for bus stops and lines.
    """
    
    # Seasons and their impact on ridership
    SEASONS = {
        "Winter": [(12, 1, 2), 0.85],  # Lower ridership in winter
        "Spring": [(3, 4, 5), 1.0],
        "Summer": [(6, 7, 8), 0.75],   # Lower ridership in summer (vacations)
        "Fall": [(9, 10, 11), 1.0]
    }
    
    def __init__(self, data: BusTransitData, network: TransitNetwork):
        """Initialize the passenger generator."""
        self.data = data
        self.network = network
        
        # Initialize waiting passengers with uneven distribution
        self.data.waiting_passengers = {}
        
        # Add initial waiting passengers unevenly across stops
        for stop_id, stop in self.data.stops.items():
            # Determine which lines serve this stop
            lines_serving_stop = []
            for line_id, line in self.data.lines.items():
                if stop_id in line.stops:
                    lines_serving_stop.append(line_id)
            
            # Determine initial waiting passenger count based on stop importance - SIGNIFICANTLY INCREASED
            if stop.is_special_location:
                base_waiting = random.randint(35, 80)  # INCREASED from 15-45 for overcrowding
            else:
                base_waiting = random.randint(15, 40)   # INCREASED from 5-20 for overcrowding
                
            # Adjust for lines serving this stop
            has_crowded_line = any(line_id in self.network.CROWDED_LINES for line_id in lines_serving_stop)
            has_underutilized_line = any(line_id in self.network.UNDERUTILIZED_LINES for line_id in lines_serving_stop)
            
            if has_crowded_line:
                base_waiting *= random.uniform(3.0, 4.5)  # INCREASED from 2.0-3.0 for heavy overcrowding
            elif has_underutilized_line:
                base_waiting *= random.uniform(0.4, 0.8)  # Increased from 0.3-0.7
                
            # Set initial waiting passengers
            self.data.waiting_passengers[stop_id] = int(base_waiting)
         
        # Initialize line-specific waiting passengers
        self.data.line_waiting_passengers = {}
        
        # Distribute passengers among lines at transfer points
        for stop_id, stop in self.data.stops.items():
            # Get lines at this stop
            lines_at_stop = []
            for line_id, line in self.data.lines.items():
                if stop_id in line.stops:
                    lines_at_stop.append((line_id, line.popularity_factor))
                    
            if stop.is_special_location or len(lines_at_stop) > 1:
                # Distribute waiting passengers by line popularity
                waiting = self.data.waiting_passengers[stop_id]
                self.data.waiting_passengers[stop_id] = 0  # Reset general waiting
                
                total_popularity = sum(pop for _, pop in lines_at_stop)
                for line_id, popularity in lines_at_stop:
                    if total_popularity > 0:
                        line_share = int(waiting * (popularity / total_popularity))
                    else:
                        line_share = int(waiting / len(lines_at_stop))
                    
                    # Set line-specific waiting
                    key = (stop_id, line_id)
                    self.data.line_waiting_passengers[key] = line_share
                
        # Initialize passenger flow patterns
        self.passenger_flow_patterns = {}
        
    def get_time_factor(self, dt: datetime, line_id: str, stop_id: int) -> float:
        """
        Calculate time factor based on hour, day, and conditions.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime.
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
            
        Returns:
        --------
        float
            A multiplier for passenger volume.
        """
        hour = dt.hour
        weekday = dt.weekday()
        month = dt.month
        line = self.data.lines.get(line_id)
        stop = self.data.stops.get(stop_id)
        
        if not line or not stop:
            return 1.0
        
        # Start with base factor
        factor = 1.0
        
        # Apply line popularity
        factor *= line.popularity_factor
        
        # Check day type
        is_weekend = weekday >= 5
        is_holiday = f"{dt.month:02d}-{dt.day:02d}" in self.data.holidays
        
        # Check rush hours - EXTENDED RUSH HOUR PERIODS FOR MORE STRESS
        is_morning_rush = any(start <= hour < end for start, end in self.network.RUSH_HOURS_MORNING)
        is_evening_rush = any(start <= hour < end for start, end in self.network.RUSH_HOURS_EVENING)
        
        # Apply time of day factors - INCREASED FOR OVERCROWDING
        if is_morning_rush and not (is_weekend or is_holiday):
            # Morning rush hour - MAJOR INCREASE for overcrowding simulation
            if line.is_crowded:
                factor *= 4.5 if stop.is_special_location else 3.5  # INCREASED from 2.5/2.0
            else:
                factor *= 3.2 if stop.is_special_location else 2.8  # INCREASED from 2.0/1.6
        elif is_evening_rush and not (is_weekend or is_holiday):
            # Evening rush hour - MAJOR INCREASE for overcrowding simulation
            if line.is_crowded:
                factor *= 4.0 if stop.is_special_location else 3.2  # INCREASED from 2.2/1.8
            else:
                factor *= 3.0 if stop.is_special_location else 2.5  # INCREASED from 1.8/1.5
        elif hour < 6:  # Very early morning (before 6 AM)
            factor *= 0.05  # Extremely low demand - mostly emergency/night workers
        elif hour >= 23:  # Late night (after 11 PM)
            factor *= 0.15  # Low demand - late workers, entertainment
        elif 6 <= hour < 7:  # Early morning service ramp-up
            factor *= 0.3  # Gradual increase as service starts
        elif 9 <= hour < 12 or 13 <= hour < 16:
            factor *= 0.8  # Reduced from 1.2 - more realistic regular hours
        elif 12 <= hour < 13:
            factor *= 1.3  # Reduced from 2.0 - more realistic lunch time
        elif 19 <= hour < 21:
            factor *= 1.2  # Reduced from 1.8 - more realistic evening activities
        
        # Weekend and holiday factors
        if is_weekend:
            if 10 <= hour < 20:  # Peak hours shift later on weekends
                factor *= 0.9  # Restored to more realistic weekend levels
            else:
                factor *= 0.5  # Restored to quieter off-peak weekends
                
        if is_holiday:
            factor *= 0.6  # Restored to more realistic holiday levels
            
        # Apply seasonal factors
        for season, (months, season_factor) in self.SEASONS.items():
            if month in months:
                factor *= season_factor
                break
        
        # Add some randomness (±20% instead of ±15% for more variability)
        factor *= random.uniform(0.8, 1.2)
        
        return factor
    
    def generate_passengers(self, dt: datetime, line_id: str, stop_id: int, 
                      bus_id: str, time_since_last_bus: float, direction: str = "outbound") -> Tuple[int, int]:
        """
        Generate boarding and alighting passengers for a specific stop.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime.
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
        bus_id : str
            ID of the bus.
        time_since_last_bus : float
            Time in minutes since the last bus served this stop.
        direction : str
            Direction of travel ("outbound" or "return").
            
        Returns:
        --------
        Tuple[int, int]
            Number of boarding and alighting passengers.
        """
        # Initialize route position variables at the very beginning
        route_progress = 0.5
        is_first_half = True
        is_second_half = False
        
        stop = self.data.stops.get(stop_id)
        bus = self.data.buses.get(bus_id)
        line = self.data.lines.get(line_id)
        
        if not stop or not bus or not line:
            return 0, 0
        
        # Get a base passenger arrival rate per minute based on stop characteristics
        if stop.is_special_location:
            # Major hubs, terminals, universities, shopping centers - SIGNIFICANTLY INCREASED
            base_minute_rate = random.uniform(1.5, 4.0)  # INCREASED from 0.8-2.5 passengers per minute
        else:
            # Regular neighborhood stops - SIGNIFICANTLY INCREASED
            base_minute_rate = random.uniform(0.8, 2.0)  # INCREASED from 0.3-1.2 passengers per minute
        
        # Apply time-based factor
        time_factor = self.get_time_factor(dt, line_id, stop_id)
        
        # Account for accumulated passengers since last bus
        # Cap time_since_last_bus to a reasonable upper limit to prevent excessive accumulation
        # No matter how long since the last bus, there's a natural ceiling to how many
        # passengers will wait at a stop
        capped_time = min(time_since_last_bus, 30.0)  # Cap at 30 minutes
        
        # Apply diminishing returns for longer waiting times
        # People are less likely to wait the longer the interval
        if capped_time > 15.0:
            effective_time = 15.0 + (capped_time - 15.0) * 0.5  # 50% effectiveness after 15 minutes
        else:
            effective_time = capped_time
            
        accumulated_rate = base_minute_rate * effective_time * time_factor
        
        # Get flow patterns based on time of day
        hour = dt.hour
        patterns = self.passenger_flow_patterns.get(stop_id, {})
        
        # Determine boarding and alighting multipliers based on time of day
        boarding_multiplier = 1.0
        alighting_multiplier = 1.0
        
        if 7 <= hour < 10:  # Morning
            boarding_multiplier = patterns.get("morning_boarding", 1.0)
            alighting_multiplier = patterns.get("morning_alighting", 1.0)
        elif 10 <= hour < 16:  # Midday
            boarding_multiplier = patterns.get("midday_boarding", 1.0)
            alighting_multiplier = patterns.get("midday_alighting", 1.0)
        elif 16 <= hour < 20:  # Evening
            boarding_multiplier = patterns.get("evening_boarding", 1.0)
            alighting_multiplier = patterns.get("evening_alighting", 1.0)
        elif 20 <= hour < 24:  # Night
            boarding_multiplier = patterns.get("night_boarding", 1.0)
            alighting_multiplier = patterns.get("night_alighting", 1.0)
        
        # Add day-of-week effects on passenger generation
        day_of_week = dt.weekday()
        day_factors = [1.0, 1.05, 1.1, 1.05, 1.15, 0.7, 0.6]  # Mon-Sun
        time_factor *= day_factors[day_of_week]
        
        # Reverse boarding/alighting multipliers for return direction
        if direction == "return":
            boarding_multiplier, alighting_multiplier = alighting_multiplier, boarding_multiplier
        
        # Calculate final boarding rate with direction and time factors
        adjusted_boarding_rate = accumulated_rate * boarding_multiplier
        
        # Apply rush hour and crowded line effects - FIXED TO REALISTIC VALUES
        weekday = dt.weekday()
        is_rush_hour = ((7 <= hour < 9) or (17 <= hour < 19)) and weekday < 5
        
        # REALISTIC variations in passenger demand - removed extreme multipliers
        if is_rush_hour and line_id in self.network.CROWDED_LINES:
            # MAJOR INCREASE during rush hour for crowded lines - OVERCROWDING INTENDED
            adjusted_boarding_rate *= random.uniform(3.5, 5.0)  # INCREASED from 1.8-2.2 for overcrowding
            
            # Extra spike on specific days for some lines - INCREASED
            if line_id in ['101', '105-2'] and weekday in [1, 3]:  # Tuesday, Thursday
                adjusted_boarding_rate *= random.uniform(1.8, 2.5)  # INCREASED from 1.2-1.4
                
        elif is_rush_hour and line_id in self.network.UNSTABLE_LINES:
            # Unstable lines have MAJOR unpredictable demand spikes during rush hour
            volatility = random.random()
            if volatility > 0.5:  # Increased chance from 40% to 50%
                adjusted_boarding_rate *= random.uniform(3.0, 4.5)  # INCREASED from 1.6-2.0
            else:  # Otherwise major increase
                adjusted_boarding_rate *= random.uniform(2.2, 3.0)  # INCREASED from 1.3-1.6
                
        elif is_rush_hour:
            # Regular lines - MAJOR increase during rush hour for overcrowding
            adjusted_boarding_rate *= random.uniform(2.8, 3.5)  # INCREASED from 1.5-1.8
            
        elif line_id in self.network.CROWDED_LINES:
            # Crowded lines have higher demand even outside rush hour
            adjusted_boarding_rate *= random.uniform(1.8, 2.2)  # INCREASED from 1.2-1.4
            
        elif line_id in self.network.UNDERUTILIZED_LINES:
            # Underutilized lines have reduced demand
            adjusted_boarding_rate *= random.uniform(0.6, 0.9)  # REDUCED from 0.8-1.2
            
            # Occasional unexpected bursts of passengers - much rarer
            if random.random() < 0.03:  # REDUCED from 10% to 3% chance
                adjusted_boarding_rate *= random.uniform(1.5, 2.0)  # REDUCED from 4.0-7.0
        
        # CRITICAL FIX: Early morning service start logic
        if hour < 6:  # Very early morning (before 6 AM)
            # Minimal passenger activity - mostly night shift workers, early commuters
            adjusted_boarding_rate *= 0.1  # Only 10% of normal demand
            
            # Cap time accumulation for first buses of the day
            if time_since_last_bus > 60:  # More than 1 hour (overnight)
                # Don't accumulate overnight - people don't wait 8+ hours for buses
                effective_time = min(effective_time, 15.0)  # Max 15 minutes worth of passengers
                adjusted_boarding_rate = base_minute_rate * effective_time * 0.1  # Recalculate with early morning factor
                
        elif 6 <= hour < 7:  # Early morning service ramp-up
            # Gradual increase as service starts
            adjusted_boarding_rate *= 0.3  # 30% of normal demand
            
            # Limit accumulation for early service
            if time_since_last_bus > 45:  # More than 45 minutes
                effective_time = min(effective_time, 20.0)  # Max 20 minutes worth
                adjusted_boarding_rate = base_minute_rate * effective_time * 0.3
        
        # Calculate potential boarding based on accumulated rate
        # Use a more conservative Poisson distribution parameter
        potential_boarding = int(np.random.poisson(adjusted_boarding_rate))
        
        # Add waiting passengers specific to this line at this stop, if available
        line_waiting_key = (stop_id, line_id)
        if hasattr(self.data, 'line_waiting_passengers') and line_waiting_key in self.data.line_waiting_passengers:
            # Use line-specific waiting passengers if available
            potential_boarding += self.data.line_waiting_passengers.get(line_waiting_key, 0)
            # Reset line-specific waiting count
            self.data.line_waiting_passengers[line_waiting_key] = 0
        else:
            # Add general waiting passengers at this stop
            potential_boarding += self.data.waiting_passengers.get(stop_id, 0)
            # Reset waiting count for this stop
            self.data.waiting_passengers[stop_id] = 0
        
        # Apply realistic cap based on stop type - INCREASED for heavy optimization testing
        # Allow more passengers to accumulate for genetic algorithm stress testing
        if stop.is_special_location:
            potential_boarding = min(potential_boarding, 500)  # INCREASED from 300 for special locations
        else:
            potential_boarding = min(potential_boarding, 350)  # INCREASED from 180 for regular stops
        
        # Calculate available capacity on the bus - REDUCED tolerance for faster overcrowding
        # Each bus has a different tolerance for overcrowding
        
        # Get or create a bus-specific overcrowding tolerance
        if not hasattr(bus, 'overcrowding_tolerance'):
            # Some buses are operated by strict drivers who don't allow as much overcrowding
            # Others are more lenient and allow more passengers to squeeze in
            strict_driver = random.random() < 0.3  # INCREASED back to 30% for more realistic mix
            if strict_driver:
                tolerance = random.uniform(1.3, 1.5)  # REDUCED to realistic levels (130-150%)
            else:
                tolerance = random.uniform(1.6, 1.9)  # REDUCED to realistic levels (160-190%)
                
            # Store the tolerance value with the bus
            bus.overcrowding_tolerance = tolerance
            
        # Calculate max capacity based on the bus's tolerance
        max_capacity = int(bus.capacity * bus.overcrowding_tolerance)
        current_load = bus.current_load
        
        # If bus is near capacity, limit new boardings 
        if current_load >= bus.capacity:
            # Bus is already at normal capacity
            # Allow limited overcrowding with decreasing probability as load increases
            # More realistic - as buses get more crowded, fewer additional passengers can board
            overcrowding_ratio = (current_load - bus.capacity) / (max(1, max_capacity - bus.capacity))
            overcrowding_factor = max(0, 1 - overcrowding_ratio * 1.2)  # Sharper dropoff in boarding
            
            # Apply stricter limits for extreme overcrowding
            if current_load >= max_capacity * 0.95:  # Near absolute maximum (95%+ of max tolerance)
                # Driver strongly resists more passengers - bus is dangerously overcrowded
                driver_frustration = random.random()
                if driver_frustration > 0.2:  # 80% chance driver refuses more boardings
                    available_capacity = 0
                else:
                    # Maybe 1 person can squeeze in at most
                    available_capacity = int(random.randint(0, 1) * overcrowding_factor)
            elif current_load >= max_capacity * 0.85:  # Very high occupancy (85%+ of max tolerance)
                # Driver gets frustrated, very few additional passengers allowed
                available_capacity = int(random.randint(0, 2) * overcrowding_factor)
            elif current_load >= max_capacity * 0.7:  # High overcrowding (70%+ of max tolerance)
                # Driver starts limiting passengers more strictly
                available_capacity = int(random.randint(1, 4) * overcrowding_factor)
            else:  # Moderate overcrowding (below 70% of max tolerance)
                # Some passengers can still board, but with resistance
                available_capacity = int(random.randint(3, 8) * overcrowding_factor)
        else:
            # Bus still has normal capacity
            available_capacity = max_capacity - current_load
            
            # As we approach capacity, fewer people board than theoretically could
            # This happens because the bus starts looking full visually before it's actually full
            if current_load >= bus.capacity * 0.8:
                # Apply a reducing factor as we get closer to capacity
                # People are less likely to board a bus that looks full
                capacity_factor = 1 - (current_load / bus.capacity) * 0.7  # More aggressive reduction
                available_capacity = int(available_capacity * capacity_factor)
        
        # APPLY FIRST-HALF vs SECOND-HALF BOARDING PATTERNS
        if is_first_half:
            # FIRST HALF: Higher boarding, encourage more passengers to get on
            boarding_boost = random.uniform(1.3, 1.8)  # 30-80% boost in first half
            potential_boarding = int(potential_boarding * boarding_boost)
        elif is_second_half:
            # SECOND HALF: Lower boarding, discourage boarding as route progresses
            boarding_reduction = random.uniform(0.4, 0.7)  # 30-60% reduction in second half
            potential_boarding = int(potential_boarding * boarding_reduction)
        
        # Final boarding count limited by available capacity
        boarding = min(potential_boarding, available_capacity)
        
        # ABSOLUTE HARD CAP: Never exceed max_capacity under any circumstances
        if current_load + boarding > max_capacity:
            excess_over_absolute_limit = (current_load + boarding) - max_capacity
            boarding = max(0, boarding - excess_over_absolute_limit)
        
        # Those who couldn't board remain waiting at the stop
        remaining_waiting = potential_boarding - boarding
        
        # Get current position in the route - ENHANCED FOR STRONG FIRST/SECOND HALF PATTERNS
        try:
            stops = line.stops
            total_stops = len(stops)
            current_stop_index = stops.index(stop_id)
            route_progress = current_stop_index / total_stops
            
            # Determine if we're in first half or second half for stronger patterns
            is_first_half = route_progress < 0.5
            is_second_half = route_progress >= 0.5
            
        except (ValueError, ZeroDivisionError):
            route_progress = 0.5  # Default to middle of route
            is_first_half = True
            is_second_half = False
        
        # Calculate alighting based on current bus load and route progress
        
        # Check if this is the final stop
        if (direction == "outbound" and stop_id == line.stops[-1]) or \
        (direction == "return" and stop_id == line.stops[0]):
            # Everyone gets off at the terminus
            alighting = bus.current_load
            
            # Process transfers at terminal stations
            if stop.is_special_location and stop.transfer_likelihood > 0 and alighting > 0:
                self._process_transfers(dt, stop_id, alighting, stop.transfer_likelihood)
                
            return boarding, alighting  # No more processing needed for terminus
        
        # For intermediate stops, use a mix of:
        # 1. Route progress (more people get off as we progress)
        # 2. Current bus load (percentage of current passengers)
        # 3. Stop importance (more activity at important stops)
        # 4. ENHANCED: Strong first-half vs second-half patterns
        
        # Base alighting percentage varies more for different line types
        if line_id in self.network.CROWDED_LINES:
            # Crowded lines have more turnover - people ride shorter distances
            base_alighting_percentage = random.uniform(0.2, 0.35)
        elif line_id in self.network.UNDERUTILIZED_LINES:
            # Underutilized lines often have people riding longer distances
            base_alighting_percentage = random.uniform(0.1, 0.2)
        elif line_id in self.network.UNSTABLE_LINES:
            # Unstable lines have unpredictable alighting patterns
            if random.random() < 0.3:  # Sometimes large groups exit
                base_alighting_percentage = random.uniform(0.3, 0.5)
            else:
                base_alighting_percentage = random.uniform(0.1, 0.25)
        else:
            # Regular lines have moderate turnover
            base_alighting_percentage = random.uniform(0.15, 0.3)
            
        # APPLY FIRST-HALF vs SECOND-HALF ALIGHTING PATTERNS
        if is_first_half:
            # FIRST HALF: Lower alighting, keep passengers on bus longer
            alighting_reduction = random.uniform(0.3, 0.6)  # 40-70% reduction in first half
            base_alighting_percentage *= alighting_reduction
        elif is_second_half:
            # SECOND HALF: Higher alighting, encourage passengers to get off
            alighting_boost = random.uniform(1.4, 2.2)  # 40-120% increase in second half
            base_alighting_percentage *= alighting_boost
        
        # Adjust based on route progress (more people get off near the end)
        if direction == "outbound":
            # More alighting as we progress toward the end
            progress_factor = route_progress
        else:
            # More alighting as we approach the beginning (for return journey)
            progress_factor = 1.0 - route_progress
        
        # Special adjustments for particular times and stops
        if stop.is_special_location:
            if any(loc in stop.stop_name for loc in ["Ankara Üniversitesi", "Atılım Üniversitesi", "Ufuk Üniversitesi", "Teknokent"]) and 7 <= hour < 10:
                # Universities get more alighting in morning rush hour
                alighting_multiplier *= random.uniform(1.5, 2.0)
            elif any(loc in stop.stop_name for loc in ["Avm", "Taurus", "Samanyolu"]) and 17 <= hour < 20:
                # Shopping centers get more alighting in evening
                alighting_multiplier *= random.uniform(1.8, 2.2)
            elif any(loc in stop.stop_name for loc in ["Kızılay", "Ulus", "Opera", "Sıhhiye"]):
                # Always high activity at central hubs
                alighting_multiplier *= random.uniform(1.3, 1.8)
        
        # Combine factors with stop-specific alighting multiplier
        alighting_percentage = base_alighting_percentage * (1.0 + progress_factor) * alighting_multiplier
        
        # For special stops, increase alighting further
        if stop.is_special_location:
            alighting_percentage *= random.uniform(1.3, 1.7)  # More variability
        
        # Add day-specific patterns
        if weekday == 4 and hour >= 16:  # Friday evening
            alighting_percentage *= random.uniform(1.2, 1.5)  # More people going out
        elif weekday == 6 and 10 <= hour < 14:  # Sunday morning/afternoon
            alighting_percentage *= random.uniform(0.6, 0.8)  # Fewer people getting off
        
        # Calculate actual alighting count from bus load with more randomness
        alighting = int(bus.current_load * alighting_percentage)
        
        # Add more variability - sometimes groups of people get off together
        if random.random() < 0.15:  # 15% chance of group alighting
            group_size = random.randint(3, 8)
            alighting += group_size
            
        # Ensure we don't exceed current load
        alighting = min(alighting, bus.current_load)
        
        # Process transfers at interchange points
        if stop.is_special_location and stop.transfer_likelihood > 0 and alighting > 0:
            self._process_transfers(dt, stop_id, alighting, stop.transfer_likelihood)
        
        # Ensure the new load never exceeds max_capacity
        new_load = current_load + boarding - alighting
        if new_load > max_capacity:
            # Adjust boarding downward to respect maximum capacity
            excess = new_load - max_capacity
            boarding = max(0, boarding - excess)
            
            # Update remaining waiting passengers
            remaining_waiting += excess
        
        # Save the remaining waiting passengers at the stop
        if hasattr(self.data, 'line_waiting_passengers'):
            # Distribute remaining passengers to different lines at this stop
            self._distribute_waiting_passengers(stop_id, remaining_waiting)
        else:
            # Use the old approach: assign all waiting to this stop
            self.data.waiting_passengers[stop_id] = remaining_waiting
            
        # Track wait time statistics for service quality metrics
        if time_since_last_bus > 15 and boarding > 0:  # Long wait threshold (15 min)
            if not hasattr(self.data, 'long_wait_incidents'):
                self.data.long_wait_incidents = []
                
            self.data.long_wait_incidents.append({
                'datetime': dt,
                'line_id': line_id,
                'stop_id': stop_id,
                'wait_time': time_since_last_bus,
                'boarding': boarding
            })
        
        # Ensure non-negative results
        return max(0, boarding), max(0, alighting)
    
    def _process_transfers(self, dt: datetime, stop_id: int, alighting: int, transfer_likelihood: float) -> None:
        """
        Process passengers transferring between lines at a stop.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime.
        stop_id : int
            ID of the stop.
        alighting : int
            Number of alighting passengers that could potentially transfer.
        transfer_likelihood : float
            Probability of a passenger transferring to another line.
        """
        # Initialize line_waiting_passengers dict if it doesn't exist
        if not hasattr(self.data, 'line_waiting_passengers'):
            self.data.line_waiting_passengers = {}
            
        # Find all lines serving this stop
        serving_lines = []
        for line_id, line in self.data.lines.items():
            if stop_id in line.stops:
                serving_lines.append(line_id)
                
        if len(serving_lines) <= 1:
            return  # No other lines to transfer to
            
        # Calculate total number of transferring passengers
        transfer_count = int(alighting * transfer_likelihood)
        
        if transfer_count <= 0:
            return
            
        # Some passengers need to wait for their next bus
        # Add them to the line-specific waiting counts
        
        # Calculate line popularity values for weighted distribution
        line_popularities = {}
        total_popularity = 0
        
        for line_id in serving_lines:
            line = self.data.lines.get(line_id)
            if line:
                popularity = line.popularity_factor
                line_popularities[line_id] = popularity
                total_popularity += popularity
                
        # Distribute transferring passengers based on line popularity
        remaining_transfers = transfer_count
        
        for line_id, popularity in line_popularities.items():
            # Calculate passengers transferring to this line
            # Use weighted distribution based on line popularity
            if total_popularity > 0:
                line_transfers = int(transfer_count * (popularity / total_popularity))
            else:
                line_transfers = int(transfer_count / len(serving_lines))
                
            # Ensure we don't assign more transfers than we have
            line_transfers = min(line_transfers, remaining_transfers)
            remaining_transfers -= line_transfers
            
            # Add to line-specific waiting passengers
            key = (stop_id, line_id)
            self.data.line_waiting_passengers[key] = self.data.line_waiting_passengers.get(key, 0) + line_transfers
            
        # If any transfers remain due to rounding, assign them to the most popular line
        if remaining_transfers > 0 and line_popularities:
            most_popular_line = max(line_popularities.items(), key=lambda x: x[1])[0]
            key = (stop_id, most_popular_line)
            self.data.line_waiting_passengers[key] = self.data.line_waiting_passengers.get(key, 0) + remaining_transfers
            
        # Track transfer statistics
        if not hasattr(self.data, 'transfer_statistics'):
            self.data.transfer_statistics = []
            
        self.data.transfer_statistics.append({
            'datetime': dt,
            'stop_id': stop_id,
            'total_transfers': transfer_count,
            'lines': [l for l in serving_lines]
        })
            
    def _distribute_waiting_passengers(self, stop_id: int, waiting_count: int) -> None:
        """
        Distribute waiting passengers among different lines at a stop.
        
        Parameters:
        -----------
        stop_id : int
            ID of the stop.
        waiting_count : int
            Number of waiting passengers to distribute.
        """
        if waiting_count <= 0:
            return
            
        # Find all lines serving this stop
        serving_lines = []
        for line_id, line in self.data.lines.items():
            if stop_id in line.stops:
                serving_lines.append(line_id)
                
        if not serving_lines:
            return
            
        # Calculate line popularity values for weighted distribution
        line_popularities = {}
        total_popularity = 0
        
        for line_id in serving_lines:
            line = self.data.lines.get(line_id)
            if line:
                popularity = line.popularity_factor
                line_popularities[line_id] = popularity
                total_popularity += popularity
                
        # Distribute waiting passengers based on line popularity
        remaining_passengers = waiting_count
        
        for line_id, popularity in line_popularities.items():
            # Calculate passengers waiting for this line
            # Use weighted distribution based on line popularity
            if total_popularity > 0:
                line_waiting = int(waiting_count * (popularity / total_popularity))
            else:
                line_waiting = int(waiting_count / len(serving_lines))
                
            # Ensure we don't assign more passengers than we have
            line_waiting = min(line_waiting, remaining_passengers)
            remaining_passengers -= line_waiting
            
            # Add to line-specific waiting passengers
            key = (stop_id, line_id)
            self.data.line_waiting_passengers[key] = self.data.line_waiting_passengers.get(key, 0) + line_waiting
            
        # If any passengers remain due to rounding, assign them to the most popular line
        if remaining_passengers > 0 and line_popularities:
            most_popular_line = max(line_popularities.items(), key=lambda x: x[1])[0]
            key = (stop_id, most_popular_line)
            self.data.line_waiting_passengers[key] = self.data.line_waiting_passengers.get(key, 0) + remaining_passengers

    def accumulate_waiting_passengers(self, dt: datetime, time_step: int) -> None:
        """
        Accumulate waiting passengers at all stops based on time step.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime.
        time_step : int
            Simulation time step in minutes.
        """
        # Apply decay for passengers who have been waiting too long (patience factor)
        # People tend to give up and find alternative transport if buses don't come
        for stop_id in list(self.data.waiting_passengers.keys()):
            current_waiting = self.data.waiting_passengers.get(stop_id, 0)
            
            # Apply a decaying factor based on waiting time
            # Longer time steps should have higher decay as people have more time to give up
            patience_decay_factor = 0.98 - (time_step * 0.002)  # More decay with longer time steps
            
            # Apply higher decay during non-peak hours when alternatives might be more attractive
            hour = dt.hour
            is_peak_hour = (7 <= hour < 10) or (16 <= hour < 19)
            
            if not is_peak_hour:
                patience_decay_factor -= 0.02  # Higher abandonment rate during off-peak
                
            # Apply decay
            decayed_passengers = int(current_waiting * patience_decay_factor)
            passengers_leaving = current_waiting - decayed_passengers
            
            # Update waiting passengers
            self.data.waiting_passengers[stop_id] = decayed_passengers
            
            # Track abandoned journeys if needed
            if hasattr(self.data, 'abandoned_passengers'):
                self.data.abandoned_passengers = self.data.abandoned_passengers + passengers_leaving
        
        # For each stop
        for stop_id, stop in self.data.stops.items():
            # Get a basic arrival rate based on stop characteristics
            if stop.is_special_location:
                base_rate = random.uniform(1.0, 2.5)  # INCREASED from 0.4-1.8 for higher passenger volumes
            else:
                base_rate = random.uniform(0.3, 1.0)  # INCREASED from 0.1-0.6 for higher passenger volumes
            
            # Apply time factors
            hour = dt.hour
            weekday = dt.weekday()
            is_weekend = weekday >= 5
            is_holiday = f"{dt.month:02d}-{dt.day:02d}" in self.data.holidays
            is_rush_hour = ((7 <= hour < 10) or (16 <= hour < 20)) and weekday < 5
            
            if is_rush_hour and not (is_weekend or is_holiday):
                base_rate *= random.uniform(2.5, 3.5)  # INCREASED from 1.8-2.2 for optimization testing
            elif is_weekend:
                base_rate *= random.uniform(0.4, 0.8)  # Lower on weekends
            elif is_holiday:
                base_rate *= random.uniform(0.3, 0.6)  # Lower on holidays
            elif hour < 6 or hour >= 22:
                base_rate *= random.uniform(0.1, 0.3)  # Much lower at night
            
            # Calculate accumulated passengers for this time step
            new_passengers = int(np.random.poisson(base_rate * time_step))
            
            # Add to waiting passengers
            self.data.waiting_passengers[stop_id] = self.data.waiting_passengers.get(stop_id, 0) + new_passengers
    
    def update_bus_load(self, bus_id: str, boarding: int, alighting: int) -> int:
        """
        Update a bus's passenger load after boarding and alighting.
        
        Parameters:
        -----------
        bus_id : str
            ID of the bus.
        boarding : int
            Number of boarding passengers.
        alighting : int
            Number of alighting passengers.
            
        Returns:
        --------
        int
            New passenger load.
        """
        bus = self.data.buses.get(bus_id)
        if not bus:
            return 0
        
        # Update load
        new_load = bus.current_load + boarding - alighting
        new_load = max(0, new_load)  # Ensure non-negative
        
        # Update bus object
        bus.current_load = new_load
        
        return new_load
    
    def record_passenger_activity(self, dt: datetime, line_id: str, stop_id: int, 
                               bus_id: str, boarding: int, alighting: int, 
                               current_load: int, new_load: int) -> PassengerGroup:
        """
        Record passenger boarding and alighting activity.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime.
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
        bus_id : str
            ID of the bus.
        boarding : int
            Number of boarding passengers.
        alighting : int
            Number of alighting passengers.
        current_load : int
            Current bus load before activity.
        new_load : int
            New bus load after activity.
            
        Returns:
        --------
        PassengerGroup
            Record of passenger activity.
        """
        # Create passenger group record
        passenger_group = PassengerGroup(
            stop_id=stop_id,
            line_id=line_id,
            bus_id=bus_id,
            timestamp=dt,
            boarding_count=boarding,
            alighting_count=alighting
        )
        
        return passenger_group