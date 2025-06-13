"""
Revised simulation engine for coordinating the transit simulation.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import json

# Handle imports with fallback for different execution contexts
try:
    from .data_models import BusTransitData, SimulationConfig, PassengerGroup
except ImportError:
    from core.data_models import BusTransitData, SimulationConfig, PassengerGroup

try:
    from ..components.transit_network import TransitNetwork
    from ..components.bus_management import BusManager
    from ..components.schedule_generator import ScheduleGenerator
    from ..components.passenger_generator import PassengerGenerator
except ImportError:
    from components.transit_network import TransitNetwork
    from components.bus_management import BusManager
    from components.schedule_generator import ScheduleGenerator
    from components.passenger_generator import PassengerGenerator


class SimulationEngine:
    """
    Main engine that coordinates the bus transit simulation.
    """
    
    def __init__(self, data: BusTransitData = None, config: SimulationConfig = None):
        """Initialize the simulation engine."""
        # Create or use provided data container
        self.data = data if data else BusTransitData()
        
        # Create or use provided configuration
        self.config = config if config else SimulationConfig(
            start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            end_date=datetime.now().replace(hour=23, minute=59, second=59, microsecond=0),
            time_step=5,
            randomize_travel_times=True,
            randomize_passenger_demand=True,
            weather_effects_probability=0.15,
            seed=42
        )
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize components
        self.network = TransitNetwork(self.data)
        self.bus_manager = BusManager(self.data)
        self.schedule_generator = ScheduleGenerator(self.data, self.network)
        self.passenger_generator = PassengerGenerator(self.data, self.network)
        
        # Store results
        self.passenger_flow_results = []
        self.bus_positions_results = []
        
        # Track the last time a bus visited each stop on each line
        self.last_bus_visit = {}  # {(line_id, stop_id): datetime}
        
        # Debug flag
        self.debug = False
    
    def load_data(self, stops_lines_file: str) -> bool:
        """
        Load necessary data files for simulation.
        
        Parameters:
        -----------
        stops_lines_file : str
            Path to CSV file with stop and line information.
            
        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        # Load transit network data
        success = self.network.load_network_data(stops_lines_file)
        if not success:
            print("Failed to load transit network data")
            return False
        
        print(f"Loaded {len(self.data.lines)} lines and {len(self.data.stops)} stops")
        
        return True
    
    def setup_simulation(self, num_buses_per_line: int = 10) -> bool:
        """
        Set up the simulation by initializing fleet and schedules.
        
        Parameters:
        -----------
        num_buses_per_line : int
            Number of buses to allocate per line.
            
        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        try:
            # Generate bus fleet
            self.bus_manager.generate_buses(num_buses_per_line)
            print(f"Generated {len(self.data.buses)} buses")
            
            # Generate line schedules and bus assignments for the simulation period
            current_date = self.config.start_date
            while current_date <= self.config.end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                print(f"Generating schedules for {date_str}...")
                schedules = self.schedule_generator.generate_line_schedules(date_str)
                
                # Count schedules per line
                schedule_counts = {}
                for line_id, entries in schedules.items():
                    schedule_counts[line_id] = len(entries)
                print(f"Generated schedules: {schedule_counts}")
                
                print(f"Assigning buses for {date_str}...")
                assignments = self.schedule_generator.assign_buses_to_schedules(date_str)
                
                # Count assignments per bus
                if self.debug:
                    assignment_counts = {}
                    for bus_id, bus_assignments in assignments.items():
                        assignment_counts[bus_id] = len(bus_assignments)
                    print(f"Bus assignments: {assignment_counts}")
                
                current_date += timedelta(days=1)
            
            # Initialize the last bus visit tracking
            self.last_bus_visit = {}
            
            return True
            
        except Exception as e:
            print(f"Error setting up simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_simulation(self) -> pd.DataFrame:
        """
        Run the complete simulation from start to end date.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with passenger flow results.
        """
        # Clear any previous results
        self.passenger_flow_results = []
        self.bus_positions_results = []
        
        try:
            # Start time tracking
            import time
            start_time = time.time()
            
            # Run simulation for the specified period
            current_time = self.config.start_date
            end_time = self.config.end_date
            step_delta = timedelta(minutes=self.config.time_step)
            
            # Create simulation progress indicator
            total_steps = int((end_time - current_time).total_seconds() / (step_delta.total_seconds()))
            step_count = 0
            last_progress = -1
            
            # Track the last operational period for graceful bus termination
            last_operational_time = None
            in_operational_hours = False
            
            # Track simulation days
            current_date = current_time.date()
            last_date = None
            
            print(f"Starting simulation from {current_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"Simulating {(end_time - current_time).days + 1} days with {self.config.time_step} minute time steps")
            
            while current_time <= end_time:
                # Check if we moved to a new day
                if current_time.date() != current_date:
                    # Save the last date
                    last_date = current_date
                    current_date = current_time.date()
                    
                    # Print day transition info
                    print(f"Simulating day: {current_date.strftime('%Y-%m-%d')}")
                    
                    # Handle the end of the previous day correctly
                    if last_date:
                        end_of_day = datetime.combine(last_date, datetime.max.time()).replace(
                            hour=23, minute=59, second=59
                        )
                        self._handle_end_of_operations(end_of_day)
                        
                        # Set up new day's schedules
                        date_str = current_date.strftime('%Y-%m-%d')
                        # Make sure this day's assignments are loaded
                        if hasattr(self.data, 'all_bus_assignments') and date_str in self.data.all_bus_assignments:
                            self.data.bus_assignments = self.data.all_bus_assignments[date_str]
                        
                        # Reset operational hours flags
                        in_operational_hours = False
                
                # Check if within operational hours
                hour = current_time.hour
                is_operational = self.network.OPERATIONAL_HOURS[0] <= hour < self.network.OPERATIONAL_HOURS[1]
                
                # Track transitions in and out of operational hours
                if is_operational and not in_operational_hours:
                    in_operational_hours = True
                elif not is_operational and in_operational_hours:
                    in_operational_hours = False
                    last_operational_time = current_time - step_delta
                    
                    # Gracefully terminate buses at end of operational hours
                    self._handle_end_of_operations(last_operational_time)
                
                if is_operational:
                    # Run a single time step during operational hours
                    self._simulate_time_step(current_time)
                    
                    # Progress indicator
                    step_count += 1
                    progress = (step_count * 100) // total_steps
                    if progress > last_progress and progress % 5 == 0:
                        print(f"Simulation {progress}% complete - " + 
                              f"Current time: {current_time.strftime('%Y-%m-%d %H:%M')}")
                        last_progress = progress
                
                # Move to next time step
                current_time += step_delta
            
            # Handle any buses still in service at end of simulation
            self._handle_end_of_operations(end_time, is_end_of_simulation=True)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.passenger_flow_results)
            positions_df = pd.DataFrame(self.bus_positions_results)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"Simulation completed in {elapsed_time:.2f} seconds")
            
            # Report statistics
            if len(results_df) > 0:
                print(f"Generated {len(results_df)} passenger flow records")
                
                # Count unique dates
                if 'datetime' in results_df.columns:
                    dates = results_df['datetime'].dt.date.unique()
                    print(f"Data covers {len(dates)} unique dates from {min(dates)} to {max(dates)}")
                    
                    # Print records per date
                    records_by_date = results_df.groupby(results_df['datetime'].dt.date).size()
                    print("Records by date:")
                    for date, count in records_by_date.items():
                        print(f"  {date}: {count} records")
                
                print(f"Generated {len(positions_df)} bus position records")
                
                # Analyze passenger flow
                total_boarding = results_df['boarding'].sum()
                total_alighting = results_df['alighting'].sum()
                print(f"Total boarding: {total_boarding}, Total alighting: {total_alighting}")
                
                # Count records by line
                line_counts = results_df.groupby('line_id').size()
                print("Records by line:")
                for line_id, count in line_counts.items():
                    print(f"  Line {line_id}: {count} records")
                
                # Check for overcrowding
                overcrowded = results_df[results_df['occupancy_rate'] > 1.0]
                print(f"Overcrowded instances: {len(overcrowded)} ({len(overcrowded)/len(results_df)*100:.1f}%)")
            else:
                print("Warning: No passenger flow records generated!")
            
            return results_df
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_bus_at_position(self, line_id: str, stop_index: int, current_time: datetime) -> Optional[str]:
        """
        Find a bus that is at a specific position in the route at the given time.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        stop_index : int
            Index of the stop in the route.
        current_time : datetime
            Current simulation time.
            
        Returns:
        --------
        Optional[str]
            Bus ID if found, None otherwise.
        """
        # Get active buses on this line
        active_buses = []
        for bus_id, assignments in self.data.bus_assignments.items():
            for assignment in assignments:
                if assignment.line_id == line_id and assignment.start_time <= current_time < assignment.end_time:
                    active_buses.append(bus_id)
                    break
        
        # Check if any bus is at or near this stop
        for bus_id in active_buses:
            if bus_id in self.data.bus_positions:
                bus_line_id, bus_stop_index, progress = self.data.bus_positions[bus_id]
                
                if bus_line_id == line_id and bus_stop_index == stop_index and 0.0 <= progress <= 0.1:
                    return bus_id
        
        return None
    
    def _get_time_since_last_bus(self, line_id: str, stop_id: int, current_time: datetime) -> float:
        """
        Calculate time since the last bus served this stop.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
        current_time : datetime
            Current simulation time.
            
        Returns:
        --------
        float
            Time in minutes since last bus visit.
        """
        key = (line_id, stop_id)
        
        # If we've never seen a bus at this stop, return a default value
        if key not in self.last_bus_visit:
            # Default to a reasonable time (between 15-30 minutes)
            return random.uniform(15, 30)
        
        # Calculate time difference
        last_visit = self.last_bus_visit[key]
        time_diff = (current_time - last_visit).total_seconds() / 60.0
        
        # Ensure reasonable bounds
        return max(5.0, min(120.0, time_diff))
    
    def _update_last_bus_visit(self, line_id: str, stop_id: int, current_time: datetime) -> None:
        """
        Update the record of the last time a bus visited this stop.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
        current_time : datetime
            Current simulation time.
        """
        self.last_bus_visit[(line_id, stop_id)] = current_time
    
    def _simulate_time_step(self, current_time: datetime) -> None:
        """
        Simulate a single time step of the transit system.
        
        Parameters:
        -----------
        current_time : datetime
            Current simulation time.
        """
        # Ensure current_time includes date information for multi-day simulations
        if current_time.date() < self.config.start_date.date() or current_time.date() > self.config.end_date.date():
            if self.debug:
                print(f"Warning: Time step {current_time} is outside simulation period.")
            return

        # Print debug info occasionally to track the day being simulated
   #     if current_time.hour == 12 and current_time.minute == 0:
    #        print(f"DEBUG: Simulating midday for {current_time.strftime('%Y-%m-%d')}")

        # Accumulate waiting passengers at all stops
        self.passenger_generator.accumulate_waiting_passengers(current_time, self.config.time_step)
        
        # Get active buses for this time step
        active_buses = self.bus_manager.get_active_buses(current_time)
        
        if self.debug and len(active_buses) == 0:
            # This shouldn't happen during operational hours
            print(f"Warning: No active buses at {current_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Track station stops for this time step
        # This will be used to track stops the buses visited in the previous time step
        # but will visit stops before the current time step
        station_visits = {}  # (bus_id, stop_id) -> details
        
        # Process each active bus
        for bus_id in active_buses:
            # Skip processing if bus_id is not a string
            if not isinstance(bus_id, str):
                bus_id = str(bus_id)
                
            # Get bus details
            bus = self.data.buses.get(bus_id)
            if not bus or bus.status == "maintenance":
                continue
            
            # Get current assignment
            current_assignment = None
            
            # Access assignments from all_bus_assignments if available
            current_date_str = current_time.strftime("%Y-%m-%d")
            if hasattr(self.data, 'all_bus_assignments') and current_date_str in self.data.all_bus_assignments:
                # Get assignments for current date
                date_assignments = self.data.all_bus_assignments[current_date_str]
                if bus_id in date_assignments:
                    for assignment in date_assignments[bus_id]:
                        if assignment.start_time <= current_time < assignment.end_time:
                            current_assignment = assignment
                            break
            
            # Fall back to general assignments if no assignment found
            if not current_assignment:
                for assignment in self.data.bus_assignments.get(bus_id, []):
                    if assignment.start_time <= current_time < assignment.end_time:
                        current_assignment = assignment
                        break
            
            if not current_assignment:
                continue
            
            line_id = current_assignment.line_id
            if line_id == "MAINTENANCE":
                continue
                
            # Get line details
            line = self.data.lines.get(line_id)
            if not line:
                continue
            
            # Get previous position and time
            prev_position = None
            if hasattr(bus, 'prev_position'):
                prev_position = bus.prev_position
                prev_time = bus.prev_time
            else:
                # Initialize tracking variables if not present
                bus.prev_position = None
                bus.prev_time = current_time - timedelta(minutes=self.config.time_step)
                prev_time = bus.prev_time
            
            # Calculate the bus's position on the route
            # based on time since departure
            time_since_departure = (current_time - current_assignment.start_time).total_seconds() / 60.0
            
            # Calculate total outbound journey time
            stops = line.stops
            total_outbound_time = 0
            
            # Track segment times for precise station timing
            segment_times = {}  # (from_stop, to_stop) -> travel_time + dwell_time
            
            # Sum travel times between consecutive stops including dynamic dwell times
            for i in range(len(stops) - 1):
                from_stop = stops[i]
                to_stop = stops[i + 1]
                
                # Get travel time for this segment
                travel_time = self.network.calculate_travel_time(
                    line_id, from_stop, to_stop, current_time, include_random=False
                )
                
                # Use a dynamic dwell time based on the stop's characteristics
                # This is a simplified model that will later be refined with actual boarding/alighting data
                stop_obj = self.data.stops.get(to_stop)
                estimated_boarding = 5
                estimated_alighting = 3
                if stop_obj and stop_obj.is_special_location:
                    estimated_boarding = 10
                    estimated_alighting = 8
                    
                dwell_time = self.network.calculate_dwell_time(
                    to_stop, current_time, estimated_boarding, estimated_alighting
                )
                
                # Store segment time for precise station timing
                segment_times[(from_stop, to_stop)] = travel_time + dwell_time
                
                total_outbound_time += travel_time + dwell_time
            
            # Total route time (outbound + return)
            total_route_time = total_outbound_time * 2
            
            # Determine if bus is on outbound or return journey
            if time_since_departure <= total_outbound_time:
                # Outbound journey
                direction = "outbound"
                elapsed_route_time = time_since_departure
            else:
                # Return journey
                direction = "return"
                elapsed_route_time = time_since_departure - total_outbound_time
            
            # Determine the current and next stop
            current_stop_index = 0
            current_stop_progress = 0.0
            current_stop_id = None
            next_stop_id = None
            
            # Total elapsed_route_time so far
            accumulated_time = 0.0
            
            if direction == "outbound":
                # Process stops in order
                for i in range(len(stops) - 1):
                    from_stop = stops[i]
                    to_stop = stops[i + 1]
                    
                    # Get segment time from our pre-calculated values
                    segment_time = segment_times.get((from_stop, to_stop), 5.0)  # Default 5 min if not found
                    
                    # Check if we're in this segment
                    if accumulated_time <= elapsed_route_time < (accumulated_time + segment_time):
                        # We're in this segment
                        current_stop_index = i
                        current_stop_id = from_stop
                        next_stop_id = to_stop
                        
                        # Calculate progress within segment
                        segment_progress = (elapsed_route_time - accumulated_time) / segment_time
                        
                        # If we're at the beginning of the segment (at the stop)
                        if segment_progress < 0.1:  # Consider first 10% of time as being at the stop
                            current_stop_progress = 0.0
                        else:
                            # We're between stops
                            current_stop_progress = min(1.0, segment_progress)
                        
                        break
                    
                    accumulated_time += segment_time
                
                # Special case: at the final stop
                if current_stop_id is None:
                    current_stop_index = len(stops) - 1
                    current_stop_id = stops[-1]
                    current_stop_progress = 0.0
            else:
                # Return journey (reverse stops)
                reversed_stops = list(reversed(stops))
                
                for i in range(len(reversed_stops) - 1):
                    from_stop = reversed_stops[i]
                    to_stop = reversed_stops[i + 1]
                    
                    # For return journey, we need to get the segment time in reverse
                    segment_time = segment_times.get((to_stop, from_stop), 5.0)  # Default 5 min if not found
                    
                    # Check if we're in this segment
                    if accumulated_time <= elapsed_route_time < (accumulated_time + segment_time):
                        # We're in this segment - convert to original stop index for consistent tracking
                        orig_index = len(stops) - 1 - i
                        current_stop_index = orig_index
                        current_stop_id = from_stop
                        next_stop_id = to_stop
                        
                        # Store index for line's stop list
                        line_stop_index = len(stops) - 1 - i
                        
                        # Calculate progress within segment
                        segment_progress = (elapsed_route_time - accumulated_time) / segment_time
                        
                        # If we're at the beginning of the segment (at the stop)
                        if segment_progress < 0.1:
                            current_stop_progress = 0.0
                        else:
                            # We're between stops
                            current_stop_progress = min(1.0, segment_progress)
                        
                        # Fix direction flag for passenger generation
                        direction = "return"
                        
                        break
                    
                    accumulated_time += segment_time
                
                # Special case: at the first stop (end of return journey)
                if current_stop_id is None:
                    current_stop_index = 0
                    current_stop_id = stops[0]
                    current_stop_progress = 0.0
            
            # Update bus position
            self.bus_manager.update_bus_position(
                bus_id, line_id, current_stop_index, current_stop_progress
            )
            self.data.bus_positions[bus_id] = (line_id, current_stop_index, current_stop_progress)
            
            # Check for stops that were visited since the previous time step
            if prev_position is not None:
                prev_stop_index, prev_progress = prev_position
                
                # Check if the bus was between stops before and is now at or past a stop
                if direction == "outbound":
                    # Forward direction - check if we passed any stops
                    for idx in range(prev_stop_index + 1, current_stop_index + 1):
                        if idx < len(stops):
                            passed_stop_id = stops[idx]
                            # Record this as a stop to process for passenger activity
                            # Calculate the estimated time this stop was visited
                            
                            # More accurate timing based on actual accumulated travel times
                            # Calculate the precise time when the bus would have reached this stop
                            stop_time = None
                            
                            if prev_stop_index != current_stop_index:
                                # Calculate accumulated time to reach the passed stop
                                acc_time = 0
                                for i in range(prev_stop_index, idx):
                                    if i < len(stops) - 1:
                                        acc_time += segment_times.get((stops[i], stops[i+1]), 5.0)
                                
                                # Linear interpolation between previous and current time
                                time_ratio = acc_time / (current_time - prev_time).total_seconds() * 60.0
                                stop_time = prev_time + timedelta(minutes=acc_time)
                            
                            if stop_time is None:
                                # Fallback to simple interpolation if accurate timing fails
                                progress_ratio = (idx - prev_stop_index) / max(1, (current_stop_index - prev_stop_index))
                                stop_time = prev_time + progress_ratio * (current_time - prev_time)
                                
                            # Store for processing
                            station_visits[(bus_id, passed_stop_id)] = {
                                'time': stop_time,
                                'line_id': line_id,
                                'direction': direction
                            }
                else:
                    # Return direction - check if we passed any stops
                    # For return, the indices are decreasing
                    for idx in range(prev_stop_index - 1, current_stop_index - 1, -1):
                        if 0 <= idx < len(stops):
                            passed_stop_id = stops[idx]
                            
                            # More accurate timing for return journey
                            stop_time = None
                            
                            if prev_stop_index != current_stop_index:
                                # Calculate accumulated time to reach the passed stop
                                acc_time = 0
                                for i in range(prev_stop_index, idx, -1):
                                    if i > 0 and i < len(stops):
                                        # For return, we use reversed segment times
                                        acc_time += segment_times.get((stops[i], stops[i-1]), 5.0)
                                
                                # Linear interpolation
                                stop_time = prev_time + timedelta(minutes=acc_time)
                            
                            if stop_time is None:
                                # Fallback
                                progress_ratio = (prev_stop_index - idx) / max(1, (prev_stop_index - current_stop_index))
                                stop_time = prev_time + progress_ratio * (current_time - prev_time)
                                
                            # Store for processing
                            station_visits[(bus_id, passed_stop_id)] = {
                                'time': stop_time,
                                'line_id': line_id,
                                'direction': direction
                            }
            
            # Store current position and time for next iteration
            bus.prev_position = (current_stop_index, current_stop_progress)
            bus.prev_time = current_time
            
            # Process passenger activity when bus is at a stop
            if current_stop_id is not None and current_stop_progress < 0.1:
                # Calculate time since last bus
                time_since_last_bus = self._get_time_since_last_bus(line_id, current_stop_id, current_time)
                
                # Update the last bus visit record
                self._update_last_bus_visit(line_id, current_stop_id, current_time)
                
                # Generate passenger activity
                boarding, alighting = self.passenger_generator.generate_passengers(
                    current_time, line_id, current_stop_id, bus_id, time_since_last_bus, direction
                )
                
                # Update bus load
                current_load = bus.current_load
                new_load = self.passenger_generator.update_bus_load(bus_id, boarding, alighting)
                
                # Always record stop visits, even if no passengers board or alight
                bus_capacity = bus.capacity
                occupancy_rate = new_load / bus_capacity if bus_capacity > 0 else 0
                
                # Store passenger flow data - log all stops
                self.passenger_flow_results.append({
                    'datetime': current_time,  # Make sure this is a proper datetime object with correct date
                    'hour': current_time.hour,
                    'minute': current_time.minute,
                    'date': current_time.date(),  # Explicitly store date for easier filtering
                    'line_id': line_id,
                    'stop_id': current_stop_id,
                    'bus_id': bus_id,
                    'boarding': boarding,
                    'alighting': alighting,
                    'current_load': current_load,
                    'new_load': new_load,
                    'capacity': bus_capacity,
                    'occupancy_rate': occupancy_rate,
                    'actual_stop': True  # Flag to indicate an actual stop event
                })
            
            # Store bus position data
            self.bus_positions_results.append({
                'datetime': current_time,
                'bus_id': bus_id,
                'line_id': line_id,
                'stop_index': current_stop_index,
                'progress': current_stop_progress,
                'current_load': bus.current_load
            })
        
        # Process any intermediate station visits
        for (bus_id, stop_id), visit_data in station_visits.items():
            visit_time = visit_data['time']
            line_id = visit_data['line_id']
            direction = visit_data['direction']
            
            # Skip if we've already processed this stop for this bus in this time step
            # (Could happen if the bus is currently at this stop)
            if self._already_processed_stop(bus_id, stop_id, visit_time):
                continue
                
            # Calculate time since last bus
            time_since_last_bus = self._get_time_since_last_bus(line_id, stop_id, visit_time)
            
            # Update the last bus visit record
            self._update_last_bus_visit(line_id, stop_id, visit_time)
            
            # Get bus details
            bus = self.data.buses.get(bus_id)
            if not bus:
                continue
                
            # Generate passenger activity
            boarding, alighting = self.passenger_generator.generate_passengers(
                visit_time, line_id, stop_id, bus_id, time_since_last_bus, direction
            )
            
            # Update bus load
            current_load = bus.current_load  # Current load at this point in time
            new_load = self.passenger_generator.update_bus_load(bus_id, boarding, alighting)
            
            # Record ALL stop visits, even with zero activity
            bus_capacity = bus.capacity
            occupancy_rate = new_load / bus_capacity if bus_capacity > 0 else 0
            
            # Store passenger flow data
            self.passenger_flow_results.append({
                'datetime': visit_time,
                'hour': visit_time.hour,
                'minute': visit_time.minute,
                'date': visit_time.date(),  # Explicitly store date for easier filtering
                'line_id': line_id,
                'stop_id': stop_id,
                'bus_id': bus_id,
                'boarding': boarding,
                'alighting': alighting,
                'current_load': current_load,
                'new_load': new_load,
                'capacity': bus_capacity,
                'occupancy_rate': occupancy_rate,
                'actual_stop': True  # Flag to indicate an actual stop event
            })
            
        # Track on-time performance using scheduled vs. actual arrivals
        if hasattr(self, 'schedule_performance') and hasattr(self.data, 'line_schedules') and len(self.data.line_schedules) > 0:
            self._update_schedule_performance(current_time)

    def _update_schedule_performance(self, current_time: datetime) -> None:
        """
        Track and update on-time performance metrics.
        
        Parameters:
        -----------
        current_time : datetime
            Current simulation time.
        """
        # Initialize performance tracking if needed
        if not hasattr(self, 'schedule_performance'):
            self.schedule_performance = []
            
        # Check recent passenger flow records (last time step)
        time_window = timedelta(minutes=self.config.time_step)
        recent_stops = []
        
        for record in self.passenger_flow_results:
            if 'actual_stop' in record and record['actual_stop'] and abs(record['datetime'] - current_time) < time_window:
                recent_stops.append((record['bus_id'], record['line_id'], record['stop_id'], record['datetime']))
        
        # Compare to scheduled stops
        for bus_id, line_id, stop_id, actual_time in recent_stops:
            # Find scheduled time for this bus, line, and stop
            scheduled_time = None
            
            for schedule in self.data.line_schedules:
                if (schedule['line_id'] == line_id and 
                    schedule['assigned_bus_id'] == bus_id and 
                    'scheduled_stops' in schedule and 
                    stop_id in schedule['scheduled_stops']):
                    scheduled_time = schedule['scheduled_stops'][stop_id]
                    break
                    
            if scheduled_time:
                # Calculate delay in minutes
                delay = (actual_time - scheduled_time).total_seconds() / 60.0
                
                # Determine status
                if abs(delay) <= 1.0:
                    status = "on_time"
                elif delay > 1.0:
                    status = "late"
                else:
                    status = "early"
                    
                # Record performance
                self.schedule_performance.append({
                    'datetime': actual_time,
                    'bus_id': bus_id,
                    'line_id': line_id,
                    'stop_id': stop_id,
                    'scheduled_time': scheduled_time,
                    'actual_time': actual_time,
                    'delay_minutes': delay,
                    'status': status
                })

    def _already_processed_stop(self, bus_id: str, stop_id: int, visit_time: datetime) -> bool:
        """
        Check if we've already processed a stop visit in this time window.
        
        Parameters:
        -----------
        bus_id : str
            ID of the bus.
        stop_id : int
            ID of the stop.
        visit_time : datetime
            Time of the visit.
            
        Returns:
        --------
        bool
            True if this stop has already been processed, False otherwise.
        """
        # Check existing passenger flow records for this bus and stop in a narrow time window
        time_tolerance = timedelta(minutes=1)  # 1-minute window
        
        for record in self.passenger_flow_results:
            if (record['bus_id'] == bus_id and 
                record['stop_id'] == stop_id and 
                abs(record['datetime'] - visit_time) < time_tolerance):
                return True
        
        return False

    def export_results(self, output_dir: str = '.') -> None:
        """
        Export simulation results to CSV files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export passenger flow results
        if self.passenger_flow_results:
            # Make sure we're not filtering by date
            df = pd.DataFrame(self.passenger_flow_results)
            
            # Print a sample of the data for verification
            if not df.empty and len(df) > 5:
                print("\nSample of passenger flow data (first 5 rows):")
                sample_df = df.head(5).copy()
                if 'datetime' in sample_df.columns:
                    sample_df['datetime'] = sample_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                print(sample_df)
                
                print("\nSample of passenger flow data (last 5 rows):")
                sample_df = df.tail(5).copy()
                if 'datetime' in sample_df.columns:
                    sample_df['datetime'] = sample_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                print(sample_df)
            
            # Check for date distribution in the data
            if 'datetime' in df.columns:
                # Verify the datetime values are actual datetime objects
                if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    print("Converting datetime column to datetime type")
                    df['datetime'] = pd.to_datetime(df['datetime'])
                
                # Calculate date distribution stats
                date_counts = df['datetime'].dt.date.value_counts().sort_index()
                unique_dates = df['datetime'].dt.date.unique()
                
                print(f"Date distribution of passenger flow data:")
                for date, count in date_counts.items():
                    print(f"  {date}: {count} records")
                
                print(f"Exporting data for {len(unique_dates)} unique dates from {min(unique_dates)} to {max(unique_dates)}")
                
                # Check if all dates in the simulation period are present
                start_date = self.config.start_date.date()
                end_date = self.config.end_date.date()
                all_dates = set([(start_date + timedelta(days=i)) for i in range((end_date - start_date).days + 1)])
                missing_dates = all_dates - set(unique_dates)
                
                if missing_dates:
                    print(f"WARNING: Missing data for {len(missing_dates)} dates:")
                    for date in sorted(list(missing_dates)):
                        print(f"  - {date}")
            
            output_file = os.path.join(output_dir, 'passenger_flow_results.csv')
            df.to_csv(output_file, index=False)
            print(f"Exported {len(self.passenger_flow_results)} passenger flow records to {output_file}")
        
        # Export bus positions results
        if self.bus_positions_results:
            df = pd.DataFrame(self.bus_positions_results)
            
            # Convert datetime if needed
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
                
            output_file = os.path.join(output_dir, 'bus_positions_results.csv')
            df.to_csv(output_file, index=False)
            print(f"Exported {len(self.bus_positions_results)} bus position records to {output_file}")
        
        # Export buses
        output_file = os.path.join(output_dir, 'buses.csv')
        self.bus_manager.export_buses(output_file)
        
        # Export line schedules
        output_file = os.path.join(output_dir, 'line_schedules.csv')
        self.schedule_generator.export_line_schedules(output_file)
        
        # Export bus assignments
        output_file = os.path.join(output_dir, 'bus_assignments.csv')
        self.schedule_generator.export_bus_assignments(output_file)
        
        # Export transfer statistics if available
        if hasattr(self.data, 'transfer_statistics') and self.data.transfer_statistics:
            try:
                # Clean up nested lists for CSV export
                cleaned_transfers = []
                for entry in self.data.transfer_statistics:
                    entry_copy = entry.copy()
                    if 'lines' in entry_copy:
                        entry_copy['lines'] = ','.join(map(str, entry_copy['lines']))
                    cleaned_transfers.append(entry_copy)
                    
                df = pd.DataFrame(cleaned_transfers)
                output_file = os.path.join(output_dir, 'transfer_statistics.csv')
                df.to_csv(output_file, index=False)
                print(f"Exported {len(cleaned_transfers)} transfer statistics records to {output_file}")
            except Exception as e:
                print(f"Error exporting transfer statistics: {str(e)}")
        
        # Export long wait incidents if available
        if hasattr(self.data, 'long_wait_incidents') and self.data.long_wait_incidents:
            try:
                df = pd.DataFrame(self.data.long_wait_incidents)
                output_file = os.path.join(output_dir, 'long_wait_incidents.csv')
                df.to_csv(output_file, index=False)
                print(f"Exported {len(self.data.long_wait_incidents)} long wait incident records to {output_file}")
            except Exception as e:
                print(f"Error exporting long wait incidents: {str(e)}")
        
        # Export schedule performance if available
        if hasattr(self, 'schedule_performance') and self.schedule_performance:
            try:
                df = pd.DataFrame(self.schedule_performance)
                output_file = os.path.join(output_dir, 'schedule_performance.csv')
                df.to_csv(output_file, index=False)
                print(f"Exported {len(self.schedule_performance)} schedule performance records to {output_file}")
            except Exception as e:
                print(f"Error exporting schedule performance: {str(e)}")
                
        # Export summary statistics
        try:
            summary = self.get_summary_statistics()
            output_file = os.path.join(output_dir, 'summary_statistics.json')
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Exported summary statistics to {output_file}")
        except Exception as e:
            print(f"Error exporting summary statistics: {str(e)}")

    def get_summary_statistics(self) -> Dict:
        """
        Generate summary statistics from simulation results.
        
        Returns:
        --------
        Dict
            Dictionary of summary statistics.
        """
        if not self.passenger_flow_results:
            return {"status": "No results available"}
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.passenger_flow_results)
            
            # Basic statistics
            total_records = len(df)
            total_boardings = df['boarding'].sum()
            total_alightings = df['alighting'].sum()
            
            # Overcrowding statistics
            overcrowded_records = len(df[df['occupancy_rate'] > 1.0])
            overcrowding_percentage = (overcrowded_records / total_records) * 100
            
            severe_overcrowding_records = len(df[df['occupancy_rate'] > 1.5])
            severe_overcrowding_percentage = (severe_overcrowding_records / total_records) * 100
            
            # Stop visits statistics
            total_stop_visits = total_records
            stops_with_activity = len(df[(df['boarding'] > 0) | (df['alighting'] > 0)])
            stops_without_activity = total_stop_visits - stops_with_activity
            stops_without_activity_percentage = (stops_without_activity / total_stop_visits) * 100 if total_stop_visits > 0 else 0
            
            # Overcrowding by line
            overcrowding_by_line = {}
            for line_id in df['line_id'].unique():
                line_df = df[df['line_id'] == line_id]
                line_records = len(line_df)
                line_overcrowded = len(line_df[line_df['occupancy_rate'] > 1.0])
                line_percentage = (line_overcrowded / line_records) * 100 if line_records > 0 else 0
                
                overcrowding_by_line[line_id] = {
                    'total_records': line_records,
                    'overcrowded_records': line_overcrowded,
                    'percentage': line_percentage
                }
            
            # Highest occupancy records
            max_loads = df.sort_values('occupancy_rate', ascending=False).head(10)[
                ['datetime', 'line_id', 'stop_id', 'bus_id', 'new_load', 'capacity', 'occupancy_rate']
            ].to_dict(orient='records')
            
            # Transfer statistics if available
            transfer_stats = {}
            if hasattr(self.data, 'transfer_statistics') and self.data.transfer_statistics:
                transfer_df = pd.DataFrame(self.data.transfer_statistics)
                transfer_stats = {
                    'total_transfers': int(transfer_df['total_transfers'].sum()),
                    'transfer_count': len(transfer_df),
                    'average_per_transfer': float(transfer_df['total_transfers'].mean()),
                    'max_transfer_count': int(transfer_df['total_transfers'].max()),
                    'top_transfer_stops': []
                }
                
                # Top transfer stops
                if 'stop_id' in transfer_df.columns:
                    top_stops = transfer_df.groupby('stop_id')['total_transfers'].sum().reset_index()
                    top_stops = top_stops.sort_values('total_transfers', ascending=False).head(5)
                    transfer_stats['top_transfer_stops'] = top_stops.to_dict(orient='records')
            
            # Wait time statistics if available
            wait_time_stats = {}
            if hasattr(self.data, 'long_wait_incidents') and self.data.long_wait_incidents:
                wait_df = pd.DataFrame(self.data.long_wait_incidents)
                wait_time_stats = {
                    'total_incidents': len(wait_df),
                    'average_wait_time': float(wait_df['wait_time'].mean()),
                    'max_wait_time': float(wait_df['wait_time'].max()),
                    'total_affected_passengers': int(wait_df['boarding'].sum())
                }
                
                # Incidents by line
                if 'line_id' in wait_df.columns:
                    by_line = wait_df.groupby('line_id').size().to_dict()
                    wait_time_stats['incidents_by_line'] = by_line
            
            # Schedule adherence if available
            schedule_stats = {}
            if hasattr(self, 'schedule_performance') and self.schedule_performance:
                sched_df = pd.DataFrame(self.schedule_performance)
                on_time = len(sched_df[sched_df['status'] == 'on_time'])
                late = len(sched_df[sched_df['status'] == 'late'])
                early = len(sched_df[sched_df['status'] == 'early'])
                total = len(sched_df)
                
                schedule_stats = {
                    'total_tracked_stops': total,
                    'on_time_percentage': (on_time / total) * 100 if total > 0 else 0,
                    'late_percentage': (late / total) * 100 if total > 0 else 0,
                    'early_percentage': (early / total) * 100 if total > 0 else 0,
                    'average_delay': float(sched_df['delay_minutes'].mean()),
                    'max_delay': float(sched_df['delay_minutes'].max())
                }
                
                # Worst performing lines
                if 'line_id' in sched_df.columns:
                    line_performance = (
                        sched_df.groupby('line_id')
                        .apply(lambda x: (len(x[x['status'] == 'on_time']) / len(x)) * 100 if len(x) > 0 else 0)
                        .reset_index()
                        .rename(columns={0: 'on_time_percentage'})
                        .sort_values('on_time_percentage')
                    )
                    schedule_stats['worst_lines'] = line_performance.head(5).to_dict(orient='records')
            
            # Passenger flow patterns
            time_patterns = {}
            if 'hour' in df.columns:
                by_hour = df.groupby('hour').agg({
                    'boarding': 'sum',
                    'alighting': 'sum'
                }).reset_index()
                time_patterns['by_hour'] = by_hour.to_dict(orient='records')
            
            # Bus utilization
            bus_utilization = {}
            if 'bus_id' in df.columns:
                # Average occupancy by bus
                bus_avg_occupancy = df.groupby('bus_id')['occupancy_rate'].mean().reset_index()
                bus_avg_occupancy = bus_avg_occupancy.sort_values('occupancy_rate', ascending=False)
                bus_utilization['highest_avg_occupancy'] = bus_avg_occupancy.head(10).to_dict(orient='records')
                
                # Total passengers carried by bus
                bus_passengers = df.groupby('bus_id')['boarding'].sum().reset_index()
                bus_passengers = bus_passengers.sort_values('boarding', ascending=False)
                bus_utilization['most_passengers'] = bus_passengers.head(10).to_dict(orient='records')
            
            # Combine all statistics
            summary = {
                'total_passenger_records': total_records,
                'total_boardings': int(total_boardings),
                'total_alightings': int(total_alightings),
                'overcrowded_records': overcrowded_records,
                'overcrowding_percentage': float(overcrowding_percentage),
                'severe_overcrowding_records': severe_overcrowding_records,
                'severe_overcrowding_percentage': float(severe_overcrowding_percentage),
                'total_stop_visits': total_stop_visits,
                'stops_with_activity': stops_with_activity,
                'stops_without_activity': stops_without_activity,
                'stops_without_activity_percentage': float(stops_without_activity_percentage),
                'overcrowding_by_line': overcrowding_by_line,
                'max_loads': max_loads,
                'bus_utilization': bus_utilization,
                'time_patterns': time_patterns
            }
            
            # Add optional statistics if available
            if transfer_stats:
                summary['transfer_statistics'] = transfer_stats
            if wait_time_stats:
                summary['wait_time_statistics'] = wait_time_stats
            if schedule_stats:
                summary['schedule_adherence'] = schedule_stats
            
            return summary
            
        except Exception as e:
            return {
                "status": "Error generating statistics",
                "error": str(e)
            }

    def _handle_end_of_operations(self, end_time: datetime, is_end_of_simulation: bool = False) -> None:
        """
        Handle buses at the end of operational hours.
        
        Parameters:
        -----------
        end_time : datetime
            End of the operational period or simulation.
        is_end_of_simulation : bool
            True if this is the end of the entire simulation, False otherwise.
        """
        # Get active buses at this time
        active_buses = self.bus_manager.get_active_buses(end_time)
        
        for bus_id in active_buses:
            # Skip processing if bus_id is not a string
            if not isinstance(bus_id, str):
                bus_id = str(bus_id)
                
            # Get bus details
            bus = self.data.buses.get(bus_id)
            if not bus or bus.status == "maintenance":
                continue
            
            # Get current assignment
            current_assignment = None
            for assignment in self.data.bus_assignments.get(bus_id, []):
                if assignment.start_time <= end_time < assignment.end_time:
                    current_assignment = assignment
                    break
            
            if not current_assignment:
                continue
                
            # If this is the end of simulation, just record the final state
            if is_end_of_simulation:
                # Record the final position
                self.bus_positions_results.append({
                    'datetime': end_time,
                    'bus_id': bus_id,
                    'line_id': current_assignment.line_id,
                    'stop_index': bus.current_position[0] if bus.current_position else 0,
                    'progress': bus.current_position[1] if bus.current_position else 0.0,
                    'current_load': bus.current_load,
                    'status': 'end_of_simulation'
                })
                
                # If there are still passengers, they're dropped off at the last stop
                if bus.current_load > 0:
                    # Record passenger alighting at end of service
                    line_id = current_assignment.line_id
                    line = self.data.lines.get(line_id)
                    
                    if line and bus.current_position:
                        stop_index = bus.current_position[0]
                        if 0 <= stop_index < len(line.stops):
                            stop_id = line.stops[stop_index]
                            
                            # All passengers alight
                            alighting = bus.current_load
                            
                            # Record this passenger activity
                            self.passenger_flow_results.append({
                                'datetime': end_time,
                                'hour': end_time.hour,
                                'minute': end_time.minute,
                                'date': end_time.date(),  # Explicitly store date
                                'line_id': line_id,
                                'stop_id': stop_id,
                                'bus_id': bus_id,
                                'boarding': 0,
                                'alighting': alighting,
                                'current_load': bus.current_load,
                                'new_load': 0,
                                'capacity': bus.capacity,
                                'occupancy_rate': 0.0,
                                'status': 'end_of_service'
                            })
                            
                            # Update bus load to zero
                            self.bus_manager.update_bus_load(bus_id, 0)
            else:
                # For normal end of operational hours, mark bus as returning to depot
                # and have passengers alight at the nearest stop
                
                # Get the current line and position
                line_id = current_assignment.line_id
                line = self.data.lines.get(line_id)
                
                if line and bus.current_position:
                    stop_index = bus.current_position[0]
                    progress = bus.current_position[1]
                    
                    # Determine nearest stop for end of service
                    if progress < 0.5 and stop_index > 0:
                        # Closer to previous stop
                        nearest_stop_index = stop_index
                    else:
                        # Closer to next stop or at a stop already
                        nearest_stop_index = min(stop_index + 1, len(line.stops) - 1)
                    
                    nearest_stop_id = line.stops[nearest_stop_index]
                    
                    # All passengers alight
                    alighting = bus.current_load
                    
                    if alighting > 0:
                        # Record this passenger activity
                        self.passenger_flow_results.append({
                            'datetime': end_time,
                            'hour': end_time.hour,
                            'minute': end_time.minute,
                            'date': end_time.date(),  # Explicitly store date
                            'line_id': line_id,
                            'stop_id': nearest_stop_id,
                            'bus_id': bus_id,
                            'boarding': 0,
                            'alighting': alighting,
                            'current_load': bus.current_load,
                            'new_load': 0,
                            'capacity': bus.capacity,
                            'occupancy_rate': 0.0,
                            'status': 'end_of_service'
                        })
                    
                    # Mark the bus as returning to depot with no passengers
                    self.bus_manager.update_bus_load(bus_id, 0)
                    
                    # Record the bus returning to depot
                    self.bus_positions_results.append({
                        'datetime': end_time,
                        'bus_id': bus_id,
                        'line_id': line_id,
                        'stop_index': nearest_stop_index,
                        'progress': 0.0,  # At the stop
                        'current_load': 0,
                        'status': 'returning_to_depot'
                    })