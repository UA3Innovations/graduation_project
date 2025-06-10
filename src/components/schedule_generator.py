"""
Schedule generation module for creating line schedules and bus assignments.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from ..core.data_models import BusTransitData, LineScheduleEntry, BusAssignment, Bus
from .transit_network import TransitNetwork


class ScheduleGenerator:
    """
    Generates line schedules and bus assignments based on demand patterns and constraints.
    """
    
    # Time periods
    EARLY_MORNING = (5, 7)    # 5:00-7:00
    MORNING_RUSH = (7, 10)    # 7:00-10:00
    MIDDAY = (10, 16)         # 10:00-16:00
    EVENING_RUSH = (16, 20)   # 16:00-20:00
    NIGHT = (20, 24)          # 20:00-24:00
    
    # Day types
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    
    def __init__(self, data: BusTransitData, network: TransitNetwork):
        """Initialize the schedule generator."""
        self.data = data
        self.network = network
    
    def generate_line_schedules(self, date: str) -> Dict[str, List[LineScheduleEntry]]:
        """
        Generate schedules for all lines for a specific date.
        
        Parameters:
        -----------
        date : str
            Date in 'YYYY-MM-DD' format.
            
        Returns:
        --------
        Dict[str, List[LineScheduleEntry]]
            Dictionary of line_id -> list of schedule entries
        """
        # Parse date
        dt = datetime.strptime(date, '%Y-%m-%d')
        weekday = dt.weekday()
        date_str = f"{dt.month:02d}-{dt.day:02d}"
        
        # Determine day type
        if date_str in self.data.holidays:
            day_type = self.HOLIDAY
        elif weekday >= 5:  # 5=Saturday, 6=Sunday
            day_type = self.WEEKEND
        else:
            day_type = self.WEEKDAY
        
        # Generate schedules for each line
        schedules = {}
        for line_id, line in self.data.lines.items():
            line_schedules = self._generate_line_schedule(line_id, dt, day_type)
            schedules[line_id] = line_schedules
        
        # Store in data container - append to existing schedules if any
        if not hasattr(self.data, 'all_line_schedules'):
            self.data.all_line_schedules = {}
            
        # Store schedules indexed by date
        self.data.all_line_schedules[date] = schedules
        self.data.line_schedules = schedules
        
        print(f"Generated {sum(len(schedules[line_id]) for line_id in schedules)} schedule entries for {date}")
        
        return schedules
    
    def assign_buses_to_schedules(self, date: str) -> Dict[str, List[BusAssignment]]:
        """
        Assign buses to line schedules for a specific date.
        
        Parameters:
        -----------
        date : str
            Date in 'YYYY-MM-DD' format.
            
        Returns:
        --------
        Dict[str, List[BusAssignment]]
            Dictionary of bus_id -> list of assignments
        """
        # Parse date
        dt = datetime.strptime(date, '%Y-%m-%d')
        
        # Make sure we have schedules for this date
        if not self.data.line_schedules:
            self.generate_line_schedules(date)
        
        # Assign buses exclusively to lines
        buses_per_line = {}
        available_buses = list(self.data.buses.keys())
        random.shuffle(available_buses)  # Randomize bus selection
        
        # Distribute buses evenly among lines
        lines = list(self.data.lines.keys())
        buses_per_line = {line_id: [] for line_id in lines}
        
        # Allocate buses to lines (ensure each line gets some buses)
        buses_per_line_count = min(10, len(available_buses) // max(1, len(lines)))
        for line_id in lines:
            if not available_buses:
                break
            # Assign buses_per_line_count buses to each line
            for _ in range(buses_per_line_count):
                if available_buses:
                    buses_per_line[line_id].append(available_buses.pop(0))
        
        # Distribute any remaining buses
        for line_id in lines:
            if not available_buses:
                break
            buses_per_line[line_id].append(available_buses.pop(0))
        
        # Create assignments for each line
        all_assignments = {}
        for line_id, schedule_entries in self.data.line_schedules.items():
            # Get buses assigned to this line
            line_buses = buses_per_line.get(line_id, [])
            if not line_buses:
                print(f"Warning: No buses available for line {line_id}")
                continue
            
            # Sort entries by departure time
            sorted_entries = sorted(schedule_entries, key=lambda e: e.departure_time)
            
            # Initialize bus availability - all available at start of day
            bus_availability = {bus_id: dt for bus_id in line_buses}
            
            # Skip some departures if we have too many for the available buses
            max_departures_per_bus = 14  # Reasonable number of trips per day per bus
            max_departures = len(line_buses) * max_departures_per_bus
            
            if len(sorted_entries) > max_departures:
                # If we have too many departures, keep regularly spaced ones
                keep_ratio = max_departures / len(sorted_entries)
                filtered_entries = []
                for i, entry in enumerate(sorted_entries):
                    if i % int(1/keep_ratio) == 0:  # Keep every Nth entry
                        filtered_entries.append(entry)
                sorted_entries = filtered_entries
                print(f"Reduced schedule for line {line_id} from {len(schedule_entries)} to {len(sorted_entries)} departures")
            
            # Create assignments
            for entry in sorted_entries:
                # Find the earliest available bus
                available_bus_id = None
                earliest_time = dt + timedelta(days=1)  # Default to next day
                
                for bus_id, available_time in bus_availability.items():
                    if available_time <= entry.departure_time and available_time < earliest_time:
                        available_bus_id = bus_id
                        earliest_time = available_time
                
                if not available_bus_id:
                    # Skip this departure instead of warning
                    continue
                
                # Calculate route duration
                route_time = self.network.get_route_travel_time(line_id, entry.departure_time)
                
                # Add buffer time for end-of-line breaks
                buffer_time = random.randint(5, 15)  # 5-15 minute break
                return_time = entry.departure_time + timedelta(minutes=route_time + buffer_time)
                
                # Create assignment
                if available_bus_id not in all_assignments:
                    all_assignments[available_bus_id] = []
                
                all_assignments[available_bus_id].append(
                    BusAssignment(
                        bus_id=available_bus_id,
                        line_id=line_id,
                        start_time=entry.departure_time,
                        end_time=return_time,
                        status="scheduled"
                    )
                )
                
                # Update bus availability
                bus_availability[available_bus_id] = return_time
                
                # Update the schedule entry with the assigned bus
                entry.assigned_bus_id = available_bus_id
        
        # Store in data container
        if not hasattr(self.data, 'all_bus_assignments'):
            self.data.all_bus_assignments = {}
            
        # Store assignments indexed by date
        self.data.all_bus_assignments[date] = all_assignments
        self.data.bus_assignments = all_assignments
        
        print(f"Generated {sum(len(assignments) for bus_id, assignments in all_assignments.items())} bus assignments for {date}")
        
        return all_assignments
    
    def export_line_schedules(self, filepath: str) -> None:
        """
        Export line schedules to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the output CSV file.
        """
        # Flatten schedules for export
        flattened = []
        
        # Check if we have historical schedules
        if hasattr(self.data, 'all_line_schedules'):
            # Export all schedules from the entire simulation period
            for date, schedules in self.data.all_line_schedules.items():
                for line_id, entries in schedules.items():
                    for entry in entries:
                        flattened.append({
                            'date': date,
                            'line_id': entry.line_id,
                            'departure_time': entry.departure_time,
                            'assigned_bus_id': entry.assigned_bus_id if entry.assigned_bus_id else '',
                            'status': entry.status
                        })
        else:
            # Fall back to current schedules only
            for line_id, entries in self.data.line_schedules.items():
                for entry in entries:
                    flattened.append({
                        'line_id': entry.line_id,
                        'departure_time': entry.departure_time,
                        'assigned_bus_id': entry.assigned_bus_id if entry.assigned_bus_id else '',
                        'status': entry.status
                    })
        
        # Convert to DataFrame and export
        if flattened:
            df = pd.DataFrame(flattened)
            df.to_csv(filepath, index=False)
            print(f"Exported {len(flattened)} line schedule entries to {filepath}")
        else:
            print("No line schedules to export.")
    
    def export_bus_assignments(self, filepath: str) -> None:
        """
        Export bus assignments to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the output CSV file.
        """
        # Flatten assignments for export
        flattened = []
        
        # Check if we have historical assignments
        if hasattr(self.data, 'all_bus_assignments'):
            # Export all assignments from the entire simulation period
            for date, assignments in self.data.all_bus_assignments.items():
                for bus_id, bus_assignments in assignments.items():
                    for assignment in bus_assignments:
                        flattened.append({
                            'date': date,
                            'bus_id': assignment.bus_id,
                            'line_id': assignment.line_id,
                            'start_time': assignment.start_time,
                            'end_time': assignment.end_time,
                            'status': assignment.status
                        })
        else:
            # Fall back to current assignments only
            for bus_id, bus_assignments in self.data.bus_assignments.items():
                for assignment in bus_assignments:
                    flattened.append({
                        'bus_id': assignment.bus_id,
                        'line_id': assignment.line_id,
                        'start_time': assignment.start_time,
                        'end_time': assignment.end_time,
                        'status': assignment.status
                    })
        
        # Convert to DataFrame and export
        if flattened:
            df = pd.DataFrame(flattened)
            df.to_csv(filepath, index=False)
            print(f"Exported {len(flattened)} bus assignments to {filepath}")
        else:
            print("No bus assignments to export.")
    
    def _generate_line_schedule(self, line_id: str, date: datetime, day_type: str) -> List[LineScheduleEntry]:
        """
        Generate a schedule for a specific line.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        date : datetime
            Date for the schedule.
        day_type : str
            Type of day (weekday, weekend, holiday).
            
        Returns:
        --------
        List[LineScheduleEntry]
            List of schedule entries for the line.
        """
        # Determine if this is a crowded line
        is_crowded = line_id in self.network.CROWDED_LINES
        
        # Define intervals (in minutes) between departures for different periods and day types
        if day_type == self.WEEKEND or day_type == self.HOLIDAY:
            intervals = {
                self.EARLY_MORNING: random.randint(30, 45),  # 30-45 mins
                self.MORNING_RUSH: random.randint(15, 25),   # 15-25 mins
                self.MIDDAY: random.randint(20, 30),         # 20-30 mins
                self.EVENING_RUSH: random.randint(15, 25),   # 15-25 mins
                self.NIGHT: random.randint(35, 60)           # 35-60 mins
            }
        else:  # Weekday
            # Shorter intervals for crowded lines
            crowd_factor = 0.7 if is_crowded else 1.0
            
            intervals = {
                self.EARLY_MORNING: int(random.randint(20, 30) * crowd_factor),
                self.MORNING_RUSH: int(random.randint(8, 15) * crowd_factor),
                self.MIDDAY: int(random.randint(12, 20) * crowd_factor),
                self.EVENING_RUSH: int(random.randint(8, 15) * crowd_factor),
                self.NIGHT: int(random.randint(25, 45) * crowd_factor)
            }
        
        # Generate departures for each time period
        all_departures = []
        
        # Early morning (5:00-7:00)
        departures = self._generate_departures_for_period(
            date, self.EARLY_MORNING, intervals[self.EARLY_MORNING]
        )
        all_departures.extend(departures)
        
        # Morning rush (7:00-10:00)
        departures = self._generate_departures_for_period(
            date, self.MORNING_RUSH, intervals[self.MORNING_RUSH]
        )
        all_departures.extend(departures)
        
        # Midday (10:00-16:00)
        departures = self._generate_departures_for_period(
            date, self.MIDDAY, intervals[self.MIDDAY]
        )
        all_departures.extend(departures)
        
        # Evening rush (16:00-20:00)
        departures = self._generate_departures_for_period(
            date, self.EVENING_RUSH, intervals[self.EVENING_RUSH]
        )
        all_departures.extend(departures)
        
        # Night (20:00-24:00)
        departures = self._generate_departures_for_period(
            date, self.NIGHT, intervals[self.NIGHT]
        )
        all_departures.extend(departures)
        
        # Sort departures chronologically
        all_departures.sort()
        
        # Create schedule entries
        schedule_entries = [
            LineScheduleEntry(
                line_id=line_id,
                departure_time=departure,
                assigned_bus_id=None,  # Will be assigned later
                status="scheduled"
            )
            for departure in all_departures
        ]
        
        return schedule_entries
    
    def _generate_departures_for_period(self, date: datetime, period: Tuple[int, int], 
                                    interval_mins: int) -> List[datetime]:
        """
        Generate departure times for a specific time period.
        
        Parameters:
        -----------
        date : datetime
            Base date.
        period : Tuple[int, int]
            Period as (start_hour, end_hour).
        interval_mins : int
            Base interval between departures in minutes.
            
        Returns:
        --------
        List[datetime]
            List of departure datetimes.
        """
        departures = []
        
        # Set start time
        start_time = date.replace(hour=period[0], minute=0)
        # If period[1] is 24, change it to 23:59
        if period[1] == 24:
            end_time = date.replace(hour=23, minute=59)
        else:
            end_time = date.replace(hour=period[1], minute=0)
        
        current_time = start_time
        while current_time < end_time:
            interval_variation = random.uniform(0.8, 1.2)
            actual_interval = int(interval_mins * interval_variation)
            
            departures.append(current_time)
            current_time += timedelta(minutes=actual_interval)
        
        return departures
