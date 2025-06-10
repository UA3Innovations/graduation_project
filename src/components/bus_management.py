"""
Bus fleet management module for creating and tracking buses.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from ..core.data_models import BusTransitData, Bus, BusType, BusAssignment


class BusManager:
    """
    Manages the bus fleet, including creation, assignments, and status tracking.
    """
    
    # Bus types and capacities
    BUS_TYPES = {
        'regular': 100,   # Regular bus capacity
        'articulated': 180  # Articulated (bendy) bus capacity
    }
    
    def __init__(self, data: BusTransitData):
        """Initialize the bus manager."""
        self.data = data
        
        # Initialize standard bus types
        self.data.bus_types = {
            "regular": BusType(
                type_name="regular",
                capacity=self.BUS_TYPES['regular'],
                description="Standard city bus"
            ),
            "articulated": BusType(
                type_name="articulated",
                capacity=self.BUS_TYPES['articulated'],
                description="Articulated (bendy) bus for high-capacity routes"
            )
        }
    
    def generate_buses(self, num_buses_per_line: int = 10) -> Dict[str, Bus]:
        """
        Generate a fleet of buses with different capacities.
        
        Parameters:
        -----------
        num_buses_per_line : int
            Number of buses to allocate per line (default: 10).
            
        Returns:
        --------
        Dict[str, Bus]
            Dictionary of bus_id -> Bus objects
        """
        buses = {}
        bus_id_counter = 1001
        
        for line_id in self.data.lines.keys():
            # Determine if this is a crowded line
            is_crowded_line = line_id in self.data.crowded_lines
            
            # Determine bus type distribution
            if is_crowded_line:
                articulated_count = random.randint(6, 8)  # 60-80% articulated buses for crowded lines
            else:
                articulated_count = random.randint(3, 5)  # 30-50% articulated buses for other lines
            
            regular_count = num_buses_per_line - articulated_count
            
            # Create articulated buses
            for i in range(articulated_count):
                bus_id = str(bus_id_counter)
                buses[bus_id] = Bus(
                    bus_id=bus_id,
                    bus_type="articulated",
                    capacity=self.BUS_TYPES['articulated'],
                    status="idle",
                    current_line=None,  # Not assigned to any line by default
                    current_position=None,
                    current_load=0
                )
                bus_id_counter += 1
            
            # Create regular buses
            for i in range(regular_count):
                bus_id = str(bus_id_counter)
                buses[bus_id] = Bus(
                    bus_id=bus_id,
                    bus_type="regular",
                    capacity=self.BUS_TYPES['regular'],
                    status="idle",
                    current_line=None,  # Not assigned to any line by default
                    current_position=None,
                    current_load=0
                )
                bus_id_counter += 1
        
        # Store in data container
        self.data.buses = buses
        
        return buses
    
    def export_buses(self, filepath: str) -> None:
        """
        Export bus fleet to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file.
        """
        # Prepare data for CSV
        rows = []
        for bus_id, bus in self.data.buses.items():
            rows.append({
                'bus_id': bus_id,
                'bus_type': bus.bus_type,
                'capacity': bus.capacity,
                'status': bus.status,
                'current_line': bus.current_line if bus.current_line else ''
            })
        
        # Create dataframe and save
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(rows)} buses to {filepath}")
    
    def update_bus_position(self, bus_id: str, line_id: str, 
                          stop_index: int, progress: float) -> None:
        """
        Update the current position of a bus.
        
        Parameters:
        -----------
        bus_id : str
            ID of the bus.
        line_id : str
            ID of the line the bus is servicing.
        stop_index : int
            Index of the current/next stop in the route (0-based).
        progress : float
            Progress between current and next stop (0.0 to 1.0).
        """
        bus = self.data.buses.get(bus_id)
        if not bus:
            print(f"Warning: Bus {bus_id} not found")
            return
        
        # Update bus attributes
        bus.current_line = line_id
        bus.current_position = (stop_index, progress)
        bus.status = "in_service"
        
        # Update position in tracking dictionary
        self.data.bus_positions[bus_id] = (line_id, stop_index, progress)
    
    def update_bus_load(self, bus_id: str, current_load: int) -> None:
        """
        Update the current passenger load of a bus.
        
        Parameters:
        -----------
        bus_id : str
            ID of the bus.
        current_load : int
            Current number of passengers.
        """
        bus = self.data.buses.get(bus_id)
        if not bus:
            print(f"Warning: Bus {bus_id} not found")
            return
        
        bus.current_load = current_load
    
    def get_active_buses(self, dt: datetime) -> List[str]:
        """
        Get all buses that are active (in service) at a specific time.
        
        Parameters:
        -----------
        dt : datetime
            The datetime to check.
            
        Returns:
        --------
        List[str]
            List of active bus IDs.
        """
        active_buses = []
        
        # Check each bus's assignments
        for bus_id, assignments in self.data.bus_assignments.items():
            for assignment in assignments:
                if assignment.start_time <= dt < assignment.end_time:
                    active_buses.append(bus_id)
                    break
        
        return active_buses
    
    def get_bus_at_stop(self, line_id: str, stop_id: int, dt: datetime) -> Optional[str]:
        """
        Find a bus that is at a specific stop at a specific time.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        stop_id : int
            ID of the stop.
        dt : datetime
            The datetime to check.
            
        Returns:
        --------
        Optional[str]
            ID of the bus if found, None otherwise.
        """
        # Get active buses on this line
        active_buses = []
        for bus_id, assignments in self.data.bus_assignments.items():
            for assignment in assignments:
                if assignment.line_id == line_id and assignment.start_time <= dt < assignment.end_time:
                    active_buses.append(bus_id)
                    break
        
        # Check if any of these buses are at this stop
        for bus_id in active_buses:
            if bus_id in self.data.bus_positions:
                bus_line_id, stop_index, progress = self.data.bus_positions[bus_id]
                
                # If the bus is on the correct line
                if bus_line_id == line_id:
                    # Get the stop list for this line
                    line = self.data.lines.get(line_id)
                    if line and stop_index < len(line.stops):
                        # Check if the bus is at this stop (progress near 0 or 1)
                        current_stop = line.stops[stop_index]
                        if current_stop == stop_id and 0.0 <= progress <= 0.1:
                            return bus_id
                        
                        # Check if approaching next stop
                        if stop_index > 0 and line.stops[stop_index-1] == stop_id and 0.9 <= progress <= 1.0:
                            return bus_id
        
        return None
    
    def get_available_buses_for_line(self, line_id: str, dt: datetime) -> List[str]:
        """
        Find buses that are available to service a specific line at a specific time.
        
        Parameters:
        -----------
        line_id : str
            ID of the line.
        dt : datetime
            The datetime to check.
            
        Returns:
        --------
        List[str]
            List of available bus IDs.
        """
        available_buses = []
        
        for bus_id, bus in self.data.buses.items():
            # Skip buses that are out of service
            if bus.status == "maintenance":
                continue
            
            # Check if the bus is already assigned at this time
            is_assigned = False
            for assignment in self.data.bus_assignments.get(bus_id, []):
                if assignment.start_time <= dt < assignment.end_time:
                    is_assigned = True
                    break
            
            if not is_assigned:
                available_buses.append(bus_id)
        
        return available_buses
    
    def maintenance_schedule(self, start_date: str, end_date: str, 
                           maintenance_rate: float = 0.05) -> None:
        """
        Schedule random maintenance periods for buses.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        maintenance_rate : float
            Probability of a bus going into maintenance on any given day.
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Iterate through each day
        current_dt = start_dt
        while current_dt <= end_dt:
            # Check each bus
            for bus_id, bus in self.data.buses.items():
                # Random chance of maintenance
                if random.random() < maintenance_rate:
                    # Schedule 4-12 hour maintenance during non-peak hours
                    maintenance_duration = random.randint(4, 12)
                    
                    # Initialize maintenance time slots to try
                    maintenance_slots = [
                        # Night maintenance (22:00-05:00)
                        (current_dt.replace(hour=22, minute=0), maintenance_duration),
                        # Early morning (4:00-8:00)
                        (current_dt.replace(hour=4, minute=0), maintenance_duration),
                        # Mid-day off-peak (10:00-15:00)
                        (current_dt.replace(hour=10, minute=0), maintenance_duration),
                        # Late night (1:00-5:00 next day)
                        ((current_dt + timedelta(days=1)).replace(hour=1, minute=0), maintenance_duration)
                    ]
                    
                    # Get existing assignments for this bus
                    existing_assignments = self.data.bus_assignments.get(bus_id, [])
                    
                    # Find a slot that doesn't overlap with existing assignments
                    valid_slot_found = False
                    
                    for start_time, duration in maintenance_slots:
                        end_time = start_time + timedelta(hours=duration)
                        
                        # Check for overlaps with existing assignments
                        has_overlap = False
                        for assignment in existing_assignments:
                            # Check if the proposed maintenance overlaps with existing assignment
                            if (start_time < assignment.end_time and 
                                end_time > assignment.start_time):
                                has_overlap = True
                                break
                        
                        if not has_overlap:
                            # Found a valid slot - schedule maintenance
                            valid_slot_found = True
                            
                            # Create a maintenance "assignment"
                            if bus_id not in self.data.bus_assignments:
                                self.data.bus_assignments[bus_id] = []
                            
                            # Add the maintenance assignment
                            self.data.bus_assignments[bus_id].append(
                                BusAssignment(
                                    bus_id=bus_id,
                                    line_id="MAINTENANCE",
                                    start_time=start_time,
                                    end_time=end_time,
                                    status="maintenance"
                                )
                            )
                            
                            # Log maintenance scheduling if debug is enabled
                            if hasattr(self, 'debug') and self.debug:
                                print(f"Scheduled maintenance for bus {bus_id} from "
                                      f"{start_time.strftime('%Y-%m-%d %H:%M')} to "
                                      f"{end_time.strftime('%Y-%m-%d %H:%M')}")
                            
                            break
                    
                    # If no valid slot found, we could either skip maintenance or force it
                    # In this case we'll skip it and potentially schedule it on another day
                    if not valid_slot_found and hasattr(self, 'debug') and self.debug:
                        print(f"Could not schedule maintenance for bus {bus_id} on "
                              f"{current_dt.strftime('%Y-%m-%d')} due to schedule conflicts")
            
            # Move to next day
            current_dt += timedelta(days=1)
