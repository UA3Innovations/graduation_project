"""
Core data models and structures for the bus transit simulation.
"""

import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Union


@dataclass
class Stop:
    """Represents a bus stop."""
    stop_id: int
    stop_name: str
    is_special_location: bool = False
    transfer_likelihood: float = 0.0
    base_boarding_rate: float = 20.0  # passengers per hour during normal periods
    base_alighting_ratio: float = 1.0  # ratio of alighting to boarding passengers


@dataclass
class BusType:
    """Represents a type of bus with specific characteristics."""
    type_name: str
    capacity: int
    description: str = ""


@dataclass
class Bus:
    """Represents a bus vehicle in the fleet."""
    bus_id: str
    bus_type: str
    capacity: int
    status: str = "idle"  # idle, in_service, maintenance
    current_line: Optional[str] = None
    current_position: Optional[Tuple[int, int]] = None  # (stop_index, progress)
    current_load: int = 0


@dataclass
class Line:
    """Represents a bus line/route."""
    line_id: str
    stops: List[int]  # List of stop_ids in order
    is_crowded: bool = False
    popularity_factor: float = 1.0
    

@dataclass
class LineScheduleEntry:
    """Represents a scheduled departure for a line."""
    line_id: str
    departure_time: datetime
    assigned_bus_id: Optional[str] = None
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled


@dataclass
class BusAssignment:
    """Represents a bus assignment to a line for a specific period."""
    bus_id: str
    line_id: str
    start_time: datetime
    end_time: datetime
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled


@dataclass
class TravelSegment:
    """Represents travel time between consecutive stops."""
    from_stop_id: int
    to_stop_id: int
    line_id: str
    base_time: float  # minutes
    
    # Time multipliers for different conditions
    rush_hour_factor: float = 1.8
    weekend_factor: float = 0.8
    night_factor: float = 0.7
    holiday_factor: float = 0.8
    bad_weather_factor: float = 2.0


@dataclass
class PassengerGroup:
    """Represents a group of passengers boarding or alighting."""
    stop_id: int
    line_id: str
    bus_id: str
    timestamp: datetime
    boarding_count: int
    alighting_count: int


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    start_date: datetime
    end_date: datetime
    time_step: int = 5  # minutes
    randomize_travel_times: bool = True
    randomize_passenger_demand: bool = True
    weather_effects_probability: float = 0.15
    seed: int = 42
    

class BusTransitData:
    """Container for all bus transit data."""
    
    def __init__(self):
        """Initialize the data container with empty collections."""
        self.stops = {}  # stop_id -> Stop
        self.lines = {}  # line_id -> Line
        self.buses = {}  # bus_id -> Bus
        self.bus_types = {}  # type_name -> BusType
        self.travel_segments = {}  # line_id -> {(from_stop_id, to_stop_id) -> TravelSegment}
        
        self.line_schedules = []  # List of LineScheduleEntry
        self.bus_assignments = {}  # bus_id -> List of BusAssignment

        
        self.passenger_flow = {}  # (stop_id, line_id) -> List of PassengerGroup
        self.waiting_passengers = {}  # stop_id -> count
        self.abandoned_passengers = 0  # Track passengers who give up waiting
        
        self.bus_positions = {}  # bus_id -> (line_id, stop_index, progress)
        self.last_bus_visit = {}  # (line_id, stop_id) -> datetime
        
        # Lists for special route classifications
        self.crowded_lines = []  # List of line_id's considered crowded
        
        # Holiday dates in MM-DD format (e.g., "01-01" for January 1st)
        self.holidays = []
        
        # Weather and environmental factors
        self.weather_effects_probability = 0.15  # Probability of weather affecting travel times
        
        # Initialize bus types with default values
        self.initialize_bus_types()
    
    def load_stops(self, filepath: str) -> None:
        """Load stops from a CSV file."""
        df = pd.read_csv(filepath)
        self.stops = {}
        
        for _, row in df.iterrows():
            stop_id = row['Stop ID']
            stop_name = row['Stop Name']
            
            # Determine if this is a special location
            is_special = any(location in stop_name for location in self.special_locations)
            
            # Base values will be refined later
            self.stops[stop_id] = Stop(
                stop_id=stop_id,
                stop_name=stop_name,
                is_special_location=is_special,
                transfer_likelihood=0.2,
                base_boarding_rate=20.0,
                base_alighting_ratio=1.0
            )
    
    def load_lines(self, filepath: str) -> None:
        """Load lines and their stops from a CSV file."""
        df = pd.read_csv(filepath)
        
        # Convert line IDs to strings for consistency
        df['Line ID'] = df['Line ID'].astype(str)
        
        # Get unique line IDs
        line_ids = df['Line ID'].unique()
        
        # Create mapping of lines to stops (preserving order)
        self.lines = {}
        for line_id in line_ids:
            # Get stops for this line in order
            line_stops = df[df['Line ID'] == line_id]['Stop ID'].tolist()
            
            # Determine if this is a crowded line
            is_crowded = line_id in self.crowded_lines
            
            self.lines[line_id] = Line(
                line_id=line_id,
                stops=line_stops,
                is_crowded=is_crowded,
                popularity_factor=1.5 if is_crowded else 1.0
            )
    
    def load_buses(self, filepath: str) -> None:
        """Load buses from a CSV file."""
        df = pd.read_csv(filepath)
        self.buses = {}
        
        for _, row in df.iterrows():
            bus_id = str(row['bus_id'])
            bus_type = row['bus_type']
            capacity = row['capacity']
            
            self.buses[bus_id] = Bus(
                bus_id=bus_id,
                bus_type=bus_type,
                capacity=capacity,
                status="idle",
                current_line=None,
                current_position=None,
                current_load=0
            )
    
    def initialize_bus_types(self):
        """Initialize standard bus types."""
        self.bus_types = {
            "regular": BusType(
                type_name="regular",
                capacity=100,
                description="Standard city bus"
            ),
            "articulated": BusType(
                type_name="articulated",
                capacity=180,
                description="Articulated (bendy) bus for high-capacity routes"
            )
        }
