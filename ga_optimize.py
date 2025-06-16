"""
Genetic Algorithm-based Schedule Optimizer for Bus Transit System
This module implements simulation-based schedule optimization using genetic algorithms.
"""

import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

# Import simulation components for evaluation
from data_models import BusTransitData, SimulationConfig, LineScheduleEntry, BusAssignment
from transit_network import TransitNetwork
from schedule_generator import ScheduleGenerator
from simulation_engine import SimulationEngine
from bus_management import BusManager
from passenger_generator import PassengerGenerator


@dataclass
class OptimizationConfig:
    """Configuration parameters for genetic algorithm optimization"""
    population_size: int = 30
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_size: int = 5
    tournament_size: int = 3
    
    # Optimization objectives weights
    passenger_wait_weight: float = 0.4
    bus_utilization_weight: float = 0.3
    overcrowding_weight: float = 0.2
    service_coverage_weight: float = 0.1
    
    # Schedule constraints
    min_interval: int = 5  # minimum minutes between departures
    max_interval: int = 60  # maximum minutes between departures
    operational_hours: Tuple[int, int] = (5, 24)
    
    # Simulation parameters for evaluation
    simulation_duration_hours: int = 6  # Hours to simulate for fitness evaluation
    buses_per_line: int = 10
    time_step: int = 10  # Larger time step for faster evaluation
    random_seed: int = 42


class ScheduleChromosome:
    """Represents a schedule solution as a chromosome"""
    
    def __init__(self, line_id: str, departure_times: List[datetime]):
        self.line_id = line_id
        self.departure_times = sorted(departure_times)
        self.fitness = None
        
    def __repr__(self):
        return f"ScheduleChromosome(line={self.line_id}, departures={len(self.departure_times)}, fitness={self.fitness:.4f if self.fitness else 'None'})"
    
    def copy(self):
        """Create a deep copy of the chromosome"""
        return ScheduleChromosome(self.line_id, self.departure_times.copy())


class GeneticScheduleOptimizer:
    """
    Genetic Algorithm optimizer for bus schedules using simulation-based evaluation
    """
    
    def __init__(self, config: OptimizationConfig = None, stops_file: str = None):
        """Initialize the optimizer with configuration"""
        self.config = config or OptimizationConfig()
        self.passenger_data = None
        self.line_data = None
        self.best_solutions = {}
        
        # Simulation components
        self.stops_file = stops_file or "ankara_bus_stops_10.csv"
        self.base_data = None
        self.network = None
        self.line_ids = []
        self.historical_constraints = None
        self.original_fitness_method = None
        
        # Setup simulation data if stops file provided
        if self.stops_file and os.path.exists(self.stops_file):
            self.setup_simulation_data()

    def load_historical_constraints_from_simulation(self, simulation_results_file: str):
        """Load constraints from Stage 1 simulation results"""
        import pandas as pd
        
        print(f"📊 Loading historical constraints from: {simulation_results_file}")
        df = pd.read_csv(simulation_results_file)
        
        # Extract bus usage constraints per line
        bus_limits = {}
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id]
            bus_limits[line_id] = line_data['bus_id'].nunique()
        
        # Extract hourly departure patterns
        hourly_patterns = {}
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id]
            hourly_count = line_data.groupby('hour').size()
            hourly_patterns[line_id] = hourly_count.to_dict()
        
        self.historical_constraints = {
            'total_buses': df['bus_id'].nunique(),
            'bus_limits_per_line': bus_limits,
            'hourly_patterns': hourly_patterns,
            'daily_passengers': {
                'boarding': df['boarding'].sum() / df['date'].nunique(),
                'alighting': df['alighting'].sum() / df['date'].nunique()
            }
        }
        
        print(f"✅ Loaded constraints:")
        print(f"   Total buses: {self.historical_constraints['total_buses']}")
        print(f"   Bus limits per line: {bus_limits}")
        
        # Store original fitness method and replace with constrained version
        self.original_fitness_method = self._calculate_fitness
        self._calculate_fitness = self._calculate_fitness_with_constraints
        
        return self.historical_constraints

    def _calculate_fitness_with_constraints(self, chromosome: ScheduleChromosome, date: datetime) -> float:
        """Enhanced fitness that enforces historical constraints"""
        
        if not self.historical_constraints:
            return self.original_fitness_method(chromosome, date)
        
        line_id = chromosome.line_id
        
        # HARD CONSTRAINT: Check bus usage
        required_buses = self._estimate_required_buses(chromosome)
        max_allowed = self.historical_constraints['bus_limits_per_line'].get(line_id, 6)
        
        if required_buses > max_allowed:
            return 0.0  # Zero fitness for constraint violation
        
        # Calculate base fitness using original method
        base_fitness = self.original_fitness_method(chromosome, date)
        
        if base_fitness == 0.0:
            return 0.0
        
        # PATTERN PRESERVATION: Reward schedules that match historical patterns
        pattern_score = self._calculate_pattern_matching_score(chromosome, line_id)
        
        # RESOURCE EFFICIENCY: Reward using fewer buses
        efficiency_bonus = (max_allowed - required_buses) / max_allowed * 0.1
        
        # Combined fitness
        final_fitness = base_fitness * 0.8 + pattern_score * 0.2 + efficiency_bonus
        
        return max(0.0, min(1.0, final_fitness))
    
    def _estimate_required_buses(self, chromosome: ScheduleChromosome) -> int:
        """Estimate buses needed for this schedule"""
        if not chromosome.departure_times:
            return 0
            
        # Assume 60-minute round trip time
        trip_duration = timedelta(hours=1)
        max_concurrent = 0
        
        # Check every 15-minute window
        start_time = min(chromosome.departure_times)
        end_time = max(chromosome.departure_times) + trip_duration
        
        current = start_time
        while current <= end_time:
            concurrent = sum(1 for dep in chromosome.departure_times 
                           if dep <= current <= dep + trip_duration)
            max_concurrent = max(max_concurrent, concurrent)
            current += timedelta(minutes=15)
        
        return max_concurrent
    
    def _calculate_pattern_matching_score(self, chromosome: ScheduleChromosome, line_id: str) -> float:
        """Calculate how well schedule matches historical hourly patterns"""
        if line_id not in self.historical_constraints['hourly_patterns']:
            return 0.5  # Neutral score
        
        historical_pattern = self.historical_constraints['hourly_patterns'][line_id]
        
        # Count departures by hour in chromosome
        schedule_pattern = {}
        for departure in chromosome.departure_times:
            hour = departure.hour
            schedule_pattern[hour] = schedule_pattern.get(hour, 0) + 1
        
        # Calculate similarity (simplified correlation)
        common_hours = set(historical_pattern.keys()) & set(schedule_pattern.keys())
        if len(common_hours) < 6:  # Need some overlap
            return 0.0
        
        hist_values = [historical_pattern[h] for h in sorted(common_hours)]
        sched_values = [schedule_pattern[h] for h in sorted(common_hours)]
        
        # Simple correlation calculation
        if len(hist_values) > 1:
            try:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(hist_values, sched_values)
                return max(0.0, corr)
            except:
                return 0.5
        
        return 0.5
        
    def setup_simulation_data(self):
        """Setup simulation data and network"""
        print(f"🔧 Setting up simulation data from {self.stops_file}...")
        
        # Create base data and network
        self.base_data = BusTransitData()
        self.network = TransitNetwork(self.base_data)
        
        if not self.network.load_network_data(self.stops_file):
            raise ValueError(f"Failed to load network data from {self.stops_file}")
        
        self.line_ids = list(self.base_data.lines.keys())
        print(f"✅ Loaded {len(self.base_data.stops)} stops, {len(self.line_ids)} lines")
        
        return True
    
    def load_historical_data(self, passenger_flow_file: str, line_schedules_file: str):
        """Load historical data from simulation results"""
        # Load passenger flow data
        self.passenger_data = pd.read_csv(passenger_flow_file)
        self.passenger_data['datetime'] = pd.to_datetime(self.passenger_data['datetime'])
        
        # Load line schedule data
        self.line_data = pd.read_csv(line_schedules_file)
        self.line_data['departure_time'] = pd.to_datetime(self.line_data['departure_time'])
        
        # Preprocess data for optimization
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data for optimization"""
        # Calculate hourly demand patterns
        self.hourly_demand = self.passenger_data.groupby(['line_id', 'hour'])['boarding'].sum().reset_index()
        
        # Calculate average occupancy rates
        self.avg_occupancy = self.passenger_data.groupby(['line_id', 'hour'])['occupancy_rate'].mean().reset_index()
        
        # Identify peak hours for each line
        self.peak_hours = {}
        for line_id in self.passenger_data['line_id'].unique():
            line_demand = self.hourly_demand[self.hourly_demand['line_id'] == line_id]
            peak_threshold = line_demand['boarding'].quantile(0.7)
            peak_hours = line_demand[line_demand['boarding'] >= peak_threshold]['hour'].tolist()
            self.peak_hours[line_id] = peak_hours
    
    def optimize_line_schedule(self, line_id: str, date: datetime) -> ScheduleChromosome:
        """Optimize schedule for a specific line using genetic algorithm"""
        print(f"Optimizing schedule for line {line_id} on {date.strftime('%Y-%m-%d')}")
        
        # Initialize population
        population = self._initialize_population(line_id, date)
        
        # Evolution loop
        for generation in range(self.config.generations):
            # Evaluate fitness
            self._evaluate_population(population, date)
            
            # Selection and reproduction
            new_population = []
            
            # Elite preservation (ensure all fitness values are not None)
            for chromo in population:
                if chromo.fitness is None:
                    chromo.fitness = 0.0
                    
            population.sort(key=lambda x: x.fitness or 0.0, reverse=True)
            new_population.extend([chr.copy() for chr in population[:self.config.elite_size]])
            
            # Generate new individuals
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2, date)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.config.mutation_rate:
                    self._mutate(offspring1, date)
                if random.random() < self.config.mutation_rate:
                    self._mutate(offspring2, date)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim to population size
            population = new_population[:self.config.population_size]
            
            # Progress report
            if generation % 20 == 0:
                try:
                    # Ensure all fitness values are not None for reporting
                    for chromo in population:
                        if chromo.fitness is None:
                            chromo.fitness = 0.0
                    
                    best = max(population, key=lambda x: x.fitness or 0.0)
                    print(f"Generation {generation}: Best fitness = {best.fitness:.4f}")
                except Exception as e:
                    print(f"Generation {generation}: Error in progress report - {e}")
                    print(f"Population fitness values: {[chromo.fitness for chromo in population[:5]]}")  # Show first 5
        
        # Final evaluation and return best solution
        self._evaluate_population(population, date)
        
        # Ensure all fitness values are not None before selection
        for chromo in population:
            if chromo.fitness is None:
                chromo.fitness = 0.0
                
        best_solution = max(population, key=lambda x: x.fitness or 0.0)
        self.best_solutions[line_id] = best_solution
        
        return best_solution
    
    def _initialize_population(self, line_id: str, date: datetime) -> List[ScheduleChromosome]:
        """Initialize population with diverse and realistic schedules"""
        population = []
        
        print(f"🧬 Initializing population for line {line_id}...")
        
        for i in range(self.config.population_size):
            departure_times = []
            
            # Create realistic schedule patterns
            if i < self.config.population_size // 3:
                # Rush hour focused schedules (1/3 of population)
                departure_times = self._create_rush_hour_schedule(date)
            elif i < 2 * self.config.population_size // 3:
                # Regular interval schedules (1/3 of population)
                departure_times = self._create_regular_interval_schedule(date)
            else:
                # Random/mixed schedules (1/3 of population)
                departure_times = self._create_mixed_schedule(date)
            
            chromosome = ScheduleChromosome(line_id, departure_times)
            population.append(chromosome)
        
        return population
    
    def _create_rush_hour_schedule(self, date: datetime) -> List[datetime]:
        """Create a schedule focused on rush hour service"""
        departures = []
        
        # Early morning: sparse service
        for hour in range(5, 7):
            departures.append(date.replace(hour=hour, minute=random.randint(0, 30), second=0))
        
        # Morning rush: frequent service
        for hour in range(7, 10):
            for minute in range(0, 60, random.randint(8, 15)):
                departures.append(date.replace(hour=hour, minute=minute, second=0))
        
        # Midday: moderate service
        for hour in range(10, 16):
            for minute in range(0, 60, random.randint(20, 30)):
                departures.append(date.replace(hour=hour, minute=minute, second=0))
        
        # Evening rush: frequent service
        for hour in range(16, 20):
            for minute in range(0, 60, random.randint(8, 15)):
                departures.append(date.replace(hour=hour, minute=minute, second=0))
        
        # Night: sparse service
        for hour in range(20, 24):
            if random.random() < 0.7:  # Not every hour
                departures.append(date.replace(hour=hour, minute=random.randint(0, 30), second=0))
        
        return sorted(list(set(departures)))  # Remove duplicates and sort
    
    def _create_regular_interval_schedule(self, date: datetime) -> List[datetime]:
        """Create a schedule with regular intervals"""
        departures = []
        interval = random.randint(15, 30)  # Regular 15-30 minute intervals
        
        current_time = date.replace(hour=5, minute=0, second=0)
        end_time = date.replace(hour=23, minute=59, second=0)
        
        while current_time <= end_time:
            departures.append(current_time)
            current_time += timedelta(minutes=interval)
            # Add small random variation
            variation = random.randint(-2, 2)
            current_time += timedelta(minutes=variation)
        
        return departures
    
    def _create_mixed_schedule(self, date: datetime) -> List[datetime]:
        """Create a mixed schedule with some randomness"""
        departures = []
        
        for hour in range(5, 24):
            # Random number of departures per hour (0-4)
            num_departures = random.randint(0, 4)
            
            for _ in range(num_departures):
                minute = random.randint(0, 59)
                departures.append(date.replace(hour=hour, minute=minute, second=0))
        
        return sorted(list(set(departures)))  # Remove duplicates and sort
    
    def _evaluate_population(self, population: List[ScheduleChromosome], date: datetime):
        """Evaluate fitness of all chromosomes in population"""
        import pandas as pd  # Ensure pandas is available for fitness calculation
        
        for chromosome in population:
            if chromosome.fitness is None:
                chromosome.fitness = self._calculate_fitness(chromosome, date)
    
    def _calculate_fitness(self, chromosome: ScheduleChromosome, date: datetime) -> float:
        """Calculate fitness score by running simulation with this schedule"""
        try:
            # Create fresh simulation components for this evaluation
            data = BusTransitData()
            network = TransitNetwork(data)
            
            if not network.load_network_data(self.stops_file):
                return 0.0
            
            # Create custom schedule generator that uses our chromosome
            class CustomScheduleGenerator(ScheduleGenerator):
                def __init__(self, data, network, test_chromosome):
                    super().__init__(data, network)
                    self.test_chromosome = test_chromosome
                
                def generate_line_schedules(self, date: str):
                    """Generate schedules using the chromosome's departure times"""
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    schedules = {}
                    
                    # Create schedule for the chromosome's line
                    line_id = self.test_chromosome.line_id
                    entries = []
                    
                    for departure_time in self.test_chromosome.departure_times:
                        # Ensure departure time is on the correct date
                        adjusted_departure = dt.replace(
                            hour=departure_time.hour,
                            minute=departure_time.minute,
                            second=departure_time.second
                        )
                        
                        entries.append(LineScheduleEntry(
                            line_id=line_id,
                            departure_time=adjusted_departure,
                            assigned_bus_id=None,
                            status="scheduled"
                        ))
                    
                    schedules[line_id] = entries
                    
                    # Add basic schedules for other lines (so simulation can run)
                    for other_line_id in data.lines.keys():
                        if other_line_id != line_id:
                            # Create basic schedule for other lines
                            other_entries = []
                            for hour in range(5, 24):
                                for minute in [0, 30]:  # Every 30 minutes
                                    dep_time = dt.replace(hour=hour, minute=minute, second=0)
                                    other_entries.append(LineScheduleEntry(
                                        line_id=other_line_id,
                                        departure_time=dep_time,
                                        assigned_bus_id=None,
                                        status="scheduled"
                                    ))
                            schedules[other_line_id] = other_entries
                    
                    # Store schedules in the data object correctly
                    self.data.line_schedules = schedules
                    return schedules
                
                def assign_buses_to_schedules(self, date: str) -> Dict[str, List[BusAssignment]]:
                    """Override to handle the schedules correctly"""
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    
                    # Ensure we have schedules
                    if not hasattr(self.data, 'line_schedules') or not self.data.line_schedules:
                        self.generate_line_schedules(date)
                    
                    # Get available buses
                    available_buses = list(self.data.buses.keys())
                    
                    # Assign buses to lines equally
                    lines = list(self.data.lines.keys())
                    buses_per_line = {line_id: [] for line_id in lines}
                    
                    for i, bus_id in enumerate(available_buses):
                        line_id = lines[i % len(lines)]
                        buses_per_line[line_id].append(bus_id)
                    
                    # Create assignments
                    all_assignments = {}
                    
                    # Process schedules (handle both dict and list formats)
                    if isinstance(self.data.line_schedules, dict):
                        schedule_items = self.data.line_schedules.items()
                    else:
                        # If it's a list, group by line_id
                        schedule_dict = {}
                        for entry in self.data.line_schedules:
                            if entry.line_id not in schedule_dict:
                                schedule_dict[entry.line_id] = []
                            schedule_dict[entry.line_id].append(entry)
                        schedule_items = schedule_dict.items()
                    
                    for line_id, schedule_entries in schedule_items:
                        line_buses = buses_per_line.get(line_id, [])
                        if not line_buses:
                            continue
                        
                        # Create simple assignments
                        for i, entry in enumerate(schedule_entries):
                            bus_id = line_buses[i % len(line_buses)]
                            
                            if bus_id not in all_assignments:
                                all_assignments[bus_id] = []
                            
                            # Simple assignment with 1 hour duration
                            end_time = entry.departure_time + timedelta(hours=1)
                            
                            all_assignments[bus_id].append(
                                BusAssignment(
                                    bus_id=bus_id,
                                    line_id=line_id,
                                    start_time=entry.departure_time,
                                    end_time=end_time,
                                    status="scheduled"
                                )
                            )
                            
                            entry.assigned_bus_id = bus_id
                    
                    self.data.bus_assignments = all_assignments
                    return all_assignments
            
            # Setup simulation with shorter duration for faster evaluation
            simulation_start = date.replace(hour=5, minute=0, second=0)
            simulation_end = simulation_start + timedelta(hours=self.config.simulation_duration_hours)
            
            sim_config = SimulationConfig(
                start_date=simulation_start,
                end_date=simulation_end,
                time_step=self.config.time_step,
                randomize_travel_times=True,
                randomize_passenger_demand=True,
                seed=self.config.random_seed
            )
            
            # Create simulation engine
            engine = SimulationEngine(data, sim_config)
            engine.network = network
            engine.schedule_generator = CustomScheduleGenerator(data, network, chromosome)
            
            # Setup and run simulation
            if not engine.setup_simulation(self.config.buses_per_line):
                return 0.0
            
            results_df = engine.run_simulation()
            
            if len(results_df) == 0:
                return 0.0
            
            # Calculate fitness metrics from simulation results
            return self._calculate_simulation_fitness(results_df, chromosome)
            
        except Exception as e:
            print(f"Error evaluating chromosome: {e}")
            return 0.0
    
    def _calculate_simulation_fitness(self, results_df: pd.DataFrame, chromosome: ScheduleChromosome) -> float:
        """Calculate fitness from simulation results"""
        
        # Filter results for this line only
        line_results = results_df[results_df['line_id'] == chromosome.line_id]
        
        if len(line_results) == 0:
            return 0.0
        
        # 1. Passenger wait metric (based on occupancy patterns)
        avg_occupancy = line_results['occupancy_rate'].mean()
        if pd.isna(avg_occupancy):
            avg_occupancy = 0.0
            
        target_occupancy = 0.7  # Target 70% occupancy
        wait_score = 1.0 - abs(avg_occupancy - target_occupancy) / 2.0
        wait_score = max(0.0, wait_score)
        
        # 2. Bus utilization score (prefer consistent moderate usage)
        occupancy_std = line_results['occupancy_rate'].std()
        if pd.isna(occupancy_std):
            occupancy_std = 0.0
            
        utilization_score = 1.0 / (1.0 + occupancy_std)  # Lower std deviation is better
        
        # 3. Overcrowding penalty
        overcrowded_count = len(line_results[line_results['occupancy_rate'] > 1.2])
        total_records = len(line_results)
        overcrowded_ratio = overcrowded_count / total_records if total_records > 0 else 0.0
        overcrowding_score = 1.0 - overcrowded_ratio
        
        # 4. Service coverage (number of unique stops served)
        unique_stops_served = len(line_results['stop_id'].unique())
        if self.base_data and chromosome.line_id in self.base_data.lines:
            total_stops_in_line = len(self.base_data.lines[chromosome.line_id].stops)
        else:
            total_stops_in_line = 10  # Default estimate
            
        coverage_score = unique_stops_served / total_stops_in_line if total_stops_in_line > 0 else 0.0
        
        # 5. Schedule efficiency (penalize too many or too few departures)
        num_departures = len(chromosome.departure_times)
        ideal_departures = self.config.simulation_duration_hours * 2  # About 2 per hour
        if ideal_departures > 0:
            efficiency_penalty = abs(num_departures - ideal_departures) / ideal_departures
        else:
            efficiency_penalty = 0.0
        efficiency_score = max(0.0, 1.0 - efficiency_penalty)
        
        # Combined fitness with safety checks
        fitness = (
            self.config.passenger_wait_weight * wait_score +
            self.config.bus_utilization_weight * utilization_score +
            self.config.overcrowding_weight * overcrowding_score +
            self.config.service_coverage_weight * coverage_score * efficiency_score
        )
        
        return max(0.0, min(1.0, fitness))  # Ensure fitness is between 0 and 1
    
    def _tournament_selection(self, population: List[ScheduleChromosome]) -> ScheduleChromosome:
        """Select individual using tournament selection"""
        tournament = random.sample(population, self.config.tournament_size)
        
        # Filter out chromosomes with None fitness and use default fitness if needed
        valid_chromosomes = []
        for chromo in tournament:
            if chromo.fitness is None:
                chromo.fitness = 0.0  # Assign default fitness
            valid_chromosomes.append(chromo)
        
        return max(valid_chromosomes, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: ScheduleChromosome, parent2: ScheduleChromosome, 
                   date: datetime) -> Tuple[ScheduleChromosome, ScheduleChromosome]:
        """Perform crossover between two parent schedules"""
        # Use time-based crossover
        crossover_hour = random.randint(
            self.config.operational_hours[0] + 2, 
            self.config.operational_hours[1] - 2
        )
        
        # Create offspring
        offspring1_times = []
        offspring2_times = []
        
        for dep in parent1.departure_times:
            if dep.hour < crossover_hour:
                offspring1_times.append(dep)
            else:
                offspring2_times.append(dep)
        
        for dep in parent2.departure_times:
            if dep.hour < crossover_hour:
                offspring2_times.append(dep)
            else:
                offspring1_times.append(dep)
        
        # Ensure minimum service frequency
        offspring1 = ScheduleChromosome(parent1.line_id, offspring1_times)
        offspring2 = ScheduleChromosome(parent2.line_id, offspring2_times)
        
        self._repair_schedule(offspring1, date)
        self._repair_schedule(offspring2, date)
        
        return offspring1, offspring2
    
    def _mutate(self, chromosome: ScheduleChromosome, date: datetime):
        """Apply mutation to a chromosome"""
        mutation_type = random.choice(['add', 'remove', 'shift'])
        
        if mutation_type == 'add' and len(chromosome.departure_times) < 200:
            # Add a new departure
            hour = random.randint(self.config.operational_hours[0], self.config.operational_hours[1] - 1)
            minute = random.randint(0, 59)
            new_departure = date.replace(hour=hour, minute=minute, second=0)
            chromosome.departure_times.append(new_departure)
            
        elif mutation_type == 'remove' and len(chromosome.departure_times) > 20:
            # Remove a random departure
            idx = random.randint(0, len(chromosome.departure_times) - 1)
            chromosome.departure_times.pop(idx)
            
        elif mutation_type == 'shift' and chromosome.departure_times:
            # Shift a departure time
            idx = random.randint(0, len(chromosome.departure_times) - 1)
            shift = random.randint(-10, 10)
            new_time = chromosome.departure_times[idx] + timedelta(minutes=shift)
            
            # Ensure within operational hours
            if self.config.operational_hours[0] <= new_time.hour < self.config.operational_hours[1]:
                chromosome.departure_times[idx] = new_time
        
        # Sort and repair
        chromosome.departure_times.sort()
        self._repair_schedule(chromosome, date)
        
        # Reset fitness for re-evaluation
        chromosome.fitness = None
    
    def _repair_schedule(self, chromosome: ScheduleChromosome, date: datetime):
        """Repair schedule to ensure constraints are met"""
        # Remove duplicates
        chromosome.departure_times = list(set(chromosome.departure_times))
        chromosome.departure_times.sort()
        
        # Ensure minimum intervals
        i = 0
        while i < len(chromosome.departure_times) - 1:
            current = chromosome.departure_times[i]
            next_dep = chromosome.departure_times[i + 1]
            interval = (next_dep - current).total_seconds() / 60
            
            if interval < self.config.min_interval:
                # Remove the later departure
                chromosome.departure_times.pop(i + 1)
            else:
                i += 1
        
        # Ensure sufficient service during peak hours (7-10 AM and 4-8 PM)
        peak_periods = [(7, 10), (16, 20)]  # Morning and evening rush
        
        for start_hour, end_hour in peak_periods:
            for hour in range(start_hour, end_hour):
                hour_departures = [d for d in chromosome.departure_times if d.hour == hour]
                if len(hour_departures) < 2:  # Minimum 2 departures per peak hour
                    # Add departures
                    for minute in [0, 30]:
                        new_dep = date.replace(hour=hour, minute=minute, second=0)
                        if new_dep not in chromosome.departure_times:
                            chromosome.departure_times.append(new_dep)
        
        self._repair_schedule_with_hourly_distribution(chromosome, date)
        chromosome.departure_times.sort()
    
    def optimize_all_lines(self, date: datetime, line_ids: Optional[List[str]] = None) -> Dict[str, ScheduleChromosome]:
        """Optimize schedules for all lines"""
        if line_ids is None:
            line_ids = self.passenger_data['line_id'].unique()
        
        optimized_schedules = {}
        
        # Use multiprocessing for parallel optimization
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            results = []
            for line_id in line_ids:
                result = pool.apply_async(self._optimize_line_wrapper, (line_id, date))
                results.append((line_id, result))
            
            for line_id, result in results:
                optimized_schedules[line_id] = result.get()
        
        return optimized_schedules
    
    def _optimize_line_wrapper(self, line_id: str, date: datetime) -> ScheduleChromosome:
        """Wrapper for multiprocessing"""
        return self.optimize_line_schedule(line_id, date)
    
    def export_optimized_schedules(self, schedules: Dict[str, ScheduleChromosome], output_file: str):
        """Export optimized schedules to CSV"""
        rows = []
        
        for line_id, chromosome in schedules.items():
            for departure_time in chromosome.departure_times:
                rows.append({
                    'line_id': line_id,
                    'departure_time': departure_time,
                    'fitness_score': chromosome.fitness,
                    'status': 'optimized'
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(rows)} optimized schedule entries to {output_file}")
    
    def compare_schedules(self, original_file: str, optimized_schedules: Dict[str, ScheduleChromosome]):
        """Compare original and optimized schedules"""
        # Load original schedules
        original = pd.read_csv(original_file)
        original['departure_time'] = pd.to_datetime(original['departure_time'])
        
        comparison = []
        
        for line_id in optimized_schedules:
            # Original schedule metrics
            orig_deps = original[original['line_id'] == line_id]['departure_time']
            orig_count = len(orig_deps)
            
            # Optimized schedule metrics
            opt_deps = optimized_schedules[line_id].departure_times
            opt_count = len(opt_deps)
            
            # Calculate average intervals
            if orig_count > 1:
                orig_intervals = [(orig_deps.iloc[i+1] - orig_deps.iloc[i]).total_seconds() / 60 
                                 for i in range(len(orig_deps)-1)]
                avg_orig_interval = np.mean(orig_intervals)
            else:
                avg_orig_interval = 0
            
            if opt_count > 1:
                opt_intervals = [(opt_deps[i+1] - opt_deps[i]).total_seconds() / 60 
                                for i in range(len(opt_deps)-1)]
                avg_opt_interval = np.mean(opt_intervals)
            else:
                avg_opt_interval = 0
            
            comparison.append({
                'line_id': line_id,
                'original_departures': orig_count,
                'optimized_departures': opt_count,
                'original_avg_interval': avg_orig_interval,
                'optimized_avg_interval': avg_opt_interval,
                'fitness_score': optimized_schedules[line_id].fitness,
                'improvement': 'Improved' if optimized_schedules[line_id].fitness > 0.7 else 'Needs Review'
            })
        
        comparison_df = pd.DataFrame(comparison)
        return comparison_df

    def _repair_schedule_with_hourly_distribution(self, chromosome, date):
        """
        Enhanced repair that enforces realistic hourly departure distribution
        """
        print(f"    Applying hourly distribution constraints...")
        
        # Target departures per hour (based on realistic transit patterns)
        target_hourly_departures = {
            5: 1, 6: 2,           # Early morning: 3 departures
            7: 4, 8: 5, 9: 4,     # Morning rush: 13 departures  
            10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,  # Midday: 21 departures
            17: 5, 18: 5, 19: 4,  # Evening rush: 14 departures
            20: 2, 21: 2, 22: 1, 23: 1   # Evening/night: 6 departures
            # Total: 57 departures (can be scaled based on target_departure_count)
        }
        
        # Scale target distribution to match chromosome's current departure count
        total_target = sum(target_hourly_departures.values())
        target_count = len(chromosome.departure_times)  # Use current departure count
        scale_factor = target_count / total_target
        
        scaled_targets = {}
        for hour, count in target_hourly_departures.items():
            scaled_targets[hour] = max(1, round(count * scale_factor))
        
        # Ensure total matches target (adjust rounding errors)
        current_total = sum(scaled_targets.values())
        if current_total != target_count:
            # Adjust largest hours first
            diff = target_count - current_total
            sorted_hours = sorted(scaled_targets.keys(), key=lambda h: scaled_targets[h], reverse=True)
            
            for hour in sorted_hours:
                if diff == 0:
                    break
                if diff > 0:
                    scaled_targets[hour] += 1
                    diff -= 1
                elif scaled_targets[hour] > 1:
                    scaled_targets[hour] -= 1
                    diff += 1
        
        print(f"    Target hourly distribution: {scaled_targets}")
        
        # Group current departures by hour
        current_hourly = {}
        for dep_time in chromosome.departure_times:
            hour = dep_time.hour
            current_hourly[hour] = current_hourly.get(hour, 0) + 1
        
        # Build new departure list with proper hourly distribution
        new_departures = []
        
        for hour in range(5, 24):  # Operating hours
            target_for_hour = scaled_targets.get(hour, 0)
            current_for_hour = current_hourly.get(hour, 0)
            
            if target_for_hour == 0:
                continue  # Skip this hour
            
            if current_for_hour >= target_for_hour:
                # Take first N departures for this hour
                hour_deps = [d for d in chromosome.departure_times if d.hour == hour]
                hour_deps.sort()
                new_departures.extend(hour_deps[:target_for_hour])
            else:
                # Keep existing and add new ones
                hour_deps = [d for d in chromosome.departure_times if d.hour == hour]
                new_departures.extend(hour_deps)
                
                # Add missing departures
                needed = target_for_hour - len(hour_deps)
                for i in range(needed):
                    # Distribute evenly across the hour
                    minute = int((i + 0.5) * 60 / target_for_hour)
                    minute = min(59, max(0, minute))
                    new_dep = date.replace(hour=hour, minute=minute, second=0)
                    new_departures.append(new_dep)
        
        # Apply minimum interval constraints within each hour
        final_departures = []
        for hour in range(5, 24):
            hour_deps = [d for d in new_departures if d.hour == hour]
            hour_deps.sort()
            
            # Ensure 5-minute minimum intervals
            filtered = []
            for dep in hour_deps:
                if not filtered or (dep - filtered[-1]).total_seconds() >= 300:
                    filtered.append(dep)
            
            final_departures.extend(filtered)
        
        # Update chromosome
        chromosome.departure_times = sorted(final_departures)
        
        print(f"    Final departures: {len(chromosome.departure_times)} (target: {target_count})")
        
        # If we're short, add a few more in flexible hours
        if len(chromosome.departure_times) < target_count:
            shortage = target_count - len(chromosome.departure_times)
            flexible_hours = [10, 11, 14, 15, 20, 21]  # Hours where we can add more
            
            for _ in range(shortage):
                hour = np.random.choice(flexible_hours)
                minute = np.random.randint(0, 60)
                new_dep = date.replace(hour=hour, minute=minute, second=0)
                chromosome.departure_times.append(new_dep)
            
            chromosome.departure_times = sorted(chromosome.departure_times)
        
        return chromosome
    
# Utility function for running optimization
def run_optimization_with_constraints(stops_file: str = "ankara_bus_stops_10.csv", 
                                     simulation_results_file: str = None,
                                     optimization_date: str = "2025-06-02", 
                                     output_file: str = "optimized_schedule.csv",
                                     target_lines: Optional[List[str]] = None):
    """Run optimization with historical constraints"""
    
    print("🚌 Bus Schedule Optimization with Historical Constraints")
    print("=" * 60)
    
    config = OptimizationConfig(
        population_size=25,
        generations=35,
        passenger_wait_weight=0.3,
        bus_utilization_weight=0.3,
        overcrowding_weight=0.2,
        service_coverage_weight=0.2
    )
    
    optimizer = GeneticScheduleOptimizer(config, stops_file)
    
    if not optimizer.base_data:
        print("❌ Failed to setup simulation data")
        return None, None
    
    # CRITICAL: Load historical constraints from Stage 1
    if simulation_results_file:
        optimizer.load_historical_constraints_from_simulation(simulation_results_file)
    
    # Parse optimization date
    opt_date = datetime.strptime(optimization_date, '%Y-%m-%d')
    
    # Select lines to optimize
    if target_lines is None:
        target_lines = optimizer.line_ids
        
    print(f"🎯 Optimizing schedules for {len(target_lines)} lines: {target_lines}")
    print(f"📅 Optimization date: {optimization_date}")
    
    # Optimize selected lines
    optimized_schedules = {}
    
    for line_id in target_lines:
        print(f"\n🧬 Starting optimization for line {line_id}...")
        best_schedule = optimizer.optimize_line_schedule(line_id, opt_date)
        optimized_schedules[line_id] = best_schedule
        
        print(f"✅ Line {line_id} optimization complete!")
        print(f"   Best fitness: {best_schedule.fitness:.4f}")
        print(f"   Departures: {len(best_schedule.departure_times)}")
    
    # Export results
    print(f"\n📄 Exporting optimized schedules to {output_file}...")
    optimizer.export_optimized_schedules(optimized_schedules, output_file)
    
    # Print summary
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    for line_id, schedule in optimized_schedules.items():
        departure_times = [dt.strftime("%H:%M") for dt in schedule.departure_times[:10]]
        print(f"Line {line_id}:")
        print(f"  Fitness: {schedule.fitness:.4f}")
        print(f"  Total departures: {len(schedule.departure_times)}")
        print(f"  Sample times: {', '.join(departure_times)}{'...' if len(schedule.departure_times) > 10 else ''}")
    
    return optimizer, optimized_schedules


def quick_test_optimization():
    """Quick test function for schedule optimization"""
    print("🔬 Running quick schedule optimization test...")
    
    # Test with just one line
    config = OptimizationConfig(
        population_size=10,
        generations=5,
        simulation_duration_hours=3  # Very short simulation for quick test
    )
    
    optimizer = GeneticScheduleOptimizer(config, "ankara_bus_stops_10.csv")
    
    if optimizer.base_data:
        test_line = optimizer.line_ids[0]
        test_date = datetime(2025, 6, 2)
        
        print(f"Testing optimization for line {test_line}...")
        result = optimizer.optimize_line_schedule(test_line, test_date)
        
        print(f"✅ Test complete!")
        print(f"Best fitness: {result.fitness:.4f}")
        print(f"Departures: {len(result.departure_times)}")
        
        return result
    
    return None


if __name__ == "__main__":
    """Main execution block for direct script running"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run quick test
        print("🧪 Running quick test...")
        quick_test_optimization()
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        # Run full optimization
        print("🚀 Running full optimization...")
        optimizer, schedules = run_optimization_with_constraints()
        print(f"✅ Optimization completed for {len(schedules)} lines")
    else:
        print("📚 GA Schedule Optimizer")
        print("Available functions:")
        print("  - run_optimization_with_constraints(): Main optimization function")
        print("  - quick_test_optimization(): Quick test function")
        print("  - GeneticScheduleOptimizer: Main optimizer class")
        print("\nUsage:")
        print("  python ga_optimize.py test    # Run quick test")
        print("  python ga_optimize.py full    # Run full optimization")
        print("\nOr import in your own script:")
        print("  from ga_optimize import run_optimization_with_constraints")