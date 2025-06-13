# Bus Transit Optimization Pipeline

A genetic algorithm-based optimization system for bus transit schedules with Azure cloud deployment capabilities.

## Overview

This package provides advanced optimization capabilities for bus transit systems using genetic algorithms. It works in conjunction with the `bus_simulation_pipeline` package to optimize bus schedules based on passenger demand patterns and operational constraints.

## Features

- **Genetic Algorithm Optimization**: Advanced evolutionary algorithms for schedule optimization
- **Multi-Objective Optimization**: Balances passenger wait times, bus utilization, overcrowding, and service coverage
- **Simulation-Based Evaluation**: Uses realistic simulation to evaluate schedule fitness
- **Cloud Deployment**: Azure Container Instances support for scalable optimization
- **Comprehensive Analysis**: Detailed evaluation and visualization of optimization results

## Dependencies

This package depends on the `bus_simulation_pipeline` package for core simulation functionality. Make sure to install and set up the simulation package first.

## Installation

### Local Installation

1. **Install the simulation package first**:
   ```bash
   cd ../bus_simulation_pipeline
   pip install -e .
   ```

2. **Install the optimization package**:
   ```bash
   cd ../bus_optimization_pipeline
   pip install -r requirements.txt
   pip install -e .
   ```

### Docker Installation

```bash
docker build -t bus-optimization:latest .
```

## Quick Start

### 1. Basic Optimization

```bash
# Run optimization with default configuration
python pipeline/optimization_runner.py

# Run with custom configuration
python pipeline/optimization_runner.py --config config/my_config.yaml

# Quick test with reduced parameters
python pipeline/optimization_runner.py --quick-test
```

### 2. Configuration

Edit `config/optimization_config.yaml` to customize:

```yaml
optimization:
  population_size: 30
  generations: 50
  target_date: "2025-06-02"
  target_lines: ["101", "102-1"]  # Optimize specific lines

data:
  stops_file: "data/ankara_bus_stops.csv"

output:
  directory: "optimization_output"
  run_evaluation: true
```

### 3. Programmatic Usage

```python
from src.optimization.ga_optimize import GeneticScheduleOptimizer, OptimizationConfig
from datetime import datetime

# Create configuration
config = OptimizationConfig(
    population_size=20,
    generations=30,
    simulation_duration_hours=4
)

# Initialize optimizer
optimizer = GeneticScheduleOptimizer(config, "data/ankara_bus_stops.csv")

# Optimize a specific line
target_date = datetime(2025, 6, 2)
result = optimizer.optimize_line_schedule("101", target_date)

print(f"Best fitness: {result.fitness:.4f}")
print(f"Departures: {len(result.departure_times)}")
```

## Optimization Algorithm

### Genetic Algorithm Components

1. **Chromosome Representation**: Each schedule is represented as a list of departure times
2. **Fitness Function**: Multi-objective evaluation based on simulation results
3. **Selection**: Tournament selection for parent selection
4. **Crossover**: Time-based crossover preserving schedule structure
5. **Mutation**: Random time adjustments and departure additions/removals

### Fitness Evaluation

The fitness function combines multiple objectives:

- **Passenger Wait Time** (40%): Minimize average passenger waiting
- **Bus Utilization** (30%): Maximize efficient use of buses
- **Overcrowding** (20%): Minimize passenger overcrowding
- **Service Coverage** (10%): Maintain adequate service frequency

### Constraints

- Minimum 5-minute intervals between departures
- Maximum 60-minute intervals between departures
- Operating hours: 5:00 AM - 12:00 AM
- Resource constraints: Limited number of buses per line

## Output Files

The optimization generates several output files:

- `optimized_schedule_YYYY-MM-DD.csv`: Optimized departure schedules
- `optimization_log.txt`: Detailed optimization progress
- `fitness_evolution.png`: Fitness evolution over generations
- `schedule_comparison.csv`: Comparison with baseline schedules

## Azure Deployment

### Prerequisites

1. Azure subscription with Container Instances enabled
2. Azure Storage Account for results
3. Azure Container Registry (optional)

### Environment Variables

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_STORAGE_ACCOUNT="your-storage-account"
export AZURE_LOG_ANALYTICS_WORKSPACE="your-workspace-id"
```

### Deploy to Azure

```bash
# Deploy with default configuration
python scripts/deploy_to_azure.py

# Deploy with custom configuration
python scripts/deploy_to_azure.py --config config/azure_config.yaml

# Monitor deployment
python scripts/deploy_to_azure.py --monitor
```

## Configuration Reference

### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 30 | Number of schedules in each generation |
| `generations` | 50 | Number of evolutionary generations |
| `crossover_rate` | 0.8 | Probability of crossover between parents |
| `mutation_rate` | 0.2 | Probability of mutation |
| `elite_size` | 5 | Number of best schedules preserved |

### Objective Weights

| Objective | Default Weight | Description |
|-----------|----------------|-------------|
| `passenger_wait_weight` | 0.4 | Minimize passenger waiting times |
| `bus_utilization_weight` | 0.3 | Maximize bus utilization efficiency |
| `overcrowding_weight` | 0.2 | Minimize passenger overcrowding |
| `service_coverage_weight` | 0.1 | Maintain service coverage |

## Performance Tuning

### For Faster Optimization

```yaml
optimization:
  population_size: 15        # Reduce population
  generations: 25            # Fewer generations
  simulation_duration_hours: 3  # Shorter simulation
  time_step: 15             # Larger time steps
```

### For Better Quality

```yaml
optimization:
  population_size: 50        # Larger population
  generations: 100           # More generations
  simulation_duration_hours: 8  # Longer simulation
  time_step: 5              # Smaller time steps
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `bus_simulation_pipeline` is installed and in Python path
2. **Memory Issues**: Reduce population size or simulation duration
3. **Slow Performance**: Use quick test mode or reduce parameters
4. **Azure Deployment**: Check environment variables and permissions

### Debug Mode

```bash
# Enable debug logging
export OPTIMIZATION_DEBUG=1
python pipeline/optimization_runner.py --config config/debug_config.yaml
```

## API Reference

### Main Classes

- `GeneticScheduleOptimizer`: Main optimization engine
- `OptimizationConfig`: Configuration container
- `ScheduleChromosome`: Individual schedule representation
- `OptimizationEvaluator`: Results analysis and evaluation

### Key Functions

- `run_optimization()`: High-level optimization function
- `optimize_line_schedule()`: Optimize a single line
- `optimize_all_lines()`: Optimize all lines in parallel

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration reference

## Related Projects

- [Bus Simulation Pipeline](../bus_simulation_pipeline/): Core simulation engine
- [Bus Analytics Dashboard](../bus_analytics_dashboard/): Visualization and analysis tools 