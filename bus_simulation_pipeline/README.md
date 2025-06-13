# Bus Transit Simulation Pipeline

A comprehensive bus transit simulation system designed for local deployment and analysis.

## 🚌 Features

- **Realistic Transit Simulation**: Models bus routes, stops, passenger flow, and real-time operations
- **Configurable Parameters**: Easily adjust simulation settings via YAML configuration files
- **Performance Analytics**: Generate detailed reports on bus utilization, passenger wait times, and system efficiency
- **Extensible Architecture**: Modular design allows for easy customization and extension

## 📁 Project Structure

```
bus_simulation_pipeline/
├── src/                        # Source code
│   ├── core/                   # Core simulation components
│   ├── components/             # Individual system components
│   └── utils/                  # Utility functions
├── config/                     # Configuration files
│   └── simulation_config.yaml  # Main simulation settings
├── data/                       # Input data
│   └── ankara_bus_stops.csv
├── pipeline/                   # Pipeline orchestration
│   └── pipeline_runner.py     # Main pipeline runner
├── scripts/                    # Utility scripts
│   ├── run_simulation.py      # Local simulation runner
│   └── setup_and_test.py      # Setup and testing script
├── tests/                      # Test files
└── output/                     # Local output directory
```

## 🛠️ Requirements

1. **Python 3.8+**
2. **Required packages** (see `requirements.txt`)

## 🚀 Quick Start

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run a quick test
python scripts/setup_and_test.py

# Run local simulation
python scripts/run_simulation.py

# Test with custom parameters
python scripts/run_simulation.py --local-run
```

## ⚙️ Configuration

Edit `config/simulation_config.yaml`:

```yaml
simulation:
  start_date: "2025-06-02"
  end_date: "2025-06-09"
  time_step: 5

data:
  stops_file: "data/ankara_bus_stops.csv"

output:
  directory: "output"
  export_format: "csv"
```

## 📊 Output

The simulation generates:
- **Passenger flow data**: Boarding/alighting patterns
- **Bus utilization metrics**: Occupancy rates and efficiency
- **Performance analytics**: Wait times and service quality
- **System reports**: Overall network performance

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_pipeline.py

# Setup and test everything
python scripts/setup_and_test.py
```

## 📈 Performance

- **Local simulation**: Handles 1-week simulations in minutes
- **Memory efficient**: Optimized for large datasets
- **Configurable precision**: Balance speed vs. accuracy

## 🐳 Docker Usage

### Build and Run with Docker

```bash
# Build the main simulation container
docker build -f docker/Dockerfile -t bus-simulation:latest .

# Run simulation with Docker
docker run --rm -v $(pwd)/output:/app/output bus-simulation:latest

# Or use docker-compose for easier management
docker-compose up bus-simulation

# Run tests with Docker
docker-compose --profile test up bus-simulation-test

# View results with web server (optional)
docker-compose --profile viewer up results-viewer
```

### Docker Development

```bash
# Build and run for development
docker-compose up --build

# Run specific services
docker-compose up bus-simulation
docker-compose --profile test up bus-simulation-test

# Clean up containers
docker-compose down
```

## 🔧 Development

```bash
# Install in development mode
pip install -e .

# Run with development dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/

# Run linting
flake8 src/ tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on the GitHub repository. 