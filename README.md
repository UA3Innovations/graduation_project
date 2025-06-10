# Bus Transit Simulation Pipeline

A comprehensive bus transit simulation system designed for cloud deployment on Microsoft Azure.

## 🚌 Overview

This pipeline simulates bus transit operations with realistic passenger flows, schedule adherence, and system performance metrics. It's designed to run efficiently on Azure Container Instances or Azure Batch for scalable simulation workloads.

## 📁 Project Structure

```
bus_simulation_pipeline/
├── config/                     # Configuration files
│   ├── simulation_config.yaml  # Simulation parameters
│   └── azure_config.yaml      # Azure deployment settings
├── src/                        # Source code
│   ├── core/                  # Core simulation components
│   │   ├── simulation_engine.py
│   │   └── data_models.py
│   └── components/            # Specialized modules
│       ├── transit_network.py
│       ├── passenger_generator.py
│       ├── bus_management.py
│       └── schedule_generator.py
├── pipeline/                   # Pipeline orchestration
│   ├── pipeline_runner.py     # Main pipeline coordinator
│   └── azure_deployer.py      # Azure deployment manager
├── docker/                     # Docker configuration
│   └── Dockerfile             # Container definition
├── scripts/                    # Deployment and utility scripts
│   ├── deploy_to_azure.py     # Main deployment script
│   └── run_simulation.py      # Local simulation runner
├── data/                       # Input data
│   └── ankara_bus_stops_10.csv
├── tests/                      # Test files
└── output/                     # Local output directory
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Azure CLI** (for deployment)
3. **Docker** (for containerization)
4. **Azure Subscription** with appropriate permissions

### Local Setup

1. **Clone and setup environment:**
```bash
cd bus_simulation_pipeline
pip install -r requirements.txt
```

2. **Run simulation locally:**
```bash
python scripts/deploy_to_azure.py --local-run
```

3. **Test configuration:**
```bash
python scripts/deploy_to_azure.py --test-only
```

### Azure Deployment

1. **Setup Azure credentials:**
```bash
# Login to Azure
az login

# Set subscription (replace with your subscription ID)
export AZURE_SUBSCRIPTION_ID="your-subscription-id"

# Optional: Setup storage connection string
export AZURE_STORAGE_CONNECTION_STRING="your-storage-connection-string"
```

2. **Deploy to development environment:**
```bash
python scripts/deploy_to_azure.py --environment development
```

3. **Deploy to production:**
```bash
python scripts/deploy_to_azure.py --environment production \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --buses-per-line 10
```

## ⚙️ Configuration

### Simulation Configuration (`config/simulation_config.yaml`)

```yaml
simulation:
  start_date: "2024-01-01"
  end_date: "2024-01-07"
  time_step: 5              # minutes
  buses_per_line: 6
  randomize_travel_times: true
  randomize_passenger_demand: true
  weather_effects_probability: 0.15
  seed: 42

data:
  stops_file: "data/ankara_bus_stops_10.csv"

output:
  directory: "output"
  summary: true
  debug: false
```

### Azure Configuration (`config/azure_config.yaml`)

```yaml
azure:
  resource_group: "bus-simulation-rg"
  location: "East US"
  
  container:
    name: "bus-simulation-container"
    image: "bus-simulation:latest"
    cpu_cores: 2.0
    memory_gb: 4.0
    restart_policy: "Never"

environment:
  development:
    cpu_cores: 1.0
    memory_gb: 2.0
    timeout_hours: 4
  
  production:
    cpu_cores: 4.0
    memory_gb: 8.0
    timeout_hours: 48
```

## 🐳 Docker Deployment

### Build Container

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t bus-simulation:latest .

# Test locally
docker run --rm -v $(pwd)/output:/app/output bus-simulation:latest
```

### Push to Azure Container Registry

```bash
# Create Azure Container Registry
az acr create --resource-group bus-simulation-rg \
  --name bussimulationacr --sku Basic

# Login to ACR
az acr login --name bussimulationacr

# Tag and push image
docker tag bus-simulation:latest bussimulationacr.azurecr.io/bus-simulation:latest
docker push bussimulationacr.azurecr.io/bus-simulation:latest
```

## 📊 Output Files

The simulation generates the following output files:

| File | Description |
|------|-------------|
| `passenger_flow_results.csv` | Detailed passenger boarding/alighting data |
| `bus_positions_results.csv` | Bus location and load tracking |
| `buses.csv` | Bus fleet configuration |
| `line_schedules.csv` | Generated line schedules |
| `bus_assignments.csv` | Bus-to-line assignments |
| `summary_statistics.json` | Aggregated performance metrics |

## 🔧 Advanced Usage

### Custom Simulation Parameters

```bash
# Extended simulation with more buses
python scripts/deploy_to_azure.py \
  --environment production \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --buses-per-line 15 \
  --time-step 2

# Quick development test
python scripts/deploy_to_azure.py \
  --environment testing \
  --start-date 2024-01-01 \
  --end-date 2024-01-02 \
  --buses-per-line 3
```

### Pipeline Monitoring

```bash
# Monitor running container
az container show --resource-group bus-simulation-rg \
  --name bus-simulation-container-development-1234567890

# View logs
az container logs --resource-group bus-simulation-rg \
  --name bus-simulation-container-development-1234567890
```

### Resource Management

```bash
# List all containers
az container list --resource-group bus-simulation-rg

# Clean up resources
az group delete --name bus-simulation-rg --yes --no-wait
```

## 🎛️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID | Yes (for Azure deployment) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage connection string | Optional |
| `AZURE_STORAGE_ACCOUNT` | Storage account name | Optional |
| `AZURE_STORAGE_CONTAINER` | Storage container name | Optional |

## 📈 Performance Tuning

### Azure Container Instances

- **Development**: 1 CPU, 2GB RAM (cost-optimized)
- **Production**: 4 CPU, 8GB RAM (performance-optimized)
- **Testing**: 0.5 CPU, 1GB RAM (minimal resources)

### Simulation Optimization

- Reduce `time_step` for higher precision (increases runtime)
- Increase `buses_per_line` for realistic scenarios
- Enable `debug: true` for detailed logging (increases I/O)

## 🚨 Troubleshooting

### Common Issues

1. **"Container deployment failed"**
   - Check Azure subscription permissions
   - Verify resource group exists
   - Ensure container image is accessible

2. **"Simulation timeout"**
   - Increase `--timeout-minutes` parameter
   - Check Azure Container Instance logs
   - Reduce simulation period or complexity

3. **"Missing output files"**
   - Verify storage configuration
   - Check container execution logs
   - Ensure simulation completed successfully

### Debug Commands

```bash
# Validate configuration
python scripts/deploy_to_azure.py --test-only

# Run locally for debugging
python scripts/deploy_to_azure.py --local-run --config config/simulation_config.yaml

# Check Azure resources
az resource list --resource-group bus-simulation-rg
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test changes locally
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🔗 Related Links

- [Azure Container Instances Documentation](https://docs.microsoft.com/en-us/azure/container-instances/)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)
- [Docker Documentation](https://docs.docker.com/)

For support, please open an issue or contact the development team. 