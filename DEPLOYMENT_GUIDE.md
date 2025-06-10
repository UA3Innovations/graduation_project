# Azure Deployment Guide

## 🎯 Complete Bus Transit Simulation Pipeline for Microsoft Azure

This guide walks you through deploying the bus transit simulation pipeline to Microsoft Azure using Container Instances.

## 📋 Prerequisites Checklist

### Required Tools
- [ ] **Python 3.11+** installed locally
- [ ] **Azure CLI** (`az` command)
- [ ] **Docker** (for container builds)
- [ ] **Git** (for version control)

### Azure Requirements
- [ ] **Azure Subscription** with Owner or Contributor permissions
- [ ] **Azure Resource Group** creation permissions
- [ ] **Azure Container Instance** deployment permissions
- [ ] **Azure Storage Account** creation permissions (optional)

## 🚀 Step-by-Step Deployment

### Step 1: Initial Setup

```bash
# Clone/navigate to the pipeline directory
cd bus_simulation_pipeline

# Install Python dependencies
pip install -r requirements.txt

# Run initial setup and validation
python scripts/setup_and_test.py
```

### Step 2: Azure Authentication

```bash
# Login to Azure
az login

# List available subscriptions
az account list --output table

# Set your subscription (replace with your subscription ID)
az account set --subscription "your-subscription-id"

# Export subscription ID for the pipeline
export AZURE_SUBSCRIPTION_ID=$(az account show --query id --output tsv)

# Verify authentication
az account show
```

### Step 3: Azure Resource Setup

```bash
# Create resource group
az group create \
  --name bus-simulation-rg \
  --location "East US"

# Create storage account (optional, for result persistence)
az storage account create \
  --name bussimulationstorage \
  --resource-group bus-simulation-rg \
  --location "East US" \
  --sku Standard_LRS

# Get storage connection string (optional)
export AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
  --name bussimulationstorage \
  --resource-group bus-simulation-rg \
  --query connectionString --output tsv)
```

### Step 4: Container Registry Setup

```bash
# Create Azure Container Registry
az acr create \
  --resource-group bus-simulation-rg \
  --name bussimulationacr \
  --sku Basic

# Login to container registry
az acr login --name bussimulationacr

# Build and push container image
docker build -f docker/Dockerfile -t bus-simulation:latest .
docker tag bus-simulation:latest bussimulationacr.azurecr.io/bus-simulation:latest
docker push bussimulationacr.azurecr.io/bus-simulation:latest
```

### Step 5: Update Configuration

Edit `config/azure_config.yaml`:

```yaml
azure:
  resource_group: "bus-simulation-rg"
  location: "East US"
  
  container:
    name: "bus-simulation-container"
    image: "bussimulationacr.azurecr.io/bus-simulation:latest"  # Update this line
    cpu_cores: 2.0
    memory_gb: 4.0
```

### Step 6: Deploy and Run

```bash
# Test deployment (validation only)
python scripts/deploy_to_azure.py --test-only

# Deploy to development environment
python scripts/deploy_to_azure.py --environment development

# Deploy to production with custom parameters
python scripts/deploy_to_azure.py \
  --environment production \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --buses-per-line 8
```

## 🔧 Configuration Options

### Environment Configurations

| Environment | CPU | Memory | Use Case |
|-------------|-----|---------|----------|
| `testing` | 0.5 cores | 1GB | Quick validation |
| `development` | 1 core | 2GB | Development testing |
| `production` | 4 cores | 8GB | Full simulations |

### Simulation Parameters

```bash
# Quick test (1 day)
python scripts/deploy_to_azure.py \
  --environment testing \
  --start-date 2024-01-01 \
  --end-date 2024-01-01 \
  --buses-per-line 3

# Weekly simulation
python scripts/deploy_to_azure.py \
  --environment development \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --time-step 5

# Monthly production run
python scripts/deploy_to_azure.py \
  --environment production \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --buses-per-line 10 \
  --time-step 2
```

## 📊 Monitoring Deployments

### Check Container Status

```bash
# List all container groups
az container list \
  --resource-group bus-simulation-rg \
  --output table

# Get specific container details
az container show \
  --resource-group bus-simulation-rg \
  --name bus-simulation-container-development-1234567890

# View container logs
az container logs \
  --resource-group bus-simulation-rg \
  --name bus-simulation-container-development-1234567890
```

### Monitor Resource Usage

```bash
# Check resource group resources
az resource list \
  --resource-group bus-simulation-rg \
  --output table

# Monitor costs
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31
```

## 🧹 Cleanup and Cost Management

### Clean Up Single Deployment

```bash
# Delete specific container group
az container delete \
  --resource-group bus-simulation-rg \
  --name bus-simulation-container-development-1234567890 \
  --yes
```

### Complete Cleanup

```bash
# Delete entire resource group (removes all resources)
az group delete \
  --name bus-simulation-rg \
  --yes \
  --no-wait
```

### Scheduled Cleanup Script

Create `cleanup_old_containers.sh`:

```bash
#!/bin/bash
# Delete containers older than 24 hours
az container list \
  --resource-group bus-simulation-rg \
  --query "[?properties.instanceView.state=='Succeeded'].name" \
  --output tsv | \
while read container_name; do
  echo "Deleting completed container: $container_name"
  az container delete \
    --resource-group bus-simulation-rg \
    --name "$container_name" \
    --yes
done
```

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **"Container deployment failed"**
   ```bash
   # Check ACR permissions
   az acr check-health --name bussimulationacr
   
   # Verify image exists
   az acr repository list --name bussimulationacr
   ```

2. **"Simulation timeout"**
   ```bash
   # Increase timeout in deployment
   python scripts/deploy_to_azure.py \
     --environment development \
     --timeout-minutes 180
   ```

3. **"Insufficient resources"**
   ```bash
   # Check Azure quotas
   az vm list-usage --location "East US" --output table
   
   # Use smaller environment
   python scripts/deploy_to_azure.py --environment testing
   ```

### Debug Commands

```bash
# Validate local configuration
python scripts/deploy_to_azure.py --test-only

# Run simulation locally
python scripts/deploy_to_azure.py --local-run

# Check Azure authentication
az account show

# Test container locally
docker run --rm bus-simulation:latest
```

## 💰 Cost Optimization

### Resource Sizing Guidelines

| Simulation Duration | Recommended Environment | Estimated Cost/Hour |
|-------------------|------------------------|-------------------|
| 1 day | testing (0.5 CPU, 1GB) | ~$0.05 |
| 1 week | development (1 CPU, 2GB) | ~$0.10 |
| 1 month | production (4 CPU, 8GB) | ~$0.40 |

### Cost Reduction Tips

1. **Use appropriate sizing**: Don't over-provision resources
2. **Enable auto-cleanup**: Delete containers after completion
3. **Use spot instances**: For non-critical workloads
4. **Monitor usage**: Set up billing alerts

## 📈 Scaling for Large Simulations

### Batch Processing

For very large simulations, consider Azure Batch:

```bash
# Create batch account
az batch account create \
  --name bussimulationbatch \
  --resource-group bus-simulation-rg \
  --location "East US"
```

### Parallel Execution

Split large simulations into smaller chunks:

```bash
# Run multiple months in parallel
for month in {01..12}; do
  python scripts/deploy_to_azure.py \
    --environment development \
    --start-date 2024-${month}-01 \
    --end-date 2024-${month}-28 &
done
wait
```

## 🔐 Security Best Practices

### Authentication
- Use Azure AD for authentication
- Store secrets in Azure Key Vault
- Use managed identities when possible

### Network Security
- Deploy in private virtual networks for sensitive data
- Use Azure Private Endpoints for storage access

### Access Control
- Implement least-privilege access
- Use Azure RBAC for fine-grained permissions

---

## 📞 Support and Next Steps

### Getting Help
- Check logs with `az container logs`
- Review Azure documentation
- Open GitHub issues for pipeline bugs

### Production Recommendations
1. Set up monitoring with Azure Monitor
2. Implement automated deployment pipelines
3. Configure backup strategies for results
4. Set up alerting for failed simulations

### Advanced Features
- **Auto-scaling**: Implement based on queue depth
- **Result processing**: Automated analysis pipelines
- **Visualization**: Power BI integration for results
- **API integration**: RESTful endpoints for simulation triggers 