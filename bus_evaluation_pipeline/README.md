# Bus Evaluation Pipeline

A comprehensive evaluation system for bus schedule optimization results, providing detailed analysis, visualization, and reporting capabilities.

## Overview

The Bus Evaluation Pipeline is designed to evaluate the effectiveness of bus schedule optimizations by comparing original simulation data with optimized results. It integrates with prediction models to assess both optimization quality and prediction accuracy.

## Features

### 🔍 **Comprehensive Analysis**
- **Constraint Validation**: Ensures optimization adheres to operational constraints
- **Statistical Analysis**: Performs rigorous statistical testing of improvements
- **Prediction Accuracy**: Evaluates prediction model performance when available
- **Performance Benchmarking**: Compares multiple optimization approaches

### 📊 **Advanced Visualizations**
- Occupancy rate distribution comparisons
- Hourly pattern analysis
- Line-wise performance metrics
- Overcrowding reduction analysis
- Constraint validation charts

### 📈 **Detailed Reporting**
- Comprehensive evaluation reports
- JSON data export for further analysis
- Statistical significance testing
- Actionable insights and recommendations

## Installation

### From Source
```bash
git clone <repository-url>
cd bus_evaluation_pipeline
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from evaluation_engine import OptimizationEvaluator

# Initialize evaluator
evaluator = OptimizationEvaluator(output_dir="my_evaluation_results")

# Load data
evaluator.load_data(
    original_file="original_simulation.csv",
    optimized_file="optimized_results.csv",
    predicted_file="predicted_data.csv",  # Optional
    schedule_file="schedule_structure.csv"  # Optional
)

# Run comprehensive evaluation
validation, stats, prediction_accuracy = evaluator.run_evaluation()

# Results are automatically saved to output_dir
```

### Command Line Usage
```bash
# Run evaluation with default settings
bus-evaluate

# Custom evaluation (modify the script as needed)
python -m evaluation_engine.optimization_evaluator
```

## Data Requirements

### Required Columns
All input CSV files must contain:
- `datetime`: Timestamp of the record
- `line_id`: Bus line identifier
- `stop_id`: Bus stop identifier
- `boarding`: Number of passengers boarding
- `alighting`: Number of passengers alighting
- `occupancy_rate`: Bus occupancy rate (0.0 to 2.0+)

### Optional Columns
- `bus_id`: Individual bus identifier
- `trip_id`: Trip identifier
- `hour`: Hour of day (auto-generated if missing)

## Output Structure

```
evaluation_outputs/
├── plots/
│   ├── optimization_impact_analysis.png
│   └── constraint_validation.png
├── reports/
│   └── optimization_evaluation_report.txt
└── data/
    └── evaluation_results.json
```

## Evaluation Metrics

### Constraint Validation
- **Passenger Conservation**: Ensures passenger flow is maintained
- **Resource Constraints**: Validates bus and route utilization
- **Operational Constraints**: Checks schedule feasibility
- **Service Quality**: Measures improvement in service metrics

### Statistical Analysis
- **Occupancy Rate Improvement**: T-tests and effect size analysis
- **Overcrowding Reduction**: Chi-square tests for significance
- **Correlation Analysis**: Relationship between original and optimized data

### Prediction Accuracy (when available)
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **MAE/RMSE**: Mean Absolute/Root Mean Square Error
- **Correlation**: Pearson correlation coefficients

## Configuration

### Custom Output Directory
```python
evaluator = OptimizationEvaluator(output_dir="custom_results")
```

### Visualization Settings
The package uses matplotlib and seaborn with customizable styling:
```python
import matplotlib.pyplot as plt
plt.style.use('your_preferred_style')
```

## Integration with Other Pipelines

### With Bus Prediction Pipeline
```python
# Use prediction results for accuracy evaluation
evaluator.load_data(
    original_file="simulation_results.csv",
    optimized_file="optimization_results.csv",
    predicted_file="prediction_results.csv"
)
```

### With Bus Optimization Pipeline
```python
# Evaluate genetic algorithm results
evaluator.load_data(
    original_file="baseline_simulation.csv",
    optimized_file="ga_optimized_schedules.csv"
)
```

## Advanced Features

### Custom Metrics
Extend the evaluator with custom metrics:
```python
class CustomEvaluator(OptimizationEvaluator):
    def custom_analysis(self):
        # Your custom analysis here
        pass
```

### Batch Evaluation
Evaluate multiple optimization runs:
```python
for run_id in range(10):
    evaluator = OptimizationEvaluator(f"results_run_{run_id}")
    # ... evaluation logic
```

## Troubleshooting

### Common Issues
1. **Missing Columns**: Ensure all required columns are present
2. **Date Format**: Use ISO format (YYYY-MM-DD HH:MM:SS)
3. **Data Alignment**: Original and optimized data should cover similar time periods
4. **Memory Usage**: Large datasets may require chunked processing

### Performance Tips
- Use smaller date ranges for initial testing
- Ensure data is properly indexed by datetime
- Consider sampling for very large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please refer to the project documentation or create an issue in the repository. 