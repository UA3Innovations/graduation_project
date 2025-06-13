# Bus Prediction Pipeline

A comprehensive package for passenger flow prediction using advanced machine learning models including LSTM, Prophet, and Hybrid approaches.

## Overview

The Bus Prediction Pipeline provides state-of-the-art passenger flow prediction capabilities for bus transit systems. It combines multiple prediction models to deliver accurate, reliable forecasts that can be used for route optimization, capacity planning, and service improvement.

## Features

### 🧠 **Multiple Prediction Models**
- **LSTM Model**: Deep learning approach with route-aware fixes and realistic constraints
- **Prophet Model**: Time series forecasting with seasonality and holiday effects
- **Hybrid Model**: Intelligent combination of LSTM and Prophet with date-based weighting

### 🚌 **Transit-Specific Features**
- Route-aware adjustments based on stop classifications
- Last stop logic (boarding=0, alighting=current_load)
- Realistic boarding constraints with driver discretion
- Night-time logic (1-4 AM specific constraints)
- Conservation of passenger flow principles

### 📊 **Advanced Capabilities**
- Historical pattern analysis and preservation
- Multi-objective optimization compatibility
- Comprehensive validation and reporting
- Special date handling (holidays, events)
- Robust error handling and debugging

## Installation

### From Source
```bash
git clone <repository-url>
cd bus_prediction_pipeline
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```python
from prediction_models import HybridModel
import pandas as pd

# Initialize the hybrid model
model = HybridModel(sequence_length=48)

# Load your historical data
historical_df = pd.read_csv('historical_passenger_data.csv')

# Train the model
model.train_hybrid_models(historical_df, save_models=True)

# Create future time periods for prediction
future_df = pd.DataFrame({
    'datetime': pd.date_range('2025-06-01', periods=144, freq='10min'),
    'line_id': '101',
    'stop_id': 1001
})

# Generate predictions
predictions = model.predict_hybrid(
    historical_df=historical_df,
    future_df=future_df,
    apply_route_adjustments=True,
    apply_realistic_constraints=True
)
```

### 2. Individual Models

#### LSTM Model
```python
from prediction_models import lstm_model

# Initialize LSTM model
lstm = lstm_model(sequence_length=48)

# Analyze patterns and train
lstm.analyze_historical_patterns(historical_df)
prepared_data = lstm.prepare_enhanced_features(historical_df)
train_X, train_y, val_X, val_y, metadata = lstm.create_pattern_sequences(prepared_data)
lstm.train_with_pattern_validation(train_X, train_y, val_X, val_y)

# Generate predictions
predictions = lstm.predict_with_patterns(historical_df, future_df)
```

#### Prophet Model
```python
from prediction_models import ProphetModel

# Initialize Prophet model
prophet = ProphetModel()

# Train and predict
prophet.analyze_historical_patterns(historical_df)
prophet.train_prophet_models(historical_df)
predictions = prophet.predict_with_patterns(historical_df, future_df)
```

## Data Format

### Input Data Requirements

Your historical data should be a pandas DataFrame with the following columns:

```python
historical_df = pd.DataFrame({
    'datetime': pd.Timestamp,     # Timestamp of the observation
    'line_id': str,               # Bus line identifier
    'stop_id': int,               # Bus stop identifier  
    'boarding': int,              # Number of passengers boarding
    'alighting': int,             # Number of passengers alighting
    'current_load': int,          # Current passenger load on bus
    'capacity': int               # Bus capacity
})
```

### Bus Stops Data

Place your bus stops CSV file in the `data/` directory as `ankara_bus_stops.csv`:

```csv
line_id,stop_id,stop_name
101,1001,Central Station
101,1002,University Campus
...
```

## Model Configuration

### Hybrid Model Weighting
- **Normal days**: 70% LSTM + 30% Prophet
- **Special days**: 30% LSTM + 70% Prophet

Special days include Turkish national holidays and Islamic holidays.

### Constraints and Logic
- **Conservation**: Total boarding ≈ Total alighting across the system
- **Capacity**: Boarding limited by available bus capacity
- **Last Stop**: Automatic passenger discharge at route terminals
- **Night Time**: Reduced activity during 1-4 AM period

## Command Line Interface

The package provides command-line tools for each model:

```bash
# LSTM predictions
predict-lstm --historical-data data.csv --future-data future.csv --output predictions.csv

# Prophet predictions  
predict-prophet --historical-data data.csv --future-data future.csv --output predictions.csv

# Hybrid predictions
predict-hybrid --historical-data data.csv --future-data future.csv --output predictions.csv
```

## Integration with Optimization

The prediction models are designed to integrate seamlessly with genetic algorithm optimization:

```python
# Use predictions in optimization fitness evaluation
from bus_optimization_pipeline import GeneticScheduleOptimizer

# Your optimization code here...
# The predictions can be used to evaluate schedule performance
```

## Package Structure

```
bus_prediction_pipeline/
├── src/
│   ├── prediction_models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py      # LSTM implementation
│   │   ├── prophet_model.py   # Prophet implementation
│   │   └── hybrid_model.py    # Hybrid model combining both
│   └── __init__.py
├── data/
│   └── ankara_bus_stops.csv   # Bus stops reference data
├── config/                    # Configuration files
├── tests/                     # Unit tests
├── scripts/                   # Utility scripts
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── pyproject.toml            # Modern Python packaging
└── README.md                 # This file
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
```

### Type Checking
```bash
mypy src/
```

## Performance

- **LSTM Training**: ~2-5 minutes for typical datasets
- **Prophet Training**: ~30-60 seconds per stop
- **Hybrid Predictions**: ~10-30 seconds for daily forecasts
- **Memory Usage**: ~500MB-2GB depending on data size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please refer to the project documentation or create an issue in the repository. 