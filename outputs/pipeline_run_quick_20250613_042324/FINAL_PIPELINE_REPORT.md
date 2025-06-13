# Schedule Optimization Pipeline Report

**Generated:** 2025-06-13 04:52:15
**Mode:** QUICK
**Output Directory:** /Users/umutulasbalci/Desktop/Python/Senior Project/schedule_optimization_project/outputs/pipeline_run_quick_20250613_042324

## Pipeline Configuration

### Simulation Parameters
- **Period:** 2025-06-01 to 2025-06-07
- **Time Step:** 15 minutes
- **Buses per Line:** 6

## Results Summary

### Overall Performance
- **Validation Status:** ❌ FAIL
- **Validation Score:** 25.0%
- **Occupancy Improvement:** -39.2%
- **Statistical Significance:** ✅ YES

## Output Files

### Simulation Results
- **Passenger Flow:** `simulation_results/passenger_flow_results.csv`
- **Bus Positions:** `simulation_results/bus_position_results.csv`
- **Summary:** `simulation_results/simulation_summary.json`

### Prediction Results
- **Predictions:** `prediction_results/predicted_passenger_flow.csv`
- **Model Metrics:** `prediction_results/model_metrics.json`

### Optimization Results
- **Schedules:** `optimization_results/optimized_schedules.csv`
- **Metrics:** `optimization_results/optimization_metrics.json`
- **Optimized Schedules:** `{'101': ScheduleChromosome(line=101, departures=36, fitness=0.8121), '101-1': ScheduleChromosome(line=101-1, departures=31, fitness=0.8126), '102-1': ScheduleChromosome(line=102-1, departures=64, fitness=0.8137), '102-2': ScheduleChromosome(line=102-2, departures=61, fitness=0.8169), '103-1': ScheduleChromosome(line=103-1, departures=45, fitness=0.7953), '103-2': ScheduleChromosome(line=103-2, departures=66, fitness=0.7769), '104-1': ScheduleChromosome(line=104-1, departures=41, fitness=0.7822), '104-2': ScheduleChromosome(line=104-2, departures=43, fitness=0.7861), '105-1': ScheduleChromosome(line=105-1, departures=56, fitness=0.7604), '105-2': ScheduleChromosome(line=105-2, departures=69, fitness=0.6943)}`
- **Passenger Flow Format:** `optimization_results/optimized_schedules_passenger_flow.csv`
