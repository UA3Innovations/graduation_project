"""
Bus Transit Optimization Pipeline

A genetic algorithm-based optimization system for bus transit schedules.
Depends on the bus_simulation_pipeline package for core simulation functionality.
"""

__version__ = "1.0.0"
__author__ = "Bus Transit Optimization Team"

from .optimization import ga_optimize, optimization_evaluation, optimize_all_lines

__all__ = [
    'ga_optimize',
    'optimization_evaluation', 
    'optimize_all_lines'
] 