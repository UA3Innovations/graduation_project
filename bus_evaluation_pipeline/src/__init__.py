"""
Bus Evaluation Pipeline

A comprehensive package for evaluating bus schedule optimization results,
comparing simulation outcomes with optimized schedules using predicted passenger flow.
"""

__version__ = "1.0.0"
__author__ = "Senior Project Team"
__description__ = "Advanced evaluation system for bus transit optimization"

from .evaluation_engine import *

__all__ = [
    'OptimizationEvaluator'
] 