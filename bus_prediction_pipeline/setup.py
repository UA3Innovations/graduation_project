#!/usr/bin/env python3
"""
Setup script for Bus Prediction Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Bus Prediction Pipeline - Advanced passenger flow prediction for bus transit systems"

setup(
    name="bus-prediction-pipeline",
    version="1.0.0",
    author="Senior Project Team",
    author_email="",
    description="Advanced passenger flow prediction for bus transit systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        
        # Deep Learning
        "tensorflow>=2.8.0",
        
        # Time Series Forecasting
        "prophet>=1.1.0",
        
        # Utilities
        "argparse",
        "warnings",
        "pickle-mixin",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "plotting": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "predict-lstm=prediction_models.lstm_model:main",
            "predict-prophet=prediction_models.prophet_model:main", 
            "predict-hybrid=prediction_models.hybrid_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
) 