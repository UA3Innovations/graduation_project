"""
Setup script for Bus Transit Simulation Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bus-transit-simulation-pipeline",
    version="1.0.0",
    author="Bus Transit Simulation Team",
    author_email="your.email@example.com",
    description="A comprehensive bus transit simulation pipeline with Azure cloud deployment capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bus-transit-simulation-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "azure": [
            "azure-containerinstance>=2.0.0",
            "azure-storage-blob>=12.0.0",
            "azure-identity>=1.7.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.csv", "config/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "bus-simulation=pipeline.pipeline_runner:main",
            "bus-deploy=scripts.deploy_to_azure:main",
        ],
    },
    keywords="simulation, transit, bus, transportation, azure, cloud, optimization",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bus-transit-simulation-pipeline/issues",
        "Source": "https://github.com/yourusername/bus-transit-simulation-pipeline",
        "Documentation": "https://github.com/yourusername/bus-transit-simulation-pipeline#readme",
    },
) 