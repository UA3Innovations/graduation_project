#!/usr/bin/env python3
"""
Basic tests for the bus simulation pipeline.
"""

import sys
import os
from pathlib import Path
import unittest

# Add project root to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
pipeline_path = project_root / 'pipeline'

# Add paths to sys.path if not already there
for path in [str(src_path), str(pipeline_path), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

class TestPipelineStructure(unittest.TestCase):
    """Test basic pipeline structure and imports."""
    
    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        try:
            # Try direct imports first
            from core.data_models import BusTransitData, SimulationConfig
            from core.simulation_engine import SimulationEngine
            self.assertTrue(True, "Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")
    
    def test_import_components(self):
        """Test that component modules can be imported."""
        try:
            # Import individual component modules directly
            from components.transit_network import TransitNetwork
            from components.passenger_generator import PassengerGenerator
            from components.bus_management import BusManager
            from components.schedule_generator import ScheduleGenerator
            self.assertTrue(True, "Component modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import component modules: {e}")
    
    def test_import_pipeline_runner(self):
        """Test that pipeline runner can be imported."""
        try:
            from pipeline_runner import SimulationPipeline
            self.assertTrue(True, "Pipeline runner imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import pipeline runner: {e}")
    
    def test_data_models_basic(self):
        """Test basic data model creation."""
        try:
            from core.data_models import BusTransitData, SimulationConfig
            from datetime import datetime
            
            # Test data container creation
            data = BusTransitData()
            self.assertIsInstance(data.stops, dict)
            self.assertIsInstance(data.lines, dict)
            self.assertIsInstance(data.buses, dict)
            
            # Test config creation
            config = SimulationConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                time_step=5
            )
            self.assertEqual(config.time_step, 5)
            
        except Exception as e:
            self.fail(f"Failed basic data model test: {e}")
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        # Check config directory structure
        config_dir = project_root / 'config'
        self.assertTrue(config_dir.exists(), "Config directory missing")
        
        # Check simulation config
        sim_config = config_dir / 'simulation_config.yaml'
        self.assertTrue(sim_config.exists(), "Simulation config file missing")
    
    def test_data_files_exist(self):
        """Test that data files exist."""
        data_dir = project_root / 'data'
        
        # Check CSV data file
        csv_file = data_dir / 'ankara_bus_stops.csv'
        self.assertTrue(csv_file.exists(), "Bus stops CSV file missing")
    
    def test_pipeline_configuration_loading(self):
        """Test that pipeline can load configuration."""
        try:
            from pipeline_runner import SimulationPipeline
            
            # Test with default config
            pipeline = SimulationPipeline()
            self.assertIsNotNone(pipeline.config)
            self.assertIn('simulation', pipeline.config)
            
        except Exception as e:
            self.fail(f"Failed to load pipeline configuration: {e}")


class TestPipelineValidation(unittest.TestCase):
    """Test pipeline validation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from pipeline_runner import SimulationPipeline
            self.pipeline = SimulationPipeline()
        except ImportError as e:
            self.skipTest(f"Cannot import pipeline runner: {e}")
    
    def test_input_validation(self):
        """Test input validation functionality."""
        try:
            # Should not raise an exception
            result = self.pipeline.validate_inputs()
            # Just check that it returns a boolean
            self.assertIsInstance(result, bool)
            
        except Exception as e:
            self.fail(f"Input validation test failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPipelineStructure))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPipelineValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code) 