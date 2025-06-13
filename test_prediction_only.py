#!/usr/bin/env python3
"""
Test script to run only the prediction phase using existing optimization results.
This helps isolate and debug the prediction step without running the full pipeline.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "bus_prediction_pipeline" / "src"))

from bus_prediction_pipeline.src.prediction_models.hybrid_model import HybridModel

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_prediction_only():
    """Test only the prediction phase using existing optimization results"""
    
    logger = setup_logging()
    logger.info("🧪 TESTING PREDICTION PHASE ONLY")
    logger.info("=" * 60)
    
    # Use the most recent pipeline run
    latest_run = "pipeline_run_quick_20250613_031137"
    base_path = project_root / "outputs" / latest_run
    
    # Input files from previous pipeline steps
    simulation_data = base_path / "simulation_results" / "passenger_flow_results.csv"
    optimized_schedule_flow = base_path / "optimization_results" / "optimized_schedules_passenger_flow.csv"
    
    # Output directory for prediction results
    prediction_output_dir = base_path / "prediction_results_test"
    prediction_output_dir.mkdir(exist_ok=True)
    
    logger.info(f"📁 Using data from: {latest_run}")
    logger.info(f"📄 Simulation data: {simulation_data}")
    logger.info(f"📄 Optimized schedule flow: {optimized_schedule_flow}")
    logger.info(f"📁 Output directory: {prediction_output_dir}")
    
    try:
        # Load the data
        logger.info("📊 Loading simulation training data...")
        train_data = pd.read_csv(simulation_data)
        logger.info(f"   Training data shape: {train_data.shape}")
        
        logger.info("📊 Loading optimized schedule structure...")
        future_data = pd.read_csv(optimized_schedule_flow)
        logger.info(f"   Future data shape: {future_data.shape}")
        
        # Check data structure
        logger.info(f"📋 Training data columns: {list(train_data.columns)}")
        logger.info(f"📋 Future data columns: {list(future_data.columns)}")
        
        # Initialize hybrid model
        logger.info("🤖 Initializing Hybrid Model (LSTM + Prophet)...")
        model = HybridModel()
        
        # Create models directory
        models_dir = prediction_output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Train the model
        logger.info("🎯 Training hybrid model...")
        success = model.train_hybrid_models(train_data, save_models=True, model_save_dir=str(models_dir))
        if not success:
            raise Exception("Model training failed")
        logger.info("✅ Model training completed")
        
        # Make predictions
        logger.info("🔮 Making predictions on optimized schedule...")
        predictions = model.predict_hybrid(train_data, future_data)
        
        # Save predictions
        predictions_file = prediction_output_dir / "hybrid_predictions.csv"
        logger.info(f"💾 Saving predictions to {predictions_file}...")
        predictions.to_csv(predictions_file, index=False)
        
        # Generate summary
        logger.info("📈 PREDICTION TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📄 Predictions saved: {predictions_file}")
        logger.info(f"📊 Predictions shape: {predictions.shape}")
        logger.info(f"🚌 Unique trips predicted: {predictions['trip_id'].nunique() if 'trip_id' in predictions.columns else 'N/A'}")
        logger.info(f"🚏 Unique stops: {predictions['stop_id'].nunique()}")
        logger.info(f"🚌 Unique buses: {predictions['bus_id'].nunique()}")
        
        # Show sample predictions
        logger.info(f"\n🔍 Sample predictions (first 5 rows):")
        sample_cols = ['datetime', 'line_id', 'stop_id', 'bus_id', 'boarding', 'alighting', 'occupancy_rate']
        available_cols = [col for col in sample_cols if col in predictions.columns]
        print(predictions[available_cols].head())
        
        # Show prediction statistics
        if 'boarding' in predictions.columns:
            logger.info(f"\n📊 Boarding statistics:")
            logger.info(f"   Total boarding: {predictions['boarding'].sum():,.0f}")
            logger.info(f"   Average boarding per stop: {predictions['boarding'].mean():.1f}")
            logger.info(f"   Max boarding at stop: {predictions['boarding'].max():.0f}")
        
        if 'alighting' in predictions.columns:
            logger.info(f"\n📊 Alighting statistics:")
            logger.info(f"   Total alighting: {predictions['alighting'].sum():,.0f}")
            logger.info(f"   Average alighting per stop: {predictions['alighting'].mean():.1f}")
            logger.info(f"   Max alighting at stop: {predictions['alighting'].max():.0f}")
        
        if 'occupancy_rate' in predictions.columns:
            logger.info(f"\n📊 Occupancy statistics:")
            logger.info(f"   Average occupancy: {predictions['occupancy_rate'].mean():.1%}")
            logger.info(f"   Max occupancy: {predictions['occupancy_rate'].max():.1%}")
            logger.info(f"   Records >100% capacity: {(predictions['occupancy_rate'] > 1.0).sum()}")
        
        logger.info(f"\n✅ PREDICTION TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_only()
    sys.exit(0 if success else 1) 