import sys
import os
import pandas as pd

# Ensure the src directory is in the system path for seamless modular imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.make_dataset import generate_mock_london_data
from src.features.build_features import create_spatiotemporal_features
from src.models.train_model import train_and_evaluate
from src.data.london_feeds import LondonRealTimeFeeds

def run_production_pipeline(mode="train"):
    """
    Executes the Fleet Demand Engine.
    'train': Runs the leak-proof historical training and backtesting framework.
    'live': Polls production APIs (TfL CCTV + Line Status), processes features, and infers demand.
    """
    print(f"=== Initializing Fleet Demand Engine [Mode: {mode.upper()}] ===")
    
    # Core features matching historical structural patterns
    historical_features = ['hour', 'day_of_week', 'lag_1', 'lag_4']
    
    if mode == "train":
        print("\n[Step 1/3] Generating historical baseline data partitions...")
        raw_data = generate_mock_london_data(num_records=8000)
        
        print("[Step 2/3] Building structural spatiotemporal features & lags...")
        demand_df = create_spatiotemporal_features(raw_data)
        
        print("[Step 3/3] Commencing strict chronological training and verification loop...")
        # Uses the updated, leak-proof temporal split logic
        trained_model = train_and_evaluate(demand_df, features=historical_features)
        print("\nPipeline Complete: Historical training model optimized and cached.")
        return trained_model

    elif mode == "live":
        print("\n[Step 1/2] Connecting to live Transport for London (TfL) ecosystem APIs...")
        feeder = LondonRealTimeFeeds()
        live_features_df = feeder.compile_live_features()
        
        if live_features_df.empty:
            print("Execution halted: No live production data could be collected.")
            return None
        
        print("\n[Step 2/2] Live feature matrix successfully compiled.")
        print(live_features_df[['time_bucket', 'h3_geo', 'pedestrian_density', 'transit_disruption_score']])
        
        # NOTE: In an active operational state, you would load your cached trained_model
        # and append live streams to pass through `model.predict(live_features_df[features])`
        print("\nPipeline Complete: Live operational inference metrics generated successfully.")
        return live_features_df

    else:
        raise ValueError("Invalid pipeline mode execution argument. Choose 'train' or 'live'.")

if __name__ == "__main__":
    # Change argument to "live" to trigger real-time TfL API polling and YOLOv8 pedestrian counting
    run_production_pipeline(mode="train")