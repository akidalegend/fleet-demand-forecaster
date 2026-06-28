import os
import sqlite3
import numpy as np
import pandas as pd
import h3

DB_PATH = "fleet_cache.db"

def init_cache_database():
    """Initializes a local SQLite database acting as our production caching layer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create feature store table for live inference tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_features (
            time_bucket TEXT,
            h3_geo TEXT,
            demand_count INTEGER,
            pedestrian_density INTEGER,
            transit_disruption_score INTEGER,
            PRIMARY KEY (time_bucket, h3_geo)
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database cache layer initialized at: {DB_PATH}")

def generate_mock_london_data(num_records=10000):
    """
    Generates realistic historical ride distributions mapping to London hotspots
    incorporating temporal variations (rush hours, weekend nightlife spikes).
    """
    init_cache_database()
    
    # Hub coordinates: Central London, West End nightlife, ExCeL center
    hubs = {
        "central": {"lat": 51.5074, "lng": -0.1278, "weight": 0.5},
        "west_end": {"lat": 51.5135, "lng": -0.1584, "weight": 0.3},
        "excel": {"lat": 51.5082, "lng": 0.0247, "weight": 0.2}
    }
    
    # Generate random chronological timestamps spanning 7 days
    base_time = pd.Timestamp("2026-06-01 00:00:00")
    timestamps = [base_time + pd.Timedelta(minutes=15 * np.random.randint(0, 672)) for _ in range(num_records)]
    
    latitudes = []
    longitudes = []
    
    for ts in timestamps:
        hour = ts.hour
        day_type = ts.dayofweek # 5,6 = weekend
        
        # Route logic based on temporal heuristics (simulating realistic city patterns)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            # Commuter rush hour concentrates heavily into Central
            chosen_hub = hubs["central"]
        elif (hour >= 22 or hour <= 2) and day_type >= 4:
            # Late-night weekend spikes concentrate into West End
            chosen_hub = hubs["west_end"]
        else:
            # Default multinomial selection based on structural hub weights
            chosen_hub = np.random.choice(list(hubs.values()), p=[0.5, 0.3, 0.2])
            
        latitudes.append(chosen_hub["lat"] + np.random.normal(0, 0.015))
        longitudes.append(chosen_hub["lng"] + np.random.normal(0, 0.015))
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'lat': latitudes,
        'lng': longitudes
    })
    
    # Discretize locations into H3 resolution 8 cells
    df['h3_geo'] = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lng'], res=8), axis=1)    
    # To hydrate our historical training dataset with realistic exogenous metrics,
    # add synthetic metrics matched to the generated temporal structure
    df['time_bucket'] = df['timestamp'].dt.floor('15min')
    
    return df

if __name__ == "__main__":
    df = generate_mock_london_data()
    print(f"Successfully compiled {len(df)} rows of patterned historical spatial data.")