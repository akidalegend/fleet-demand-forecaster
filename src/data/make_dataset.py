import numpy as np
import pandas as pd
import h3
from fetch_external_data import enrich_with_external_variables

def generate_mock_london_data(num_records=5000):
    # Core London Coordinates (approx Trafalgar Square centroid)
    lat_center, lon_center = 51.5074, -0.1278
    
    latitudes = lat_center + np.random.normal(0, 0.05, num_records)
    longitudes = lon_center + np.random.normal(0, 0.05, num_records)
    
    # Generate 15-minute intervals over 5 days
    base_time = pd.Timestamp("2026-06-01 00:00:00")
    timestamps = [base_time + pd.Timedelta(minutes=15 * np.random.randint(0, 480)) for _ in range(num_records)]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'lat': latitudes,
        'lng': longitudes
    })
    
    # Map coordinates to Uber H3 Hexagons (Resolution 8 ~ 700m wide)
    df['h3_geo'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lng'], resolution=8), axis=1)
    
    # Introduce real-world systemic variance from external APIs
    df = enrich_with_external_variables(df)
    
    return df

if __name__ == "__main__":
    df = generate_mock_london_data()
    print(f"Generated {len(df)} records of mock data.")
