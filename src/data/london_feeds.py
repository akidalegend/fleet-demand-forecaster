import os
import sqlite3
import requests
import cv2
import numpy as np
import pandas as pd
import h3
from ultralytics import YOLO

class LondonRealTimeFeeds:
    def __init__(self, db_path="fleet_cache.db"):
        # Initialize standard pre-trained YOLOv8 nano model (lightweight, optimized for edge-inference)
        # Automatically downloads 'yolov8n.pt' to the local execution workspace on the first run
        self.vision_model = YOLO("yolov8n.pt")
        self.tfl_camera_api_url = "https://api.tfl.gov.uk/Place/Type/JamCam"
        self.db_path = db_path

    def fetch_live_camera_registry(self, limit=5):
        """
        Queries the live TfL API to locate active JamCams, extracting their 
        current dynamic image URLs, IDs, and precise lat/lng coordinates.
        """
        try:
            print("Querying TfL Unified API for live camera registry...")
            response = requests.get(self.tfl_camera_api_url, timeout=12)
            if response.status_code != 200:
                print(f"TfL API Error: Status code {response.status_code}")
                return []
            
            places = response.json()
            active_cameras = []
            
            for place in places[:limit]:
                camera_id = place.get("id")
                lat = place.get("lat")
                lng = place.get("lon")
                
                # Extract the dynamic live video/image URL from nested additional properties
                image_url = None
                for prop in place.get("additionalProperties", []):
                    if prop.get("key") == "imageUrl":
                        image_url = prop.get("value")
                        break
                
                if camera_id and lat and lng and image_url:
                    active_cameras.append({
                        "camera_id": camera_id,
                        "lat": float(lat),
                        "lng": float(lng),
                        "video_url": image_url
                    })
            
            print(f"Successfully registered {len(active_cameras)} live London cameras.")
            return active_cameras
        except Exception as e:
            print(f"Failed to fetch live TfL camera registry: {e}")
            return []

    def extract_pedestrian_density(self, image_url):
        """
        Fetches the active live frame from a verified TfL traffic camera 
        and applies YOLOv8 object detection to count pedestrians.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0 FleetDemandForecaster/1.0'}
            response = requests.get(image_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return 0
                
            # Decode image stream from network bytes directly into OpenCV matrix in memory
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return 0

            # Run inference targeting only class 0 (Person) within the COCO dataset
            results = self.vision_model(frame, classes=[0], verbose=False)
            
            # Count bounding boxes corresponding to detected people
            pedestrian_count = len(results[0].boxes)
            return pedestrian_count
        except Exception as e:
            print(f"Vision inference anomaly bypassed: {e}")
            return 0

    def fetch_tfl_disruptions(self):
        """
        Queries TfL Line Status API to detect systemic transit delays.
        """
        try:
            url = "https://api.tfl.gov.uk/Line/Mode/tube/Status"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return 0
                
            data = response.json()
            disrupted_lines = 0
            for line in data:
                statuses = line.get("lineStatuses", [])
                for status in statuses:
                    # Severity codes below 10 indicate delays, part suspensions, or structural closures
                    if status.get("statusSeverity", 10) < 10:
                        disrupted_lines += 1
                        break 
            return disrupted_lines
        except Exception as e:
            print(f"Transit API feed failure bypassed: {e}")
            return 0

    def write_to_cache(self, df):
        """
        Persists a live feature dataframe slice into the local SQLite feature cache
        to construct a chronological history for lag generation.
        """
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        write_df = df.copy()
        
        # Format timestamps to strings for pure SQLite text field compatibility
        write_df['time_bucket'] = write_df['time_bucket'].astype(str)
        
        db_df = write_df[['time_bucket', 'h3_geo', 'demand_count', 'pedestrian_density', 'transit_disruption_score']]
        
        # Multi-row insert tracking
        db_df.to_sql("live_features", conn, if_exists="append", index=False, method="multi")
        conn.close()
        print("Live features cached into local database store successfully.")

    def compile_live_features(self):
        """
        Aggregates dynamic vision metrics, transit disruption states, and 
        spatial indexing, then writes them to the caching tier.
        """
        current_time = pd.Timestamp.now().floor('15min')
        records = []
        
        camera_registry = self.fetch_live_camera_registry(limit=5)
        network_friction = self.fetch_tfl_disruptions()
        
        if not camera_registry:
            print("Warning: Camera registry is empty. Execution halted.")
            return pd.DataFrame()
        
        for cam in camera_registry:
            print(f"Processing YOLOv8 pedestrian count for: {cam['camera_id']}")
            pedestrians = self.extract_pedestrian_density(cam['video_url'])
            h3_index = h3.latlng_to_cell(cam['lat'], cam['lng'], res=8)
            
            # Proxy calculation: Simulate immediate fleet transaction volume proportional to pedestrian presence
            mock_live_demand = int(np.random.poisson(lam=pedestrians * 0.2 + 2))
            
            records.append({
                "time_bucket": current_time,
                "h3_geo": h3_index,
                "demand_count": mock_live_demand,
                "pedestrian_density": pedestrians,
                "transit_disruption_score": network_friction,
                "hour": current_time.hour,
                "day_of_week": current_time.dayofweek
            })
            
        live_df = pd.DataFrame(records)
        self.write_to_cache(live_df)
        return live_df

    def hydrate_lags_from_cache(self, live_df):
        """
        Queries the database to reconstruct lag_1 and lag_4 metrics for 
        the active live spatial coordinates, preventing cold-start failures.
        """
        conn = sqlite3.connect(self.db_path)
        current_time = live_df['time_bucket'].iloc[0]
        
        t_minus_15 = str(current_time - pd.Timedelta(minutes=15))
        t_minus_60 = str(current_time - pd.Timedelta(minutes=60))
        
        hydrated_records = []
        
        for _, row in live_df.iterrows():
            h3_geo = row['h3_geo']
            
            # Extract historical states from cache matching specific H3 cells
            res_1 = pd.read_sql_query(
                f"SELECT demand_count FROM live_features WHERE time_bucket='{t_minus_15}' AND h3_geo='{h3_geo}'", conn
            )
            res_4 = pd.read_sql_query(
                f"SELECT demand_count FROM live_features WHERE time_bucket='{t_minus_60}' AND h3_geo='{h3_geo}'", conn
            )
            
            # Assign cached values or default to 0 if the historical window is unpopulated
            row['lag_1'] = float(res_1['demand_count'].iloc[0]) if not res_1.empty else 0.0
            row['lag_4'] = float(res_4['demand_count'].iloc[0]) if not res_4.empty else 0.0
            hydrated_records.append(row)
            
        conn.close()
        return pd.DataFrame(hydrated_records)

if __name__ == "__main__":
    feeder = LondonRealTimeFeeds()
    live_df = feeder.compile_live_features()
    print("\n--- Testing Live Cached Lookup Execution ---")
    if not live_df.empty:
        hydrated_df = feeder.hydrate_lags_from_cache(live_df)
        print(hydrated_df[['time_bucket', 'h3_geo', 'pedestrian_density', 'lag_1', 'lag_4']])