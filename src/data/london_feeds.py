import os
import requests
import cv2
import pandas as pd
import h3
from datetime import datetime
from ultralytics import YOLO

class LondonRealTimeFeeds:
    def __init__(self):
        # Initialize standard pre-trained YOLOv8 nano model (lightweight, rapid inference)
        self.vision_model = YOLO("yolov8n.pt") 
        
        # Selected high-yield TfL Jam Cam URLs for London hotspots mapping to approximate coordinates
        # In a full production build, parse the TfL camera list XML/JSON dynamically
        self.camera_registry = [
            {
                "camera_id": "JAM_CAM_OXFORD_CIRCUS",
                "lat": 51.5150, "lng": -0.1419,  
                "video_url": "https://s3-eu-west-1.amazonaws.com/tfl-jamcams-carportal/images/0000.07812.jpg" 
            },
            {
                "camera_id": "JAM_CAM_PICCADILLY",
                "lat": 51.5101, "lng": -0.1349,
                "video_url": "https://s3-eu-west-1.amazonaws.com/tfl-jamcams-carportal/images/0000.01635.jpg"
            }
        ]

    def extract_pedestrian_density(self, image_url):
        """
        Fetches the latest live frame from a TfL traffic camera 
        and applies YOLOv8 object detection to count pedestrians.
        """
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return 0
                
            # Stream image bytes directly into OpenCV format
            import numpy as np
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return 0

            # Run inference targeting only class 0 (Person) in COCO dataset
            results = self.vision_model(frame, classes=[0], verbose=False)
            
            # Count detected bounding boxes
            pedestrian_count = len(results[0].boxes)
            return pedestrian_count
        except Exception as e:
            print(f"Vision inference anomaly bypassed: {e}")
            return 0

    def fetch_tfl_disruptions(self):
        """
        Queries the TfL Unified API to detect active disruptions 
        across the London Underground network.
        """
        try:
            # TfL Open Data API endpoints are publicly accessible
            url = "https://api.tfl.gov.uk/Line/Mode/tube/Status"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return 0 # Default to no disruption multiplier
                
            data = response.json()
            # Quantify severity: count lines experiencing non-good service delays
            disrupted_lines = 0
            for line in data:
                statuses = line.get("lineStatuses", [])
                for status in statuses:
                    if status.get("statusSeverity", 10) < 10: # Severity < 10 indicates delays/closures
                        disrupted_lines += 1
            return disrupted_lines
        except Exception as e:
            print(f"Transit API feed failure bypassed: {e}")
            return 0

    def compile_live_features(self):
        """
        Aggregates vision metrics, transit disruption states, and 
        spatial indexing into an enriched structured dataframe.
        """
        current_time = pd.Timestamp.now().floor('15min')
        records = []
        
        # Check overall transit system health
        network_friction = self.fetch_tfl_disruptions()
        
        for cam in self.camera_registry:
            print(f"Processing computer vision pipeline for: {cam['camera_id']}")
            pedestrians = self.extract_pedestrian_density(cam['video_url'])
            
            # Map camera coordinates to our core H3 Spatial Resolution 8 cells
            h3_index = h3.geo_to_h3(cam['lat'], cam['lng'], resolution=8)
            
            records.append({
                "time_bucket": current_time,
                "h3_geo": h3_index,
                "pedestrian_density": pedestrians,
                "transit_disruption_score": network_friction,
                "hour": current_time.hour,
                "day_of_week": current_time.dayofweek
            })
            
        return pd.DataFrame(records)

if __name__ == "__main__":
    feeder = LondonRealTimeFeeds()
    live_df = feeder.compile_live_features()
    print("\n--- Live Processed Real-Time Feature Matrix Generated ---")
    print(live_df.to_string())