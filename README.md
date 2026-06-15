# Algorithmic Fleet Positioning & Yield Optimization Engine (London)

An end-to-end spatial-temporal machine learning pipeline that forecasts micro-neighborhood ride-hailing demand 1 to 2 hours in advance. By integrating multi-stream edge-inference over live public transit metrics and real-time Transport for London (TfL) CCTV feeds, this engine replaces reactive vehicle routing with proactive fleet allocation, directly addressing driver idle overhead and supply-demand mismatches.

## 🚀 Core Architectural Pillars

### 1. Computer Vision & Exogenous Feature Ingestion Loop
Traditional forecasting engines rely exclusively on historical demand lags, failing to anticipate real-time event anomalies. This engine handles high-frequency data streams natively:
* **TfL Jam Cams (Live Computer Vision):** Connects to the live TfL Place API asset registry to stream real-time roadside intersection images. It passes raw network binary frames directly into a pre-trained **YOLOv8 (Nano)** object detection model in memory, dynamically engineering a localized `pedestrian_density` metric.
* **TfL Unified Network Status API:** Continuously monitors active subway, rail, and bus line alerts. Delays or structural line suspensions are computed as an index of immediate transit-modality substitutions (commuters transitioning from trains to ride-hailing vehicles).

### 2. Spatial Discretization (Uber H3 Indexing)
To prevent generalized regional smoothing, the Greater London area is mapped into discrete geographic boundaries using Uber’s **H3 Hierarchical Spatial Index** at Resolution 8 (~700m wide hexagons). Demand metrics, crowd counts, and localized lag states are bound entirely to unique hexagonal spatial keys.

### 3. Leak-Proof Temporal Validation (The Chronological Split)
Standard random or row-based K-Fold cross-validation creates structural look-ahead bias in spatial-temporal problems when multiple regional entries occur within identical timestamps. This framework utilizes a rigid chronological time-series validation loop. Data splits occur strictly across a sorted timeline, ensuring that concurrent spatial indicators from the future cannot leak into past training weights.

---

## 📂 Project Structure

```text
├── pipeline.py                 # Core production gateway (Training vs. Live Inference execution)
├── requirements.txt            # System dependencies
└── src/
    ├── data/
    │   ├── london_feeds.py     # Live REST API orchestration, TfL parsing, and YOLOv8 pedestrian engine
    │   └── make_dataset.py     # Historical spatial distribution baseline generator
    ├── features/
    │   └── build_features.py   # Spatiotemporal aggregation & multi-period hexagonal lag engineering
    └── models/
        └── train_model.py      # Strict temporal split validation suite & XGBoost training pipeline