# Fleet Demand Forecaster

## Problem Statement
Drivers are wasting fuel, time, and money idling in low-demand zones or getting caught in gridlock, while platforms face supply-demand mismatches during localized surges (e.g., a conference ending at ExCeL or peak hours in the West End). **An engine that accurately forecasts micro-neighborhood demand 1 to 2 hours in advance optimizes fleet positioning, lowers driver churn, and maximizes yield.**

## Project Structure
- `src/data/make_dataset.py`: Generates mock geographic data for London.
- `src/features/build_features.py`: Aggregates the demand data into Uber's H3 spatial hexes and creates time-series lag features.
- `src/models/train_model.py`: Implements an XGBoost Regressor with a TimeSeriesSplit backtesting framework structure.
- `pipeline.py`: Runs the complete workflow.

## Usage
1. Make sure you have python 3 installed and start up an environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete pipeline directly:
    ```bash
    python pipeline.py
    ```
