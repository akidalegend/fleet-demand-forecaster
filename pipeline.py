from src.data.make_dataset import generate_mock_london_data
from src.features.build_features import create_spatiotemporal_features
from src.models.train_model import train_and_evaluate

def main():
    print("1. Generating mock London data...")
    raw_data = generate_mock_london_data()
    
    print("2. Aggregating demand and building features...")
    demand_df = create_spatiotemporal_features(raw_data)
    
    print("3. Training and evaluating the model...")
    features = ['hour', 'day_of_week', 'lag_1', 'lag_4']
    model = train_and_evaluate(demand_df, features=features)
    
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
