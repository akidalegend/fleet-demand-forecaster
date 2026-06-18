import numpy as np
import lightgbm as lgb

def train_and_evaluate(demand_df, features, target='demand_count', n_splits=3):
    """
    Trains and evaluates a LightGBM Regressor using a strict temporal split
    to prevent cross-spatial data leakage across simultaneous time buckets.
    """
    unique_timestamps = sorted(demand_df['time_bucket'].unique())
    total_intervals = len(unique_timestamps)
    
    test_window_size = total_intervals // (n_splits + 1)
    
    print("\n--- Starting Rigid Spatial-Temporal Split Verification ---")
    final_model = None
    
    for fold in range(n_splits):
        train_end_idx = test_window_size * (fold + 1)
        test_end_idx = train_end_idx + test_window_size
        
        train_times = unique_timestamps[:train_end_idx]
        test_times = unique_timestamps[train_end_idx:test_end_idx]
        
        train_df = demand_df[demand_df['time_bucket'].isin(train_times)]
        test_df = demand_df[demand_df['time_bucket'].isin(test_times)]
        
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]
        
        # Evaluate Naive Baseline
        naive_predictions = test_df['lag_1']
        naive_mape = np.mean(np.abs((y_test - naive_predictions) / np.maximum(y_test, 1))) * 100
        
        # Train LightGBM Challenger
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        model_mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100
        
        delta = naive_mape - model_mape
        print(f"Fold {fold + 1} -> Naive MAPE: {naive_mape:.2f}% | Model MAPE: {model_mape:.2f}% | Delta: {delta:+.2f}%")
        
        final_model = model
        
    return final_model