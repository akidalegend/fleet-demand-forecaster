import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

def train_and_evaluate(demand_df, features, target='demand_count', n_splits=3):
    X = demand_df[features]
    y = demand_df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    print("--- Starting Analytical Time-Series Split Verification ---")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100
        print(f"Fold {fold + 1} Out-of-Sample Validation MAPE: {mape:.2f}%")
        
    return model
