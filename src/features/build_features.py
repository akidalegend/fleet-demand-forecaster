import pandas as pd

def create_spatiotemporal_features(raw_data):
    raw_data['time_bucket'] = raw_data['timestamp'].dt.floor('15min')

    # Group to get demand count per hex per interval
    demand_df = raw_data.groupby(['time_bucket', 'h3_geo']).size().reset_index(name='demand_count')
    demand_df = demand_df.sort_values(by='time_bucket').reset_index(drop=True)

    # Time metrics
    demand_df['hour'] = demand_df['time_bucket'].dt.hour
    demand_df['day_of_week'] = demand_df['time_bucket'].dt.dayofweek

    # Create 1-period (15m) and 4-period (1hr) demand lags per hexagon
    demand_df['lag_1'] = demand_df.groupby('h3_geo')['demand_count'].shift(1)
    demand_df['lag_4'] = demand_df.groupby('h3_geo')['demand_count'].shift(4)
    
    return demand_df.dropna().reset_index(drop=True)
