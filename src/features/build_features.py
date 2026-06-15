import pandas as pd
import h3

def create_spatiotemporal_features(raw_data):
    # Depending on pandas version, '15min' could be '15min' or '15T'
    # Ensuring consistency across operations
    raw_data['time_bucket'] = raw_data['timestamp'].dt.floor('15min')

    # Aggregating demand AND carrying over our systemic variance parameters
    # The external parameters are uniform across the city for a given time bucket
    agg_funcs = {
        'h3_geo': 'size', # Count serves as 'demand_count'
    }
    
    # Preserve our proprietary environmental/infrastructure factors
    for col in ['precipitation_mm', 'temperature_2m', 'is_tube_strike']:
        if col in raw_data.columns:
            agg_funcs[col] = 'first'
            
    # Group to get demand count per hex per interval, alongside the weather/transit conditions
    demand_df = raw_data.groupby(['time_bucket', 'h3_geo']).agg(agg_funcs).rename(columns={'h3_geo': 'demand_count'}).reset_index()
    demand_df = demand_df.sort_values(by='time_bucket').reset_index(drop=True)

    # Time metrics
    demand_df['hour'] = demand_df['time_bucket'].dt.hour
    demand_df['day_of_week'] = demand_df['time_bucket'].dt.dayofweek

    # Create temporal lags: 1-period (15m) and 4-period (1hr) demand lags per hexagon
    demand_df['temporal_lag_1'] = demand_df.groupby('h3_geo')['demand_count'].shift(1)
    demand_df['temporal_lag_4'] = demand_df.groupby('h3_geo')['demand_count'].shift(4)
    
    # Establish Domain Competence: Spatial Lags (Demand in adjacent hexagons during the same time threshold)
    # This reflects ride-hailing/fleet network effects (supply bleeding from adjacent zones)
    def compute_spatial_lag(df):
        # Create a mapping of (time_bucket, h3_geo) -> demand_count for quick lookup
        demand_map = df.set_index(['time_bucket', 'h3_geo'])['demand_count'].to_dict()
        spatial_lags = []
        
        for idx, row in df.iterrows():
            tb = row['time_bucket']
            hex_id = row['h3_geo']
            
            try:
                # Get immediate geographic neighbors (k=1)
                neighbors = h3.k_ring(hex_id, 1)
                # Ensure we exclude the center hex itself from the 'neighbors' aggregate
                neighbors.discard(hex_id)
                # Sum the demand from all neighboring hexagons at the exact same time bucket
                neighbor_demand = sum(demand_map.get((tb, n), 0) for n in neighbors)
                spatial_lags.append(neighbor_demand)
            except Exception:
                spatial_lags.append(0)
                
        return spatial_lags

    demand_df['spatial_lag_1'] = compute_spatial_lag(demand_df)
    
    return demand_df.dropna().reset_index(drop=True)
