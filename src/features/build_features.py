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
        unique_hexes = df['h3_geo'].drop_duplicates()

        neighbor_df = pd.DataFrame({
            'h3_geo': unique_hexes,
            'neighbor_h3_geo': [
                [neighbor for neighbor in h3.k_ring(hex_id, 1) if neighbor != hex_id]
                if pd.notna(hex_id) else []
                for hex_id in unique_hexes
            ]
        }).explode('neighbor_h3_geo', ignore_index=True)

        if neighbor_df.empty:
            return pd.Series(0, index=df.index)

        base_df = df[['time_bucket', 'h3_geo']].merge(neighbor_df, on='h3_geo', how='left')

        demand_lookup = df[['time_bucket', 'h3_geo', 'demand_count']].rename(
            columns={'h3_geo': 'neighbor_h3_geo', 'demand_count': 'neighbor_demand'}
        )

        spatial_lag_df = (
            base_df
            .merge(demand_lookup, on=['time_bucket', 'neighbor_h3_geo'], how='left')
            .groupby(['time_bucket', 'h3_geo'], as_index=False)['neighbor_demand']
            .sum()
            .rename(columns={'neighbor_demand': 'spatial_lag_1'})
        )

        return df[['time_bucket', 'h3_geo']].merge(
            spatial_lag_df,
            on=['time_bucket', 'h3_geo'],
            how='left'
        )['spatial_lag_1'].fillna(0)

    demand_df['spatial_lag_1'] = compute_spatial_lag(demand_df)
    
    return demand_df.dropna().reset_index(drop=True)
