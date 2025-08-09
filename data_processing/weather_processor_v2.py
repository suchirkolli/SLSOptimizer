"""
Process weather data and ensure it's associated with districts.
Since weather data is at state level, this simply replicates it for all districts.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def replicate_weather_for_districts(state_weather_df, district_ids):
    """
    Replicate state-level weather data for each district.
    
    Parameters:
    -----------
    state_weather_df : DataFrame
        State-level weather data with 'timestamp', 'temperature', etc.
    district_ids : list
        List of district IDs to replicate data for
        
    Returns:
    --------
    DataFrame
        District-level weather data (same data replicated)
    """
    # Create empty list to store district records
    district_records = []
    
    # For each timestamp in state data
    for _, state_row in state_weather_df.iterrows():
        timestamp = state_row['timestamp']
        
        # Extract weather metrics
        metrics = {col: state_row[col] for col in state_row.index 
                  if col != 'timestamp' and col != 'district_id' and not pd.isna(state_row[col])}
        
        # Replicate for each district
        for district_id in district_ids:
            # Create district record with same weather
            record = {'timestamp': timestamp, 'district_id': district_id}
            record.update(metrics)
            
            district_records.append(record)
    
    return pd.DataFrame(district_records)

def main():
    # Load data
    weather_df = pd.read_parquet('data/processed/weather.parquet')
    district_ref = pd.read_csv('data/geographic/ap_districts_reference.csv')
    
    district_ids = district_ref['district_id'].tolist()
    
    # Check if weather data already has district_id
    if 'district_id' in weather_df.columns:
        # Count unique districts
        n_districts = weather_df['district_id'].nunique()
        if n_districts == len(district_ids):
            print("Weather data already has district associations.")
            district_weather = weather_df
        else:
            print(f"Weather data has {n_districts} districts, but reference has {len(district_ids)}.")
            print("Regenerating district associations...")
            district_weather = replicate_weather_for_districts(weather_df, district_ids)
    else:
        # Replicate state weather for all districts
        print("Replicating state-level weather data for all districts...")
        district_weather = replicate_weather_for_districts(weather_df, district_ids)
    
    # Save processed data
    output_file = 'data/processed/district_weather.parquet'
    district_weather.to_parquet(output_file)
    print(f"Saved district-level weather data to {output_file}")
    
    return district_weather

if __name__ == "__main__":
    main()