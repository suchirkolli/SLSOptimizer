"""
Module for accessing weather data for Andhra Pradesh Smart Load Shedding Optimizer.

Note: Currently using pre-downloaded CSV files, but this module defines
the interface for future live data retrieval.
"""

import pandas as pd
from pathlib import Path


def fetch_weather_data(config, use_existing=True):
    """
    Access weather data, either from existing files or from source.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with API keys, paths, etc.
    use_existing : bool
        Whether to use existing data files (True) or fetch new data (False)
        
    Returns:
    --------
    pandas.DataFrame
        Weather data
    """
    raw_dir = Path(config['data']['raw_dir'])
    existing_file = raw_dir / 'ap_weather.csv'
    
    if use_existing and existing_file.exists():
        print(f"Loading existing weather data from {existing_file}")
        df = pd.read_csv(existing_file)
        print(f"Loaded {len(df)} records")
        return df
    
    print("Fetching weather data...")
    # TODO: Implement actual API calls to OpenWeatherMap or IMD for future updates
    # For now, return empty DataFrame with expected columns
    
    df = pd.DataFrame(columns=[
        'timestamp',
        'temperature',
        'humidity',
        'precipitation',
        'wind_speed'
    ])
    
    print(f"Note: No data fetched. This is a placeholder for future API integration.")
    return df


def load_processed_weather_data(config):
    """
    Load processed weather data from Parquet file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with paths
        
    Returns:
    --------
    pandas.DataFrame
        Processed weather data
    """
    processed_dir = Path(config['data']['processed_dir'])
    parquet_file = processed_dir / 'weather.parquet'
    
    if parquet_file.exists():
        print(f"Loading processed weather data from {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} records from processed weather data")
        return df
    else:
        print(f"Warning: Processed weather data file {parquet_file} not found")
        return pd.DataFrame()


if __name__ == "__main__":
    # Simple test to ensure the module can be run independently
    from config.config_loader import load_config
    
    config = load_config()
    raw_df = fetch_weather_data(config)
    processed_df = load_processed_weather_data(config)
    
    print("\nRaw data preview:")
    print(raw_df.head())
    
    print("\nProcessed data preview:")
    print(processed_df.head())