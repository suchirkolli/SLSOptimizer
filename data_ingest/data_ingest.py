"""
Main entry point for data access in the Smart Load Shedding Optimizer project.
This module coordinates access to data from all sources.

Note: Currently focused on loading existing processed data, with placeholders
for future data fetching capabilities.
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from data_ingest.fetch_posoco import fetch_posoco_data, load_processed_power_data
from data_ingest.fetch_weather import fetch_weather_data, load_processed_weather_data


def verify_data_availability():
    """
    Check if all required data files exist and are accessible.
    
    Returns:
    --------
    bool
        True if all required data is available, False otherwise
    """
    config = load_config()
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    
    # Define required files
    required_raw = [
        raw_dir / 'power_supply_data.csv',
        raw_dir / 'ap_weather.csv'
    ]
    
    required_processed = [
        processed_dir / 'demand.parquet',
        processed_dir / 'weather.parquet'
    ]
    
    # Check raw files
    missing_raw = [str(f) for f in required_raw if not f.exists()]
    if missing_raw:
        print(f"Missing raw data files: {missing_raw}")
    
    # Check processed files
    missing_processed = [str(f) for f in required_processed if not f.exists()]
    if missing_processed:
        print(f"Missing processed data files: {missing_processed}")
    
    return not (missing_raw or missing_processed)


def load_all_processed_data():
    """
    Load all processed data files into DataFrames.
    
    Returns:
    --------
    dict
        Dictionary with data type keys and DataFrame values
    """
    config = load_config()
    
    data = {}
    data['demand'] = load_processed_power_data(config)
    data['weather'] = load_processed_weather_data(config)
    
    # Load other data types as needed
    # data['outages'] = load_processed_outage_data(config)
    # data['grid_infra'] = load_processed_grid_data(config)
    
    # Check for data availability
    empty_datasets = [name for name, df in data.items() if df.empty]
    if empty_datasets:
        print(f"Warning: The following datasets are empty: {empty_datasets}")
    
    return data


def main():
    """
    Main function to verify data availability and load datasets.
    """
    print("Smart Load Shedding Optimizer - Data Verification")
    
    # Check data availability
    all_data_available = verify_data_availability()
    
    if all_data_available:
        print("All required data is available.")
        
        # Load processed data
        data = load_all_processed_data()
        
        # Print data summary
        print("\nData Summary:")
        for name, df in data.items():
            if not df.empty:
                print(f"- {name}: {len(df)} records, {df.shape[1]} columns")
    else:
        print("\nSome data files are missing. Please ensure all required data is available.")


if __name__ == "__main__":
    main()