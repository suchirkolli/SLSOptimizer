"""
Module for accessing power supply data from POSOCO (Power System Operation Corporation Limited)
for Andhra Pradesh Smart Load Shedding Optimizer.

Note: Currently using pre-downloaded CSV files, but this module defines
the interface for future live data retrieval.
"""

import pandas as pd
from pathlib import Path


def fetch_posoco_data(config, use_existing=True):
    """
    Access power supply data, either from existing files or from source.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with API keys, paths, etc.
    use_existing : bool
        Whether to use existing data files (True) or fetch new data (False)
        
    Returns:
    --------
    pandas.DataFrame
        Power supply data
    """
    raw_dir = Path(config['data']['raw_dir'])
    existing_file = raw_dir / 'power_supply_data.csv'
    
    if use_existing and existing_file.exists():
        print(f"Loading existing power supply data from {existing_file}")
        df = pd.read_csv(existing_file)
        print(f"Loaded {len(df)} records")
        return df
    
    print("Fetching POSOCO power supply data...")
    # TODO: Implement actual data fetching logic for future updates
    # For now, we'll return an empty DataFrame with expected columns
    
    df = pd.DataFrame(columns=[
        'Date',
        'Energy Required (MU)',
        'Energy Met (MU)', 
        'Energy +/- (MU)',
        'Unrestricted Peak Demand (MW)',
        'Deficit/Surplus (MW)'
    ])
    
    print(f"Note: No data fetched. This is a placeholder for future API integration.")
    return df


def load_processed_power_data(config):
    """
    Load processed power data from Parquet file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with paths
        
    Returns:
    --------
    pandas.DataFrame
        Processed power data
    """
    processed_dir = Path(config['data']['processed_dir'])
    parquet_file = processed_dir / 'demand.parquet'
    
    if parquet_file.exists():
        print(f"Loading processed power data from {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} records from processed power data")
        return df
    else:
        print(f"Warning: Processed power data file {parquet_file} not found")
        return pd.DataFrame()


if __name__ == "__main__":
    # Simple test to ensure the module can be run independently
    from config.config_loader import load_config
    
    config = load_config()
    raw_df = fetch_posoco_data(config)
    processed_df = load_processed_power_data(config)
    
    print("\nRaw data preview:")
    print(raw_df.head())
    
    print("\nProcessed data preview:")
    print(processed_df.head())