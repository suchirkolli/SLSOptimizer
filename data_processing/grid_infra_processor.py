"""
Process grid infrastructure data from grid_infra_data.csv
"""

import pandas as pd
import os
import numpy as np
from .schemas import check_missing_values, REGION_ID_COL

def process_grid_data(input_file: str, output_file: str = None):
    """
    Process grid infrastructure data from grid_infra_data.csv
    
    Args:
        input_file: Path to grid_infra_data.csv
        output_file: Path to save processed output (default: None)
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing grid infrastructure data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Display initial info
    print(f"Raw grid infra data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Based on your file, columns include:
    # 'substation_id', 'name', 'district', 'location', 'latitude', 'longitude',
    # 'capacity_mva', 'voltage_level_kv (input/output)', 'peak_load_mw'
    
    # Check for missing values
    check_missing_values(df, "Grid infrastructure data")
    
    # Parse voltage levels
    df['input_voltage_kv'] = df['voltage_level_kv (input/output)'].apply(
        lambda x: float(str(x).split('/')[0]) if pd.notna(x) else np.nan
    )
    df['output_voltage_kv'] = df['voltage_level_kv (input/output)'].apply(
        lambda x: float(str(x).split('/')[-1]) if pd.notna(x) and '/' in str(x) else np.nan
    )
    
    # Calculate utilization ratio (peak load / capacity)
    # Convert MVA to MW using power factor of 0.9 (typical assumption)
    df['capacity_mw'] = df['capacity_mva'] * 0.9
    df['utilization_ratio'] = df['peak_load_mw'] / df['capacity_mw']
    
    # Classify substations by voltage level
    voltage_bins = [0, 33, 132, 220, 400, float('inf')]
    voltage_labels = ['Low', 'Medium', 'High', 'Extra High', 'Ultra High']
    df['voltage_class'] = pd.cut(
        df['input_voltage_kv'], 
        bins=voltage_bins, 
        labels=voltage_labels,
        include_lowest=True
    )
    
    # Classify substations by utilization
    util_bins = [0, 0.5, 0.7, 0.9, float('inf')]
    util_labels = ['Low', 'Medium', 'High', 'Critical']
    df['utilization_class'] = pd.cut(
        df['utilization_ratio'], 
        bins=util_bins, 
        labels=util_labels,
        include_lowest=True
    )
    
    # Create aggregated district statistics
    district_stats = df.groupby(REGION_ID_COL).agg(
        substation_count=('substation_id', 'count'),
        total_capacity_mw=('capacity_mw', 'sum'),
        total_peak_load_mw=('peak_load_mw', 'sum'),
        avg_utilization=('utilization_ratio', 'mean'),
        critical_substations=('utilization_class', lambda x: sum(x == 'Critical'))
    ).reset_index()
    
    # Merge back district-level statistics
    df = pd.merge(df, district_stats, on=REGION_ID_COL, suffixes=('', '_district'))
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed grid infrastructure data to {output_file}")
        
        # Also save district-level aggregates
        district_output = output_file.replace('.parquet', '_district.parquet')
        district_stats.to_parquet(district_output)
        print(f"Saved district-level grid stats to {district_output}")
    
    print(f"Processed grid infrastructure data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage when run as script
    process_grid_data("data/raw/grid_infra_data.csv", "data/processed/grid_infra.parquet")