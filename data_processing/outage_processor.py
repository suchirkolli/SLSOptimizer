"""
Process outage data from outage_data.csv
"""

import pandas as pd
import os
from .schemas import standardize_timestamp, validate_date_range, check_missing_values, TIMESTAMP_COL, REGION_ID_COL

def process_outage_data(input_file: str, output_file: str = None):
    """
    Process historical outage data from outage_data.csv
    
    Args:
        input_file: Path to outage_data.csv
        output_file: Path to save processed output (default: None)
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing outage data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Display initial info
    print(f"Raw outage data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Based on your file, columns include:
    # 'outage_id', 'start_timestamp', 'end_timestamp', 'district',
    # 'affected_substation', 'feeder_affected', 'capacity_affected_mw',
    # 'customers_affected', 'outage_type', 'cause', 'season'
    
    # Standardize timestamp columns
    df = standardize_timestamp(df, 'start_timestamp')
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])
    
    # Calculate outage duration in hours
    df['outage_duration_hours'] = (df['end_timestamp'] - df[TIMESTAMP_COL]).dt.total_seconds() / 3600
    
    # Filter to study period (May 2020-May 2023)
    df = validate_date_range(df)
    
    # Check for missing values
    check_missing_values(df, "Outage data")
    
    # Add severity classification based on duration and affected customers
    df['severity'] = pd.cut(
        df['outage_duration_hours'], 
        bins=[0, 1, 4, 12, float('inf')], 
        labels=['minor', 'moderate', 'major', 'critical'],
        include_lowest=True
    )
    
    # Create normalized impact score (considering both affected capacity and customers)
    # Scale both to 0-1 range and take average
    if df['capacity_affected_mw'].max() > 0:
        df['capacity_impact'] = df['capacity_affected_mw'] / df['capacity_affected_mw'].max()
    else:
        df['capacity_impact'] = 0
        
    if df['customers_affected'].max() > 0:
        df['customer_impact'] = df['customers_affected'] / df['customers_affected'].max()
    else:
        df['customer_impact'] = 0
        
    df['impact_score'] = (df['capacity_impact'] + df['customer_impact']) / 2
    
    # Extract time components
    df['hour'] = df[TIMESTAMP_COL].dt.hour
    df['dayofweek'] = df[TIMESTAMP_COL].dt.dayofweek
    df['month'] = df[TIMESTAMP_COL].dt.month
    df['year'] = df[TIMESTAMP_COL].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)  # 5=Sat, 6=Sun
    
    # Binary flags for analysis
    df['is_planned'] = (df['outage_type'] == 'Planned').astype(int)
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed outage data to {output_file}")
    
    print(f"Processed outage data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage when run as script
    process_outage_data("data/raw/outage_data.csv", "data/processed/outages.parquet")