"""
Process historical electricity demand data from power_supply_data.csv
"""

import pandas as pd
import os
from .schemas import standardize_timestamp, validate_date_range, check_missing_values, TIMESTAMP_COL, REGION_ID_COL

def process_demand_data(input_file: str, output_file: str = None):
    """
    Process historical electricity demand data from power_supply_data.csv
    
    Args:
        input_file: Path to power_supply_data.csv
        output_file: Path to save processed output (default: None)
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing demand data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Display initial info
    print(f"Raw demand data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Based on your file, columns include:
    # 'Date', 'Energy Required (MU)', 'Energy Met (MU)', 'Energy +/- (MU)',
    # 'Unrestricted Peak Demand (MW)', 'Deficit/Surplus (MW)'
    
    # Standardize timestamp column
    df = standardize_timestamp(df, 'Date')
    
    # Since no district/region information, we'll default to a single region
    # In a real-world scenario, you might want to extract the region from the filename
    # or add it as a parameter
    df[REGION_ID_COL] = 'ANDHRA_PRADESH'  # Default to AP based on substation naming
    
    # Rename demand columns
    df = df.rename(columns={
        'Energy Required (MU)': 'energy_required_mu',
        'Energy Met (MU)': 'energy_met_mu',
        'Energy +/- (MU)': 'energy_diff_mu',
        'Unrestricted Peak Demand (MW)': 'peak_demand_mw',
        'Deficit/Surplus (MW)': 'deficit_surplus_mw'
    })
    
    # Convert MU (Million Units) to MW-hour equivalent for consistency
    # 1 MU = 1000 MWh, but if we're using daily data, we need to calculate avg MW
    df['avg_demand_mw'] = df['energy_required_mu'] * 1000 / 24  # Convert to avg MW assuming daily data
    
    # Filter to study period (May 2020-May 2023)
    df = validate_date_range(df)
    
    # Check for missing values 
    check_missing_values(df, "Demand data")
    
    # Add time-based features
    df['hour'] = df[TIMESTAMP_COL].dt.hour
    df['dayofweek'] = df[TIMESTAMP_COL].dt.dayofweek
    df['month'] = df[TIMESTAMP_COL].dt.month
    df['year'] = df[TIMESTAMP_COL].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)  # 5=Sat, 6=Sun
    
    # Calculate deficit percent when applicable
    df['deficit_percent'] = 0.0  # Default to 0
    mask = df['energy_required_mu'] > 0
    df.loc[mask, 'deficit_percent'] = 100 * (df.loc[mask, 'energy_diff_mu'] / df.loc[mask, 'energy_required_mu'])
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed demand data to {output_file}")
    
    print(f"Processed demand data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage when run as script
    process_demand_data("data/raw/power_supply_data.csv", "data/processed/demand.parquet")