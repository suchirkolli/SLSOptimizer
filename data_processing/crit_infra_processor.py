"""
Process critical infrastructure data from crit_infra_data.csv
"""

import pandas as pd
import os
from .schemas import check_missing_values, REGION_ID_COL

def process_critical_data(input_file: str, output_file: str = None):
    """
    Process critical infrastructure data from crit_infra_data.csv
    
    Args:
        input_file: Path to crit_infra_data.csv
        output_file: Path to save processed output (default: None)
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing critical infrastructure data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Display initial info
    print(f"Raw critical infrastructure data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Based on your file, columns include:
    # 'facility_id', 'name', 'type', 'district', 'location', 'latitude', 
    # 'longitude', 'priority_level', 'backup_power', 'avg_load_kw'
    
    # Check for missing values
    check_missing_values(df, "Critical infrastructure data")
    
    # Standardize backup power to boolean flag
    df['has_backup'] = df['backup_power'].str.startswith('Yes').astype(int)
    
    # Create flag for full vs partial backup
    df['full_backup'] = (df['backup_power'] == 'Yes-Full').astype(int)
    
    # Convert kW to MW for consistency with other data
    df['avg_load_mw'] = df['avg_load_kw'] / 1000
    
    # Create facility type categories
    df['facility_category'] = df['type'].str.split(' - ').str[0]
    
    # Classify facilities by criticality (combine priority level and backup)
    # Lower priority_level is more critical (1 is highest priority)
    # Facilities without backup are more vulnerable
    df['criticality_score'] = df['priority_level'] - df['has_backup'] * 0.5
    
    # Create district-level aggregates
    district_stats = df.groupby(REGION_ID_COL).agg(
        critical_facility_count=('facility_id', 'count'),
        hospital_count=('facility_category', lambda x: sum(x == 'Hospital')),
        emergency_count=('facility_category', lambda x: sum(x == 'Emergency')),
        priority1_count=('priority_level', lambda x: sum(x == 1)),
        no_backup_count=('has_backup', lambda x: sum(x == 0)),
        total_critical_load_mw=('avg_load_mw', 'sum')
    ).reset_index()
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed critical infrastructure data to {output_file}")
        
        # Also save district-level aggregates
        district_output = output_file.replace('.parquet', '_district.parquet')
        district_stats.to_parquet(district_output)
        print(f"Saved district-level critical infrastructure stats to {district_output}")
    
    print(f"Processed critical infrastructure data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage when run as script
    process_critical_data("data/raw/crit_infra_data.csv", "data/processed/critical_infra.parquet")