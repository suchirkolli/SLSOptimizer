# run_processor.py
import pandas as pd
import os
import yaml
from pathlib import Path
import sys

# Make sure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_demand_data(input_file, output_file=None):
    """Process historical electricity demand data"""
    print(f"Processing demand data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Raw demand data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize timestamp column
    df['timestamp'] = pd.to_datetime(df['Date'])
    
    # Since no district/region information, we'll default to a single region
    df['district'] = 'ANDHRA_PRADESH'
    
    # Rename demand columns
    df = df.rename(columns={
        'Energy Required (MU)': 'energy_required_mu',
        'Energy Met (MU)': 'energy_met_mu',
        'Energy +/- (MU)': 'energy_diff_mu',
        'Unrestricted Peak Demand (MW)': 'peak_demand_mw',
        'Deficit/Surplus (MW)': 'deficit_surplus_mw'
    })
    
    # Convert MU to MW
    df['avg_demand_mw'] = df['energy_required_mu'] * 1000 / 24
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed demand data to {output_file}")
    
    return df

def process_weather_data(input_file, output_file=None):
    """Process weather data"""
    print(f"Processing weather data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    print(f"Raw weather data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop the unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)
    
    # Standardize timestamp column
    df['timestamp'] = pd.to_datetime(df['Date'])
    
    # Default region
    df['district'] = 'ANDHRA_PRADESH'
    
    # Rename weather columns
    df = df.rename(columns={
        'temp': 'temperature_c',
        'rain (cm)': 'precipitation_cm'
    })
    
    # Convert precipitation to mm for consistency (1cm = 10mm)
    df['precipitation_mm'] = df['precipitation_cm'] * 10
    
    # Add derived weather features
    df['is_hot_day'] = (df['temperature_c'] > 30).astype(int)
    df['is_rainy_day'] = (df['precipitation_mm'] > 5).astype(int)
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed weather data to {output_file}")
    
    return df

def process_outage_data(input_file, output_file=None):
    """Process outage data"""
    print(f"Processing outage data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    print(f"Raw outage data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize timestamp columns
    df['timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])
    
    # Calculate outage duration in hours
    df['outage_duration_hours'] = (df['end_timestamp'] - df['timestamp']).dt.total_seconds() / 3600
    
    # Add severity classification
    df['severity'] = pd.cut(
        df['outage_duration_hours'], 
        bins=[0, 1, 4, 12, float('inf')], 
        labels=['minor', 'moderate', 'major', 'critical'],
        include_lowest=True
    )
    
    # Create normalized impact score
    if df['capacity_affected_mw'].max() > 0:
        df['capacity_impact'] = df['capacity_affected_mw'] / df['capacity_affected_mw'].max()
    else:
        df['capacity_impact'] = 0
        
    if df['customers_affected'].max() > 0:
        df['customer_impact'] = df['customers_affected'] / df['customers_affected'].max()
    else:
        df['customer_impact'] = 0
        
    df['impact_score'] = (df['capacity_impact'] + df['customer_impact']) / 2
    
    # Binary flags for analysis
    df['is_planned'] = (df['outage_type'] == 'Planned').astype(int)
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed outage data to {output_file}")
    
    return df

def process_grid_data(input_file, output_file=None):
    """Process grid infrastructure data"""
    print(f"Processing grid infrastructure data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    print(f"Raw grid infra data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Calculate utilization ratio
    df['capacity_mw'] = df['capacity_mva'] * 0.9
    df['utilization_ratio'] = df['peak_load_mw'] / df['capacity_mw']
    
    # Classify substations by utilization
    util_bins = [0, 0.5, 0.7, 0.9, float('inf')]
    util_labels = ['Low', 'Medium', 'High', 'Critical']
    df['utilization_class'] = pd.cut(
        df['utilization_ratio'], 
        bins=util_bins, 
        labels=util_labels,
        include_lowest=True
    )
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed grid infrastructure data to {output_file}")
    
    return df

def process_critical_data(input_file, output_file=None):
    """Process critical infrastructure data"""
    print(f"Processing critical infrastructure data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    print(f"Raw critical infrastructure data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize backup power to boolean flag
    df['has_backup'] = df['backup_power'].str.startswith('Yes').astype(int)
    
    # Create flag for full vs partial backup
    df['full_backup'] = (df['backup_power'] == 'Yes-Full').astype(int)
    
    # Convert kW to MW for consistency with other data
    df['avg_load_mw'] = df['avg_load_kw'] / 1000
    
    # Create facility type categories
    df['facility_category'] = df['type'].str.split(' - ').str[0]
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed critical infrastructure data to {output_file}")
    
    return df

def main():
    """Process all datasets"""
    # Default configuration
    config = {
        'data_files': {
            'demand': 'data/raw/power_supply_data.csv',
            'outages': 'data/raw/outage_data.csv',
            'weather': 'data/raw/weather_data.csv',
            'grid_infra': 'data/raw/grid_infra_data.csv',
            'critical_infra': 'data/raw/crit_infra_data.csv'
        },
        'processed_data_dir': 'data/processed/'
    }
    
    # Try to load config from file
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print("Loaded configuration from config.yaml")
    except (FileNotFoundError, yaml.YAMLError):
        print("Using default configuration")
    
    # Create output directory
    output_dir = Path(config.get('processed_data_dir', 'data/processed/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    print("\n" + "="*50)
    print("SMART LOAD SHEDDING DATA PROCESSING")
    print("="*50)
    
    # Process demand data
    demand_file = config['data_files']['demand']
    demand_output = os.path.join(output_dir, 'demand.parquet')
    process_demand_data(demand_file, demand_output)
    
    # Process weather data
    weather_file = config['data_files']['weather']
    weather_output = os.path.join(output_dir, 'weather.parquet')
    process_weather_data(weather_file, weather_output)
    
    # Process outage data
    outage_file = config['data_files']['outages']
    outage_output = os.path.join(output_dir, 'outages.parquet')
    process_outage_data(outage_file, outage_output)
    
    # Process grid infrastructure data
    grid_file = config['data_files']['grid_infra']
    grid_output = os.path.join(output_dir, 'grid_infra.parquet')
    process_grid_data(grid_file, grid_output)
    
    # Process critical infrastructure data
    critical_file = config['data_files']['critical_infra']
    critical_output = os.path.join(output_dir, 'critical_infra.parquet')
    process_critical_data(critical_file, critical_output)
    
    print("\n" + "="*50)
    print("DATA PROCESSING COMPLETE")
    print("="*50)
    print(f"Processed data saved to: {output_dir}")

if __name__ == "__main__":
    main()