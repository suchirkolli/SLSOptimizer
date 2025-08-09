"""
Validate all data for the Smart Load Shedding Optimizer.
Ensures all required datasets are ready for modeling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def validate_datasets():
    """Check all required datasets are available and properly formatted."""
    required_files = {
        'district_reference': 'data/geographic/ap_districts_reference.csv',
        'district_demand': 'data/processed/district_demand.parquet',
        'district_weather': 'data/processed/district_weather.parquet',
        'outages': 'data/processed/outages.parquet',
        'grid_infra': 'data/processed/grid_infra.parquet',
        'critical_infra': 'data/processed/critical_infra.parquet'
    }
    
    datasets = {}
    issues = []
    
    # Check each file
    for name, path in required_files.items():
        file_path = Path(path)
        
        if not file_path.exists():
            issues.append(f"Missing file: {path}")
            datasets[name] = None
            continue
        
        # Load file
        if path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            issues.append(f"Unknown file format: {path}")
            datasets[name] = None
            continue
        
        datasets[name] = df
        print(f"Loaded {name}: {len(df)} rows, {df.shape[1]} columns")
        
        # Check for empty dataframe
        if df.empty:
            issues.append(f"Empty dataset: {name}")
    
    # Validate district references in all datasets
    if all(df is not None for df in [datasets['district_reference'], 
                                     datasets['district_demand'], 
                                     datasets['district_weather']]):
        district_ids = set(datasets['district_reference']['district_id'])
        
        demand_districts = set(datasets['district_demand']['district_id'])
        if not demand_districts.issubset(district_ids):
            issues.append("Demand data contains district_ids not in reference")
        
        weather_districts = set(datasets['district_weather']['district_id'])
        if not weather_districts.issubset(district_ids):
            issues.append("Weather data contains district_ids not in reference")
    
    return datasets, issues

def create_merged_dataset(datasets):
    """Create merged dataset for modeling."""
    # Check required datasets
    required = ['district_demand', 'district_weather']
    if any(datasets[name] is None for name in required):
        print("Cannot create merged dataset - missing required data")
        return None
    
    # Merge demand and weather
    merged = pd.merge(
        datasets['district_demand'],
        datasets['district_weather'],
        on=['timestamp', 'district_id'],
        how='inner'
    )
    
    # Add district reference info if available
    if datasets['district_reference'] is not None:
        district_cols = ['district_id', 'district_name', 'pop_density']
        district_cols = [col for col in district_cols 
                       if col in datasets['district_reference'].columns]
        
        merged = pd.merge(
            merged,
            datasets['district_reference'][district_cols],
            on='district_id',
            how='left'
        )
    
    print(f"Created merged dataset with {len(merged)} rows")
    return merged

def plot_data_summary(merged_df):
    """Plot summary visualizations of the merged dataset."""
    # Set style
    sns.set(style='whitegrid')
    
    # Plot 1: Demand by district
    plt.figure(figsize=(12, 6))
    district_avg = merged_df.groupby('district_id')['peak_demand_mw'].mean().sort_values(ascending=False)
    sns.barplot(x=district_avg.index, y=district_avg.values)
    plt.title('Average Demand by District')
    plt.xlabel('District')
    plt.ylabel('Average Demand (MW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/avg_demand_by_district.png')
    
    # Plot 2: Demand vs. temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df.sample(1000), x='temperature_c', y='peak_demand_mw', alpha=0.6)
    plt.title('Demand vs. Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Demand (MW)')
    plt.tight_layout()
    plt.savefig('outputs/demand_vs_temperature.png')
    
    # Plot 3: Time series of demand
    plt.figure(figsize=(14, 6))
    daily_demand = merged_df.groupby(merged_df['timestamp'].dt.date)['peak_demand_mw'].sum()
    daily_demand.plot()
    plt.title('Daily Total Demand (All Districts)')
    plt.xlabel('Date')
    plt.ylabel('Total Demand (MW)')
    plt.tight_layout()
    plt.savefig('outputs/daily_demand_timeseries.png')

def main():
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("Validating datasets for Smart Load Shedding Optimizer...")
    datasets, issues = validate_datasets()
    
    # Report any issues
    if issues:
        print("\nData issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nAll datasets validated successfully!")
    
    # Create merged dataset
    merged_df = create_merged_dataset(datasets)
    
    if merged_df is not None:
        # Save for modeling
        merged_path = 'data/processed/modeling_dataset.parquet'
        merged_df.to_parquet(merged_path)
        print(f"Saved merged modeling dataset to {merged_path}")
        
        # Create summary plots
        plot_data_summary(merged_df)
        print("Generated summary visualizations in outputs/ directory")
    
    return datasets, merged_df

if __name__ == "__main__":
    main()