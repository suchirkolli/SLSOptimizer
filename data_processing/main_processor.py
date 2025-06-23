"""
Main data processor for Smart Load Shedding Optimizer.
Orchestrates processing of all data sources.
"""

import argparse
import yaml
import os
from pathlib import Path

from .demand_processor import process_demand_data
from .outage_processor import process_outage_data
from .weather_processor import process_weather_data
from .grid_infra_processor import process_grid_data
from .crit_infra_processor import process_critical_data

def main():
    """Main entry point for data processing pipeline"""
    parser = argparse.ArgumentParser(description='Process Smart Load Shedding data')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        # Create a default configuration
        config = {
            'data_files': {
                'demand': 'data/raw/power_supply_data.csv',
                'outages': 'data/raw/outage_data.csv',
                'weather': 'data/raw/weather_data.csv',
                'grid_infra': 'data/raw/grid_infra_data.csv',
                'critical_infra': 'data/raw/crit_infra_data.csv'
            },
            'processed_data_dir': 'data/processed/',
            'output_dir': 'outputs/'
        }
        print("Using default configuration")
    
    # Create output directories
    output_dir = Path(config.get('processed_data_dir', 'data/processed/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_output_dir = Path(config.get('output_dir', 'outputs/'))
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    print("\n" + "="*50)
    print("SMART LOAD SHEDDING DATA PROCESSING PIPELINE")
    print("="*50)
    
    try:
        # Process demand data
        print("\n1. PROCESSING DEMAND DATA")
        demand_file = config['data_files']['demand']
        demand_output = os.path.join(output_dir, 'demand.parquet')
        demand_df = process_demand_data(demand_file, demand_output)
    except Exception as e:
        print(f"Error processing demand data: {e}")
        demand_df = None
    
    try:
        # Process outage data
        print("\n2. PROCESSING OUTAGE DATA")
        outage_file = config['data_files']['outages']
        outage_output = os.path.join(output_dir, 'outages.parquet')
        outage_df = process_outage_data(outage_file, outage_output)
    except Exception as e:
        print(f"Error processing outage data: {e}")
        outage_df = None
    
    try:
        # Process weather data
        print("\n3. PROCESSING WEATHER DATA")
        weather_file = config['data_files']['weather']
        weather_output = os.path.join(output_dir, 'weather.parquet')
        weather_df = process_weather_data(weather_file, weather_output)
    except Exception as e:
        print(f"Error processing weather data: {e}")
        weather_df = None
    
    try:
        # Process grid infrastructure data
        print("\n4. PROCESSING GRID INFRASTRUCTURE DATA")
        grid_file = config['data_files']['grid_infra']
        grid_output = os.path.join(output_dir, 'grid_infra.parquet')
        grid_df = process_grid_data(grid_file, grid_output)
    except Exception as e:
        print(f"Error processing grid infrastructure data: {e}")
        grid_df = None
    
    try:
        # Process critical infrastructure data
        print("\n5. PROCESSING CRITICAL INFRASTRUCTURE DATA")
        critical_file = config['data_files']['critical_infra']
        critical_output = os.path.join(output_dir, 'critical_infra.parquet')
        critical_df = process_critical_data(critical_file, critical_output)
    except Exception as e:
        print(f"Error processing critical infrastructure data: {e}")
        critical_df = None
    
    print("\n" + "="*50)
    print("DATA PROCESSING COMPLETE")
    print("="*50)
    print(f"Processed data saved to: {output_dir}")
    print(f"Output visualizations will be saved to: {vis_output_dir}")

if __name__ == "__main__":
    main()