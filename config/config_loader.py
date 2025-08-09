"""
Configuration loading utilities for the Smart Load Shedding Optimizer.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to the configuration file. If None, looks for config.yaml
        in the config directory.
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Use default config location
        config_path = Path(__file__).parent / "config.yaml"
        
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def default_config():
    """
    Return default configuration when config file is not available.
    
    Returns:
    --------
    dict
        Default configuration dictionary
    """
    return {
        "api": {
            "openweathermap": {
                "api_key": "YOUR_API_KEY_HERE",
                "location": "Andhra Pradesh, India",
                "lat": 15.9129,
                "lon": 79.7400
            }
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "geographic_dir": "data/geographic"
        },
        "time_range": {
            "start_date": "2020-05-14",
            "end_date": "2023-05-14"
        },
        "load_shedding": {
            "max_percentage_per_district": 30.0,
            "critical_infrastructure_weight": 2.0
        }
    }