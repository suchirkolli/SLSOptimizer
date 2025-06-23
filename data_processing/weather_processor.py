"""
Process weather data from weather_data.csv
"""

import pandas as pd
import os
from .schemas import standardize_timestamp, validate_date_range, check_missing_values, TIMESTAMP_COL, REGION_ID_COL

def process_weather_data(input_file: str, output_file: str = None):
    """
    Process weather data from weather_data.csv
    
    Args:
        input_file: Path to weather_data.csv
        output_file: Path to save processed output (default: None)
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing weather data from {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Display initial info
    print(f"Raw weather data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Based on your file, columns include:
    # an index, 'Date', 'temp', 'rain (cm)'
    
    # Drop the unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)
    
    # Standardize timestamp column
    df = standardize_timestamp(df, 'Date')
    
    # Since no district/region information, we'll default to a single region
    df[REGION_ID_COL] = 'ANDHRA_PRADESH'  # Default to AP based on other data
    
    # Rename weather columns
    df = df.rename(columns={
        'temp': 'temperature_c',
        'rain (cm)': 'precipitation_cm'
    })
    
    # Convert precipitation to mm for consistency (1cm = 10mm)
    df['precipitation_mm'] = df['precipitation_cm'] * 10
    
    # Filter to study period (May 2020-May 2023)
    df = validate_date_range(df)
    
    # Check for missing values
    check_missing_values(df, "Weather data")
    
    # Add derived weather features
    # Heat index (using a simplified version - actual heat index uses humidity too)
    df['is_hot_day'] = (df['temperature_c'] > 30).astype(int)
    df['is_rainy_day'] = (df['precipitation_mm'] > 5).astype(int)  # >5mm is considered rainy
    
    # Add time-based features
    df['month'] = df[TIMESTAMP_COL].dt.month
    df['year'] = df[TIMESTAMP_COL].dt.year
    
    # Define seasons for India
    # Winter: Dec-Feb, Summer: Mar-May, Monsoon: Jun-Sep, Post-Monsoon: Oct-Nov
    season_map = {
        1: 'Winter', 2: 'Winter', 
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon',
        12: 'Winter'
    }
    df['season'] = df['month'].map(season_map)
    
    # Save processed data
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(output_file)
        print(f"Saved processed weather data to {output_file}")
    
    print(f"Processed weather data shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Example usage when run as script
    process_weather_data("data/raw/weather_data.csv", "data/processed/weather.parquet")