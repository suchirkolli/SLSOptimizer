"""
Defines common schemas and data structures for the Smart Load Shedding project.
May 2020-May 2023 data focus.
"""

import pandas as pd
from datetime import datetime

# Define standard column names across all datasets
TIMESTAMP_COL = 'timestamp'
REGION_ID_COL = 'district'  # Using district as the common region identifier

# Function to standardize timestamp format across all data
def standardize_timestamp(df, orig_timestamp_col):
    """
    Convert timestamp column to standard Pandas datetime format
    
    Args:
        df: DataFrame containing timestamp data
        orig_timestamp_col: Name of the column containing timestamp data
        
    Returns:
        DataFrame with standardized timestamp column
    """
    df[TIMESTAMP_COL] = pd.to_datetime(df[orig_timestamp_col])
    return df

# Function to validate date range (May 2020-May 2023)
def validate_date_range(df):
    """
    Filter DataFrame to only include dates within study period (May 2020-May 2023)
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        Filtered DataFrame
    """
    start_date = pd.Timestamp('2020-05-14')
    end_date = pd.Timestamp('2023-05-14')
    
    # Filter to our study period
    filtered_df = df[(df[TIMESTAMP_COL] >= start_date) & 
                     (df[TIMESTAMP_COL] <= end_date)]
    
    # Log any data loss
    if len(filtered_df) < len(df):
        print(f"Filtered out {len(df) - len(filtered_df)} rows outside May 2020-May 2023")
    
    return filtered_df

# Function to check and report on missing values
def check_missing_values(df, name="Dataset"):
    """Report on missing values in the DataFrame"""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n{name} - Missing values:")
        print(missing[missing > 0])
    else:
        print(f"\n{name} - No missing values found")