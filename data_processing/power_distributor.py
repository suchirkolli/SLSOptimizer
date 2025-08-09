import pandas as pd
import numpy as np
from pathlib import Path

def distribute_power_data(state_power_df: pd.DataFrame, district_ref_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize timestamp column
    if 'timestamp' not in state_power_df.columns:
        if 'Date' in state_power_df.columns:
            state_power_df = state_power_df.rename(columns={'Date': 'timestamp'})
        else:
            raise KeyError("state_power_df must contain 'timestamp' or 'Date' column")

    state_power_df['timestamp'] = pd.to_datetime(state_power_df['timestamp'], errors='coerce')
    if state_power_df['timestamp'].isnull().any():
        print("Warning: some timestamps could not be parsed and will be dropped")
        state_power_df = state_power_df.dropna(subset=['timestamp'])

    # Ensure district proportions exist and are numeric
    if 'pop_proportion' not in district_ref_df.columns:
        district_ref_df['pop_proportion'] = pd.to_numeric(district_ref_df['population'], errors='coerce') / pd.to_numeric(district_ref_df['population'], errors='coerce').sum()
    else:
        district_ref_df['pop_proportion'] = pd.to_numeric(district_ref_df['pop_proportion'], errors='coerce')
        # if NaNs, recompute from population
        if district_ref_df['pop_proportion'].isna().any():
            district_ref_df['pop_proportion'] = pd.to_numeric(district_ref_df['population'], errors='coerce') / pd.to_numeric(district_ref_df['population'], errors='coerce').sum()

    # Select numeric metric columns to distribute (exclude timestamp and any id/text columns)
    numeric_cols = state_power_df.select_dtypes(include=[np.number]).columns.tolist()
    # If some numeric metrics are stored as strings, coerce them:
    for c in state_power_df.columns:
        if c not in numeric_cols and c != 'timestamp':
            coerced = pd.to_numeric(state_power_df[c], errors='coerce')
            if coerced.notna().any():
                state_power_df[c] = coerced
                numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("No numeric columns found in state_power_df to distribute")

    records = []
    for _, state_row in state_power_df.iterrows():
        ts = state_row['timestamp']
        for _, d in district_ref_df.iterrows():
            proportion = float(d['pop_proportion']) if not pd.isna(d['pop_proportion']) else 0.0
            rec = {
                'timestamp': ts,
                'district_id': d['district_id']
            }
            for metric in numeric_cols:
                val = state_row.get(metric, np.nan)
                if pd.isna(val):
                    rec[metric] = np.nan
                else:
                    # ensure numeric before multiply
                    try:
                        rec[metric] = float(val) * proportion
                    except Exception:
                        rec[metric] = np.nan
            records.append(rec)

    district_power_df = pd.DataFrame.from_records(records)
    return district_power_df


def main():
    processed_dir = Path('data/processed')
    district_ref_path = Path('data/geographic/ap_districts_reference.csv')
    if not district_ref_path.exists():
        raise FileNotFoundError(district_ref_path)
    district_ref = pd.read_csv(district_ref_path)

    # load state-level power data (ensure it's the state-level file)
    state_power_path = processed_dir / 'demand.parquet'
    if not state_power_path.exists():
        raise FileNotFoundError(state_power_path)
    state_power = pd.read_parquet(state_power_path)

    # Distribute and save
    district_power = distribute_power_data(state_power, district_ref)
    out_path = processed_dir / 'district_demand.parquet'
    district_power.to_parquet(out_path, index=False)
    print(f"Saved distributed district demand to {out_path}")

if __name__ == "__main__":
    main()