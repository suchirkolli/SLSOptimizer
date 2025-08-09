"""
Demand forecasting model for the Smart Load Shedding Optimizer.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_features(df):
    """Create time-based features for demand forecasting."""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Create district-specific lags and rolling statistics
    districts = df['district_id'].unique()
    for district in districts:
        district_mask = df['district_id'] == district
        
        # Sort by time for proper lags
        district_data = df.loc[district_mask].sort_values('timestamp')
        
        # Calculate lags (24h, 7d)
        df.loc[district_mask, 'demand_lag_24h'] = district_data['peak_demand_mw'].shift(24)
        df.loc[district_mask, 'demand_lag_7d'] = district_data['peak_demand_mw'].shift(24*7)
        
        # Calculate rolling averages
        df.loc[district_mask, 'demand_rolling_24h'] = district_data['peak_demand_mw'].rolling(24, min_periods=1).mean()
        df.loc[district_mask, 'demand_rolling_7d'] = district_data['peak_demand_mw'].rolling(24*7, min_periods=1).mean()
    
    return df

def train_model(df, test_start_date='2023-03-01'):
    """Train XGBoost model for electricity demand forecasting."""
    print("Training demand forecasting model...")
    
    # Create features
    print("Creating features...")
    df = create_features(df)
    
    # Define features to use
    feature_cols = [
        'hour', 'dayofweek', 'month', 'is_weekend',
        'temperature', 'humidity',
        'demand_lag_24h', 'demand_lag_7d', 
        'demand_rolling_24h', 'demand_rolling_7d'
    ]
    
    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using features: {available_features}")
    
    # Fill NAs (from lag features)
    for col in available_features:
        if df[col].isna().sum() > 0:
            print(f"Filling {df[col].isna().sum()} NaN values in {col}")
            df[col] = df[col].fillna(df[col].mean())
    
    # Split into train/test by time
    train_df = df[df['timestamp'] < test_start_date]
    test_df = df[df['timestamp'] >= test_start_date]
    
    print(f"Training data: {len(train_df)} records")
    print(f"Testing data: {len(test_df)} records")
    
    # Create district-specific models
    districts = df['district_id'].unique()
    district_models = {}
    district_metrics = {}
    
    for district in districts:
        print(f"\nTraining model for district: {district}")
        
        # Filter data for this district
        district_train = train_df[train_df['district_id'] == district]
        district_test = test_df[test_df['district_id'] == district]
        
        if len(district_train) < 100:
            print(f"  Insufficient training data for district {district}, skipping.")
            continue
            
        if len(district_test) < 10:
            print(f"  Insufficient test data for district {district}, skipping.")
            continue
        
        # Prepare train/test sets
        X_train = district_train[available_features]
        y_train = district_train['peak_demand_mw']
        X_test = district_test[available_features]
        y_test = district_test['peak_demand_mw']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Store model and metrics
        district_models[district] = model
        district_metrics[district] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': dict(zip(available_features, model.feature_importances_)),
            'test_actuals': y_test.values,
            'test_predictions': y_pred
        }
    
    # Calculate overall metrics
    overall_metrics = {
        'avg_r2': np.mean([m['r2'] for m in district_metrics.values()]),
        'avg_mae': np.mean([m['mae'] for m in district_metrics.values()]),
        'avg_rmse': np.mean([m['rmse'] for m in district_metrics.values()]),
    }
    
    print(f"\nAverage performance across all districts:")
    print(f"  Avg R²: {overall_metrics['avg_r2']:.4f}")
    print(f"  Avg MAE: {overall_metrics['avg_mae']:.2f}")
    print(f"  Avg RMSE: {overall_metrics['avg_rmse']:.2f}")
    
    return district_models, district_metrics, overall_metrics

def predict_demand(models, df, features=None):
    """Make demand predictions using trained models."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create features if needed
    if features is None:
        result_df = create_features(result_df)
    
    # Make predictions for each district
    for district_id, model in models.items():
        # Filter data for this district
        district_mask = result_df['district_id'] == district_id
        
        if district_mask.sum() == 0:
            continue
            
        # Get feature columns from model
        feature_cols = model.feature_names_in_
        
        # Check that all features exist
        missing_features = [col for col in feature_cols if col not in result_df.columns]
        if missing_features:
            print(f"Warning: Missing features for district {district_id}: {missing_features}")
            continue
        
        # Make predictions
        X = result_df.loc[district_mask, feature_cols]
        result_df.loc[district_mask, 'predicted_demand_mw'] = model.predict(X)
    
    return result_df

def plot_forecast_vs_actual(district_id, metrics, output_dir='outputs'):
    """Plot forecast vs actual for a specific district."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    if district_id not in metrics:
        print(f"No metrics available for district {district_id}")
        return
    
    district_metrics = metrics[district_id]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(district_metrics['test_actuals'], label='Actual Demand', alpha=0.7)
    plt.plot(district_metrics['test_predictions'], label='Predicted Demand', alpha=0.7)
    plt.title(f'Forecast vs Actual Demand - District {district_id}')
    plt.xlabel('Test Data Points')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = f"{output_dir}/forecast_vs_actual_{district_id}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved comparison plot to {output_file}")

def main():
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Load modeling dataset
    df = pd.read_parquet('data/processed/modeling_dataset.parquet')
    
    # Train models
    models, metrics, overall = train_model(df)
    
    # Save models
    import joblib
    models_dir = Path('models/saved')
    models_dir.mkdir(exist_ok=True, parents=True)
    
    for district_id, model in models.items():
        joblib.dump(model, models_dir / f'demand_model_{district_id}.joblib')
    
    # Plot results for each district
    for district_id in metrics.keys():
        plot_forecast_vs_actual(district_id, metrics)
    
    print("\nDemand forecasting models trained and saved.")

if __name__ == "__main__":
    main()