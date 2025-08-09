# tests/test_predictor.py
import pytest
from pathlib import Path
import pandas as pd

from services import predictor

MODEL_DIR = Path("models/saved")
MODELING_PATH = Path("data/processed/modeling_dataset.parquet")

def test_load_demand_models_returns_dict():
    models = predictor.load_demand_models(str(MODEL_DIR))
    assert isinstance(models, dict)

@pytest.mark.skipif(not MODELING_PATH.exists(), reason="modeling dataset missing")
def test_predict_snapshot_all_shape():
    # If no saved demand models, this will still run but may return fallbacks.
    forecasts, risks = predictor.predict_snapshot_all(model_dir=str(MODEL_DIR), modeling_path=str(MODELING_PATH))
    assert isinstance(forecasts, pd.DataFrame)
    assert isinstance(risks, pd.DataFrame)
    # Both should have district_id column
    assert 'district_id' in forecasts.columns
    assert 'district_id' in risks.columns
    # lengths should match snapshot districts
    snap = predictor.get_latest_snapshot(str(MODELING_PATH))
    assert forecasts['district_id'].nunique() == snap['district_id'].nunique()
    assert risks['district_id'].nunique() == snap['district_id'].nunique()