# tests/test_optimizer.py
import pandas as pd
import numpy as np
from optimizer.load_shedding_optimizer import calculate_impact_scores, allocate_load_shedding

def make_sample_df():
    return pd.DataFrame({
        'district_id': ['D1','D2','D3'],
        'peak_demand_mw': [100.0, 200.0, 300.0],
        'pop_density': [50.0, 200.0, 100.0],
        'outage_risk': [0.1, 0.9, 0.2],
        'district_name': ['D1','D2','D3']
    })

def test_allocate_invariants():
    df = make_sample_df()
    scored = calculate_impact_scores(df, critical_infra=None)
    target = 150.0
    alloc_df, summary = allocate_load_shedding(scored, target_reduction_mw=target, max_pct_per_district=0.3)
    # sum of shed_mw <= target (allow tiny float rounding)
    assert alloc_df['shed_mw'].sum() <= target + 1e-6
    # no negative allocations
    assert (alloc_df['shed_mw'] >= -1e-9).all()
    # each allocation <= 30% of demand
    for _, r in alloc_df.iterrows():
        assert r['shed_mw'] <= 0.30001 * r['peak_demand_mw'] + 1e-9