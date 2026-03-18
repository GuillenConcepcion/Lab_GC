import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import create_temporal_features, create_lag_features

def test_create_temporal_features():
    dates = pd.date_range(start="2023-01-01", end="2023-01-05")
    df = pd.DataFrame({'quantity': [10, 20, 30, 40, 50]}, index=dates)
    
    df_feat = create_temporal_features(df)
    
    assert 'day_of_week' in df_feat.columns
    assert 'weekend' in df_feat.columns
    assert df_feat.loc['2023-01-01', 'weekend'] == 1 # Sunday
    assert df_feat.loc['2023-01-02', 'weekend'] == 0 # Monday
    
def test_create_lag_features():
    dates = pd.date_range(start="2023-01-01", periods=4)
    df = pd.DataFrame({
        'item_name': ['A', 'A', 'A', 'A'],
        'quantity': [10, 20, 30, 40]
    }, index=dates)
    df.index.name = 'date'
    
    df_feat = create_lag_features(df, target_col='quantity', lags=[1, 2])
    
    assert 'quantity_lag_1' in df_feat.columns
    assert 'quantity_lag_2' in df_feat.columns
    assert pd.isna(df_feat['quantity_lag_1'].iloc[0])
    assert df_feat['quantity_lag_1'].iloc[1] == 10.0
    assert df_feat['quantity_lag_2'].iloc[2] == 10.0
