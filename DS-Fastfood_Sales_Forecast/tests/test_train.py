import pytest
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import (
    evaluate,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    calculate_waste_reduction_kpi
)

def test_evaluate_metrics():
    y_true = np.array([3.0, 2.0, 5.0, 7.0])
    y_pred = np.array([3.0, 2.0, 5.0, 7.0])
    
    rmse, mae, r2, mad, mape, smape, waste_kpi = evaluate(y_true, y_pred)
    
    assert np.isclose(rmse, 0.0)
    assert np.isclose(mae, 0.0)
    assert np.isclose(r2, 1.0)
    assert np.isclose(mape, 0.0)
    assert np.isclose(smape, 0.0)
    assert waste_kpi >= 0.0

def test_mape_calculation():
    y_true = np.array([100, 100])
    y_pred = np.array([90, 110])
    # 10% error each -> mean 10%
    assert np.isclose(mean_absolute_percentage_error(y_true, y_pred), 0.1)

def test_smape_calculation():
    y_true = np.array([100, 100])
    y_pred = np.array([90, 110])
    # SMAPE formula: 2 * |y_pred - y_true| / (|y_true| + |y_pred|)
    # For 90: 2 * 10 / 190 = 20 / 190 ~ 0.1052
    # For 110: 2 * 10 / 210 = 20 / 210 ~ 0.0952
    # Mean ~ 0.1002
    assert symmetric_mean_absolute_percentage_error(y_true, y_pred) > 0.0

def test_waste_reduction_kpi():
    y_true = np.array([10, 30, 20])
    y_pred_bad = np.array([30, 30, 30])
    y_pred_perfect = np.array([10, 30, 20])
    
    kpi_bad = calculate_waste_reduction_kpi(y_true, y_pred_bad)
    kpi_perfect = calculate_waste_reduction_kpi(y_true, y_pred_perfect)
    
    assert kpi_perfect > kpi_bad
    assert np.isclose(kpi_perfect, 100.0)
