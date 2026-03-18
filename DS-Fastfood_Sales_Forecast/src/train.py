import pandas as pd
import numpy as np
import os
import argparse
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1)))

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def calculate_waste_reduction_kpi(y_true, y_pred):
    """
    Business KPI: Waste Reduction %.
    """
    naive_pred = np.full_like(y_true, np.mean(y_true))
    naive_waste = np.sum(np.maximum(0, naive_pred - y_true))
    model_waste = np.sum(np.maximum(0, y_pred - y_true))
    
    if naive_waste == 0:
        return 0.0
    return max(0, (naive_waste - model_waste) / naive_waste * 100)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mad = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    waste_kpi = calculate_waste_reduction_kpi(y_true, y_pred)
    return rmse, mae, r2, mad, mape, smape, waste_kpi

def train_models(df, config):
    logger.info("Preparing data for multi-model competition...")
    df = df.sort_values(by='date')
    
    if 'item_name' in df.columns:
        df = pd.get_dummies(df, columns=['item_name'], drop_first=True)
    
    target = config['features']['target']
    log_target = config.get('preprocessing', {}).get('log_target', False)
    
    features = [col for col in df.columns if col not in [target, 'transaction_amount']]
    
    X = df[features].fillna(0)
    y = df[target]
    
    if log_target:
        logger.info("Applying log transformation to target variable.")
        y = np.log1p(y)
    
    dates = df.index.unique()
    test_ratio = config['training'].get('test_size_ratio', 0.2)
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]
    
    logger.info(f"Splitting data with split date: {split_date}")
    
    train_mask = df.index < split_date
    test_mask = df.index >= split_date
    
    X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
    X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
    
    seed = config['project']['seed']
    
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed, objective='reg:squarederror'),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=seed, verbose=0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    tscv = TimeSeriesSplit(n_splits=config['training'].get('cv_splits', 5))
    
    metrics_list = []
    
    # We will compute results on original scale
    y_test_orig = np.expm1(y_test) if log_target else y_test
    
    predictions_df = pd.DataFrame(index=y_test_orig.index)
    predictions_df['Actual'] = y_test_orig
    
    best_model_name = None
    best_rmse = float('inf')
    best_model_obj = None

    logger.info("--- Starting Model Training and Evaluation ---")
    
    for name, model in models.items():
        logger.info(f"Training and validating {name} using TimeSeriesSplit...")
        
        cv_rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            cv_X_train, cv_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            cv_y_train, cv_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(cv_X_train, cv_y_train)
            cv_preds = model.predict(cv_X_val)
            
            # Inverse transform for CV evaluation
            if log_target:
                cv_preds_orig = np.expm1(cv_preds)
                cv_y_val_orig = np.expm1(cv_y_val)
            else:
                cv_preds_orig = cv_preds
                cv_y_val_orig = cv_y_val
                
            cv_rmses.append(np.sqrt(mean_squared_error(cv_y_val_orig, cv_preds_orig)))
            
        avg_cv_rmse = np.mean(cv_rmses)
        logger.info(f"{name} CV RMSE: {avg_cv_rmse:.4f}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Inverse transform for final evaluation
        y_pred_orig = np.expm1(y_pred) if log_target else y_pred
        
        rmse, mae, r2, mad, mape, smape, waste_kpi = evaluate(y_test_orig, y_pred_orig)
        
        metrics_list.append({
            'Model': name,
            'CV_RMSE': avg_cv_rmse,
            'Test_RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'Waste_Reduction_%': waste_kpi
        })
        
        predictions_df[f'Predicted_{name}'] = y_pred_orig
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model
            
    metrics_df = pd.DataFrame(metrics_list).sort_values(by='Test_RMSE', ascending=True)
    logger.info(f"\nModel Comparison Results (Original Scale):\n{metrics_df.to_string(index=False)}")
    
    models_dir = config['data']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    best_model_name_clean = best_model_name.replace(' ', '_')
    best_model_path = os.path.join(models_dir, f"best_model_{best_model_name_clean}.pkl")
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Saved Best Model ({best_model_name}) to {best_model_path}")
    
    metrics_path = config['data']['metrics']
    predictions_path = config['data']['predictions']
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    predictions_df.to_csv(predictions_path)
    
    logger.info("Metrics and Predictions saved successfully.")
    return best_model_obj

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config['project']['seed'])
    input_path = config['data']['features']
    
    if not os.path.exists(input_path):
        logger.error(f"{input_path} not found. Run feature engineering first.")
        return
        
    df = pd.read_csv(input_path, index_col='date', parse_dates=['date'])
    train_models(df, config)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
