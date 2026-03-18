import pandas as pd
import numpy as np
import holidays
import os
import argparse
import yaml
from src.logger import get_logger

logger = get_logger(__name__)

def create_temporal_features(df):
    logger.info("Creating temporal features...")
    if not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)
            
    df['day_of_week'] = df.index.dayofweek
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    return df

def create_lag_features(df, target_col='quantity', lags=[1, 7]):
    logger.info(f"Creating lag features for target '{target_col}'...")
    df = df.sort_values(by=['item_name', 'date'])
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('item_name')[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='quantity', windows=[7, 30]):
    logger.info(f"Creating rolling average features for target '{target_col}'...")
    for w in windows:
        df[f'{target_col}_roll_mean_{w}'] = df.groupby('item_name')[target_col].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
    return df

def add_holidays(df, country='US'):
    logger.info(f"Adding {country} holidays...")
    years = df.index.year.unique()
    holiday_dict = holidays.country_holidays(country, years=years)
    df['is_holiday'] = df.index.to_series().apply(lambda x: 1 if x in holiday_dict else 0)
    return df

def encode_categorical(df):
    logger.info("Encoding categorical variables...")
    if 'item_type' in df.columns:
        df = pd.get_dummies(df, columns=['item_type'], drop_first=True)
    return df

def apply_log_transform(df, cols):
    logger.info(f"Applying log transformation to columns: {cols}")
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config['project']['seed'])
    input_path = config['data']['processed_daily']
    output_path = config['data']['features']
    target = config['features']['target']
    lags = config['features']['lags']
    roll_windows = config['features']['roll_windows']
    country = config['features']['country']
    log_cols = config.get('preprocessing', {}).get('log_transform_cols', [])
    
    if not os.path.exists(input_path):
        logger.error(f"{input_path} not found. Run ETL pipeline first.")
        return
        
    df = pd.read_csv(input_path, index_col='date', parse_dates=['date'])
    
    df = create_temporal_features(df)
    df = create_lag_features(df, target_col=target, lags=lags)
    df = create_rolling_features(df, target_col=target, windows=roll_windows)
    df = add_holidays(df, country=country)
    df = encode_categorical(df)
    
    if log_cols:
        df = apply_log_transform(df, log_cols)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Feature engineering completed! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
