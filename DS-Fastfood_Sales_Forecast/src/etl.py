import pandas as pd
import numpy as np
import os
import argparse
import yaml
from src.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    logger.info("Cleaning data...")
    df = df.drop_duplicates().copy()
    
    for col in ['item_price', 'transaction_amount']:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
    df = df.drop_duplicates().copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    df = df[df['quantity'] > 0]
    q_high = df['quantity'].quantile(0.99)
    df = df[df['quantity'] <= q_high]
    return df

def aggregate_daily_sales(df):
    logger.info("Aggregating to daily sales by item...")
    daily_sales = df.groupby(['date', 'item_name']).agg(
        quantity=('quantity', 'sum'),
        transaction_amount=('transaction_amount', 'sum'),
        item_type=('item_type', 'first')
    ).reset_index()
    
    daily_sales = daily_sales.set_index('date')
    return daily_sales

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config['project']['seed'])
    input_path = config['data']['raw']
    output_path = config['data']['processed_daily']
    
    if not os.path.exists(input_path):
        logger.error(f"{input_path} not found. Please download the dataset.")
        return
    
    df = load_data(input_path)
    df = clean_data(df)
    daily_sales = aggregate_daily_sales(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_sales.to_csv(output_path)
    logger.info(f"Data aggregation completed! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
