import os
import time
import logging
import pandas as pd
import requests
import schedule
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from config import (
    DATABASE_URI, NBA_API_URL, NBA_API_KEY, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR
)

# Setup Logging
log_file = os.path.join(LOG_DIR, 'etl_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
try:
    engine = create_engine(DATABASE_URI)
    logger.info("Database engine initialized.")
except Exception as e:
    logger.error(f"Failed to initialize database engine: {e}")
    engine = None

def extract_sales_data():
    """Extract local CSV data."""
    file_path = os.path.join(RAW_DATA_DIR, 'sales_data.csv')
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Sales data not found at {file_path}. Skipping.")
            return None
        
        df = pd.read_csv(file_path, encoding='latin1')
        logger.info(f"Successfully extracted {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error extracting sales data: {e}")
        return None

def extract_nba_data():
    """Extract data from external JSON API (balldontlie)."""
    try:
        headers = {}
        if NBA_API_KEY:
            headers['Authorization'] = NBA_API_KEY
            
        # Get first page of players as sample
        response = requests.get(f"{NBA_API_URL}", headers=headers, params={"per_page": 100})
        response.raise_for_status()
        
        data = response.json().get('data', [])
        logger.info(f"Successfully extracted {len(data)} records from NBA API.")
        return data
    except requests.exceptions.HTTPError as errh:
         logger.error(f"HTTP Error querying API: {errh}")
         return None
    except Exception as e:
        logger.error(f"Error extracting NBA data: {e}")
        return None

def transform_sales_data(df):
    """Clean and transform sales data."""
    if df is None or df.empty:
        return None
        
    try:
        # Create a copy to avoid SettingWithCopyWarning
        transformed_df = df.copy()
        
        # Standardize column names (lowercase, replace spaces with underscores)
        transformed_df.columns = [col.lower().strip().replace(' ', '_') for col in transformed_df.columns]
        
        # Handle Missing Values (Example: Drop rows where critical fields are null)
        transformed_df = transformed_df.dropna(subset=['ordernumber', 'sales'])
        
        # Ensure correct data types (e.g. orderdate to datetime)
        if 'orderdate' in transformed_df.columns:
            transformed_df['orderdate'] = pd.to_datetime(transformed_df['orderdate'], errors='coerce')
        
        logger.info(f"Successfully transformed sales data. Shape: {transformed_df.shape}")
        return transformed_df
    except Exception as e:
         logger.error(f"Error transforming sales data: {e}")
         return None

def transform_nba_data(data):
    """Flatten and transform JSON NBA data."""
    if not data:
        return None
        
    try:
        # Flatten the JSON list of dictionaries into a DataFrame
        # balldontlie players have nested 'team' objects
        processed_data = []
        for player in data:
            flat_player = {
                'id': player.get('id'),
                'first_name': player.get('first_name'),
                'last_name': player.get('last_name'),
                'position': player.get('position'),
                'team_id': player.get('team', {}).get('id'),
                'team_name': player.get('team', {}).get('full_name'),
                'team_city': player.get('team', {}).get('city')
            }
            processed_data.append(flat_player)
            
        transformed_df = pd.DataFrame(processed_data)
        logger.info(f"Successfully transformed NBA data. Shape: {transformed_df.shape}")
        return transformed_df
    except Exception as e:
        logger.error(f"Error transforming NBA data: {e}")
        return None

def load_data(df, table_name):
    """Load dataframe into PostgreSQL and save to CSV."""
    if df is None or df.empty:
        logger.warning(f"No data to load for table {table_name}.")
        return False
        
    try:
        # Save to processed directory as backup
        processed_file = os.path.join(PROCESSED_DATA_DIR, f"{table_name}_processed.csv")
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")
        
        # Load into PostgreSQL
        if engine:
            logger.info(f"Loading data to database table: {table_name}...")
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {len(df)} rows into {table_name}")
            return True
        else:
            logger.warning("Database engine not available. Skipping DB load.")
            return False
    except SQLAlchemyError as e:
        logger.error(f"Database error during load to {table_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading {table_name}: {e}")
        return False

def run_etl():
    """Execute the full ETL pipeline."""
    logger.info("--- Starting ETL Pipeline Job ---")
    
    # Extract
    sales_raw = extract_sales_data()
    nba_raw = extract_nba_data()
    
    # Transform
    sales_clean = transform_sales_data(sales_raw)
    nba_clean = transform_nba_data(nba_raw)
    
    # Load
    load_data(sales_clean, 'sales_data')
    load_data(nba_clean, 'nba_players')
    
    logger.info("--- ETL Pipeline Job Completed ---")

if __name__ == "__main__":
    # Execute immediately on start
    run_etl()
    
    # Schedule to run every day at 10 AM (example)
    # logger.info("Scheduling job to run daily at 10:00 AM")
    # schedule.every().day.at("10:00").do(run_etl)
    # 
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
