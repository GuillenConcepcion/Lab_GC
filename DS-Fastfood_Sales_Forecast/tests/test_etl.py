import pytest
import pandas as pd
import numpy as np

# Adjust path to import from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.etl import clean_data

def test_clean_data_removes_duplicates():
    # Arrange
    data = {
        'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'item_price': ['$10', '$10', '$15', '$15', '$15', '$15'],
        'transaction_amount': ['$10', '$10', '$30', '$30', '$30', '$30'],
        'quantity': [1, 1, 2, 2, 2, 2]
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned_df = clean_data(df)
    
    # Assert
    assert len(cleaned_df) == 5, "Should have removed 1 exact duplicate row."

def test_clean_data_parses_prices():
    # Arrange
    data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'item_price': ['$1,200.50', '$10.00', '$20.00'],
        'transaction_amount': ['$2,500.00', '$10.00', '$20.00'],
        'quantity': [10, 10, 10]
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned_df = clean_data(df)
    
    # Assert
    assert cleaned_df['item_price'].iloc[0] == 1200.50, "Failed to strip $ and , from item_price"
    assert cleaned_df['transaction_amount'].iloc[0] == 2500.00, "Failed to strip $ and , from transaction_amount"
