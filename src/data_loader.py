"""
Data loading utilities for the sentiment analysis project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_train_data(data_dir="."):
    """
    Load training data from CSV file.
    
    Args:
        data_dir: Directory containing train.csv
        
    Returns:
        DataFrame with columns: Id, Review, Rating
    """
    train_path = Path(data_dir) / "train.csv"
    df = pd.read_csv(train_path)
    return df


def load_test_data(data_dir="."):
    """
    Load test data from CSV file.
    
    Args:
        data_dir: Directory containing test.csv
        
    Returns:
        DataFrame with columns: Id, Review
    """
    test_path = Path(data_dir) / "test.csv"
    df = pd.read_csv(test_path)
    return df


def load_sample_submission(data_dir="."):
    """
    Load sample submission file to understand format.
    
    Args:
        data_dir: Directory containing sample_submission.csv
        
    Returns:
        DataFrame with columns: Id, Rating
    """
    sub_path = Path(data_dir) / "sample_submission.csv"
    df = pd.read_csv(sub_path)
    return df


def get_data_info(df, name="Dataset"):
    """
    Print basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset
    """
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    if 'Rating' in df.columns:
        print(f"\nRating distribution:")
        print(df['Rating'].value_counts().sort_index())
        print(f"\nRating statistics:")
        print(df['Rating'].describe())


if __name__ == "__main__":
    # Test data loading
    print("Loading training data...")
    train_df = load_train_data()
    get_data_info(train_df, "Training Data")
    
    print("\n" + "="*50)
    print("Loading test data...")
    test_df = load_test_data()
    get_data_info(test_df, "Test Data")

