"""
Training script for sentiment analysis models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_train_data
from src.preprocess import TextPreprocessor
from src.models import SentimentModel
from src.feature_engineering import create_features


def train_baseline_model(data_dir=".", model_type='ridge', 
                        feature_type='tfidf', preprocess=True, 
                        test_size=0.2, random_state=42, **kwargs):
    """
    Train a baseline model on the training data.
    
    Args:
        data_dir: Directory containing train.csv
        model_type: Type of model to train ('ridge', 'rf', 'logistic', etc.)
        feature_type: Type of features ('tfidf', 'word2vec', 'fasttext', 'glove')
        preprocess: Whether to preprocess text
        test_size: Proportion of data to use for validation
        random_state: Random seed
        **kwargs: Additional arguments for feature creation
        
    Returns:
        Tuple of (model, feature_model, metrics)
    """
    # Load data
    print("Loading training data...")
    df = load_train_data(data_dir)
    
    X = df['Review'].values
    y = df['Rating'].values
    
    print(f"Loaded {len(X)} samples")
    print(f"Rating distribution: {np.bincount(y.astype(int))}")
    
    # Preprocess text if specified
    if preprocess:
        print("Preprocessing text...")
        preprocessor = TextPreprocessor()
        X = preprocessor.preprocess_series(pd.Series(X)).values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Create features
    print(f"Creating {feature_type.upper()} features...")
    feature_kwargs = kwargs.copy()
    if feature_type == 'tfidf' and 'max_features' not in feature_kwargs:
        feature_kwargs['max_features'] = 10000
    
    X_train_features, X_val_features, feature_model = create_features(
        X_train, X_val, feature_type=feature_type, **feature_kwargs
    )
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # For Logistic Regression, convert ratings to integers (1-5) for classification
    if model_type == 'logistic':
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
    
    # Train model
    print(f"Training {model_type} model...")
    model = SentimentModel(model_type=model_type)
    model.train(X_train_features, y_train)
    
    # Evaluate
    print("Evaluating model...")
    train_metrics = model.evaluate(X_train_features, y_train)
    val_metrics = model.evaluate(X_val_features, y_val)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, feature_model, {'train': train_metrics, 'val': val_metrics}


if __name__ == "__main__":
    # Train a baseline model
    model, vectorizer, metrics = train_baseline_model(
        model_type='ridge',
        preprocess=True
    )
    
    print("\nModel training completed!")

