"""
Model definitions and utilities for sentiment analysis.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib


class SentimentModel:
    """Wrapper class for sentiment analysis models."""
    
    def __init__(self, model_type='ridge', **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ('ridge', 'rf', 'svm', 'gbm', 'linear', 'logistic')
            **kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_trained = False
    
    def _create_model(self, model_type, **kwargs):
        """Create model instance based on type."""
        models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'logistic': LogisticRegression,
            'rf': RandomForestRegressor,
            'svm': SVR,
            'gbm': GradientBoostingRegressor
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set default parameters
        defaults = {
            'ridge': {'alpha': 1.0},
            'logistic': {'multi_class': 'multinomial', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
            'rf': {'n_estimators': 100, 'random_state': 42},
            'svm': {'kernel': 'linear', 'C': 1.0},
            'gbm': {'n_estimators': 100, 'random_state': 42}
        }
        
        params = defaults.get(model_type, {})
        params.update(kwargs)
        
        return models[model_type](**params)
    
    def train(self, X, y):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target values (ratings)
        """
        if self.model_type == 'logistic':
            y = y.astype(int)
            y = np.clip(y, 1, 5)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        
        if self.model_type == 'logistic':
            predictions = predictions.astype(float)
        
        # Clip predictions to valid range [1, 5]
        predictions = np.clip(predictions, 1, 5)
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X)
        
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of folds
            
        Returns:
            Dictionary with CV results
        """
        scores = cross_val_score(self.model, X, y, cv=cv, 
                                scoring='neg_mean_absolute_error')
        return {
            'mean_MAE': -scores.mean(),
            'std_MAE': scores.std(),
            'scores': -scores
        }
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    models_to_test = ['ridge', 'rf', 'linear']
    for model_type in models_to_test:
        model = SentimentModel(model_type=model_type)
        print(f"Created {model_type} model: {type(model.model)}")

