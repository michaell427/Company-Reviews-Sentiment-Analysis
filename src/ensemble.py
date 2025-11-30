"""
Ensemble methods for combining multiple sentiment analysis models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EnsembleModel:
    """Ensemble of multiple models for sentiment analysis."""
    
    def __init__(self, models, weights=None, method='average'):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models
            weights: Optional list of weights for each model (if None, equal weights)
            method: Ensemble method ('average', 'weighted_average', 'median')
        """
        self.models = models
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.is_trained = all(hasattr(m, 'is_trained') and m.is_trained for m in models)
    
    def predict(self, X):
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix or text data (depending on model types)
            
        Returns:
            Array of ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("All models must be trained before making ensemble predictions")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Apply ensemble method
        if self.method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.method == 'weighted_average':
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        # Clip to valid range [1, 5]
        ensemble_pred = np.clip(ensemble_pred, 1, 5)
        return ensemble_pred
    
    def evaluate(self, X, y):
        """
        Evaluate ensemble model performance.
        
        Args:
            X: Feature matrix or text data
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
    
    def get_individual_predictions(self, X):
        """
        Get predictions from each individual model.
        
        Args:
            X: Feature matrix or text data
            
        Returns:
            Dictionary mapping model index to predictions
        """
        individual_preds = {}
        for i, model in enumerate(self.models):
            individual_preds[i] = model.predict(X)
        return individual_preds


def create_ensemble(models, weights=None, method='average'):
    """
    Create an ensemble from a list of models.
    
    Args:
        models: List of trained models
        weights: Optional weights for each model
        method: Ensemble method ('average', 'weighted_average', 'median')
        
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(models, weights=weights, method=method)


if __name__ == "__main__":
    # Test ensemble creation
    print("Testing ensemble model...")
    
    # This would be used with actual trained models
    print("Ensemble model class created successfully")
    print("Usage: Create ensemble from trained models")

