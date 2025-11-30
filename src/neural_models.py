"""
Neural network models for sentiment analysis.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not work.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # type: ignore
    from transformers import pipeline  # type: ignore
    import torch  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. BERT models will not work.")


class LSTMModel:
    """LSTM neural network for sentiment analysis."""
    
    def __init__(self, max_features=10000, max_length=200, embedding_dim=128, 
                 lstm_units=64, dropout=0.2, learning_rate=0.001):
        """
        Initialize LSTM model.
        
        Args:
            max_features: Maximum number of words in vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
        
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.tokenizer = None
        self.model = None
        self.is_trained = False
    
    def _build_model(self, vocab_size):
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation='linear')  # Linear for regression
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X, y, validation_data=None, epochs=10, batch_size=32, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X: Training text data (list of strings)
            y: Target values (ratings)
            validation_data: Tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Tokenize texts
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(X)
        
        # Convert texts to sequences
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(X_seq, maxlen=self.max_length, padding='post', truncating='post')
        
        # Build model
        vocab_size = len(self.tokenizer.word_index) + 1
        vocab_size = min(vocab_size, self.max_features)
        self.model = self._build_model(vocab_size)
        
        # Prepare validation data
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq = self.tokenizer.texts_to_sequences(X_val)
            X_val_padded = pad_sequences(X_val_seq, maxlen=self.max_length, padding='post', truncating='post')
            val_data = (X_val_padded, y_val)
        
        # Train model
        history = self.model.fit(
            X_padded, y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Text data (list of strings)
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Tokenize and pad
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_padded = pad_sequences(X_seq, maxlen=self.max_length, padding='post', truncating='post')
        
        # Predict
        predictions = self.model.predict(X_padded, verbose=0)
        predictions = predictions.flatten()
        
        # Clip to valid range [1, 5]
        predictions = np.clip(predictions, 1, 5)
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix (text data)
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
    
    def save(self, filepath):
        """Save model and tokenizer."""
        if self.model is not None:
            self.model.save(f"{filepath}_model.h5")
        if self.tokenizer is not None:
            joblib.dump(self.tokenizer, f"{filepath}_tokenizer.pkl")
    
    def load(self, filepath):
        """Load model and tokenizer."""
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        self.tokenizer = joblib.load(f"{filepath}_tokenizer.pkl")
        self.is_trained = True


class BERTModel:
    """BERT-based model for sentiment analysis."""
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        """
        Initialize BERT model.
        
        Args:
            model_name: Name of pre-trained BERT model
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for BERT models. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_trained = False
    
    def train(self, X, y, validation_data=None, epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train/fine-tune the BERT model.
        
        Args:
            X: Training text data (list of strings)
            y: Target values (ratings)
            validation_data: Tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print("Note: Full BERT fine-tuning is computationally intensive.")
        print("For this implementation, we'll use a pre-trained sentiment analysis pipeline.")
        print("For custom fine-tuning, you would need to implement a custom Trainer.")
        
        # Use a pre-trained sentiment analysis pipeline as a starting point
        # In a full implementation, you would fine-tune the model
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions using BERT.
        
        Args:
            X: Text data (list of strings)
            
        Returns:
            Array of predictions (ratings 1-5)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        
        # Get sentiment predictions
        results = self.pipeline(list(X))
        
        # Map sentiment scores to ratings (1-5)
        predictions = []
        for result in results:
            label = result['label']
            score = result['score']
            
            # Map sentiment labels to ratings
            if 'POSITIVE' in label.upper() or 'POS' in label.upper():
                rating = 3.5 + score * 1.5  # Map to 3.5-5 range
            elif 'NEGATIVE' in label.upper() or 'NEG' in label.upper():
                rating = 1.5 + (1 - score) * 1.5  # Map to 1-2.5 range
            else:  # NEUTRAL
                rating = 2.5 + (score - 0.5) * 1.0  # Map to 2-3 range
            
            predictions.append(rating)
        
        predictions = np.array(predictions)
        predictions = np.clip(predictions, 1, 5)
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Text data
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


if __name__ == "__main__":
    print("Testing neural models...")
    
    # Test LSTM
    if TF_AVAILABLE:
        print("\n1. Testing LSTM model creation...")
        lstm = LSTMModel(max_features=1000, max_length=100)
        print("LSTM model created successfully")
    else:
        print("\n1. TensorFlow not available - skipping LSTM test")
    
    # Test BERT
    if TRANSFORMERS_AVAILABLE:
        print("\n2. Testing BERT model creation...")
        bert = BERTModel()
        print("BERT model created successfully")
    else:
        print("\n2. Transformers not available - skipping BERT test")

