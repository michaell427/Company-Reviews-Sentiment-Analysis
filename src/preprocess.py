"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for reviews."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True, lowercase=True):
        """
        Initialize preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def clean_text(self, text):
        """
        Clean a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_tokens(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatize or self.lemmatizer is None:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_tokens(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Filter out punctuation-only tokens
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_series(self, series):
        """
        Preprocess a pandas Series of texts.
        
        Args:
            series: pandas Series of text strings
            
        Returns:
            pandas Series of preprocessed texts
        """
        return series.apply(self.preprocess)


def simple_clean(text):
    """
    Simple text cleaning function (faster, less thorough).
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    test_text = "This is a GREAT product! I love it so much. Check it out at https://example.com"
    print(f"Original: {test_text}")
    print(f"Preprocessed: {preprocessor.preprocess(test_text)}")

