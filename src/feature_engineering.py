"""
Feature engineering utilities for sentiment analysis.
Supports TF-IDF, Word2Vec, GloVe, and FastText embeddings.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors
import os
import warnings
warnings.filterwarnings('ignore')


def create_tfidf_features(X_train, X_test, max_features=10000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        max_features: Maximum number of features
        ngram_range: Range of n-grams to use
        
    Returns:
        Tuple of (X_train_features, X_test_features, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    return X_train_features, X_test_features, vectorizer


def tokenize_texts(texts):
    """
    Tokenize a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of tokenized texts (list of word lists)
    """
    tokenized = []
    for text in texts:
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = []
        tokenized.append(tokens)
    return tokenized


def create_word2vec_features(X_train, X_test, vector_size=100, window=5, min_count=2, workers=4):
    """
    Create Word2Vec embeddings from text data.
    
    Args:
        X_train: Training text data (list of strings)
        X_test: Test text data (list of strings)
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        workers: Number of worker threads
        
    Returns:
        Tuple of (X_train_features, X_test_features, model)
    """
    # Tokenize texts
    all_texts = list(X_train) + list(X_test)
    tokenized_texts = tokenize_texts(all_texts)
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0  # 0 for CBOW, 1 for skip-gram
    )
    
    # Create document embeddings by averaging word vectors
    def text_to_vector(text, model):
        tokens = tokenize_texts([text])[0]
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    print("Creating document embeddings...")
    X_train_features = np.array([text_to_vector(text, model) for text in X_train])
    X_test_features = np.array([text_to_vector(text, model) for text in X_test])
    
    return X_train_features, X_test_features, model


def create_fasttext_features(X_train, X_test, vector_size=100, window=5, min_count=2, workers=4):
    """
    Create FastText embeddings from text data.
    
    Args:
        X_train: Training text data (list of strings)
        X_test: Test text data (list of strings)
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        workers: Number of worker threads
        
    Returns:
        Tuple of (X_train_features, X_test_features, model)
    """
    # Tokenize texts
    all_texts = list(X_train) + list(X_test)
    tokenized_texts = tokenize_texts(all_texts)
    
    # Train FastText model
    print("Training FastText model...")
    model = FastText(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0  # 0 for CBOW, 1 for skip-gram
    )
    
    # Create document embeddings by averaging word vectors
    def text_to_vector(text, model):
        tokens = tokenize_texts([text])[0]
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    print("Creating document embeddings...")
    X_train_features = np.array([text_to_vector(text, model) for text in X_train])
    X_test_features = np.array([text_to_vector(text, model) for text in X_test])
    
    return X_train_features, X_test_features, model


def load_glove_embeddings(glove_path, vector_size=100):
    """
    Load pre-trained GloVe embeddings.
    
    Args:
        glove_path: Path to GloVe embedding file
        vector_size: Dimension of word vectors (100, 200, 300, etc.)
        
    Returns:
        Dictionary mapping words to vectors
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings = {}
    
    if not os.path.exists(glove_path):
        print(f"Warning: GloVe file not found at {glove_path}")
        print("You can download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
        return None
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            if len(vector) == vector_size:
                embeddings[word] = vector
    
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings


def create_glove_features(X_train, X_test, glove_path, vector_size=100):
    """
    Create document embeddings using pre-trained GloVe vectors.
    
    Args:
        X_train: Training text data (list of strings)
        X_test: Test text data (list of strings)
        glove_path: Path to GloVe embedding file
        vector_size: Dimension of word vectors
        
    Returns:
        Tuple of (X_train_features, X_test_features, embeddings_dict)
    """
    embeddings = load_glove_embeddings(glove_path, vector_size)
    
    if embeddings is None:
        raise ValueError("Could not load GloVe embeddings")
    
    def text_to_vector(text, embeddings):
        tokens = tokenize_texts([text])[0]
        word_vectors = [embeddings[word] for word in tokens if word in embeddings]
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(vector_size)
    
    print("Creating document embeddings from GloVe...")
    X_train_features = np.array([text_to_vector(text, embeddings) for text in X_train])
    X_test_features = np.array([text_to_vector(text, embeddings) for text in X_test])
    
    return X_train_features, X_test_features, embeddings


def create_features(X_train, X_test, feature_type='tfidf', **kwargs):
    """
    Create features from text data using specified method.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        feature_type: Type of features ('tfidf', 'word2vec', 'fasttext', 'glove')
        **kwargs: Additional arguments for specific feature types
        
    Returns:
        Tuple of (X_train_features, X_test_features, feature_model)
    """
    if feature_type == 'tfidf':
        return create_tfidf_features(X_train, X_test, **kwargs)
    elif feature_type == 'word2vec':
        return create_word2vec_features(X_train, X_test, **kwargs)
    elif feature_type == 'fasttext':
        return create_fasttext_features(X_train, X_test, **kwargs)
    elif feature_type == 'glove':
        if 'glove_path' not in kwargs:
            raise ValueError("glove_path required for GloVe features")
        return create_glove_features(X_train, X_test, **kwargs)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


if __name__ == "__main__":
    # Test feature creation
    print("Testing feature engineering...")
    
    sample_texts = [
        "This is a great product",
        "I love this item",
        "Not very good quality",
        "Terrible experience"
    ]
    
    X_train = sample_texts[:2]
    X_test = sample_texts[2:]
    
    # Test TF-IDF
    print("\n1. Testing TF-IDF...")
    X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(X_train, X_test, max_features=100)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # Test Word2Vec
    print("\n2. Testing Word2Vec...")
    X_train_w2v, X_test_w2v, _ = create_word2vec_features(X_train, X_test, vector_size=50)
    print(f"Word2Vec shape: {X_train_w2v.shape}")

