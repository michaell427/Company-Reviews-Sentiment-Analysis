# Sentiment Analysis - Company Reviews

Kaggle Competition: Predicting sentiment ratings 1-5 for company reviews using NLP techniques.

## Competition Details

- **Goal**: Predict star ratings 1-5 for company reviews
- **Metric**: Mean Absolute Error
- **Dataset**: 100,000 reviews from Trustpilot across 40+ companies
- **Split**: 60% train, 10% public leaderboard, 30% private leaderboard

## Setup

1. Create a virtual environment:
```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Dataset

- **train.csv**: Training data with Id, Review, and Rating
- **test.csv**: Test data with Id and Review
- **sample_submission.csv**: Submission format template

## Approach

1. **Exploratory Data Analysis**: Understand data distribution, review lengths, rating distribution
2. **Text Preprocessing**: Clean and normalize text data
3. **Feature Engineering**: 
   - TF-IDF vectorization with n-grams
   - Word embeddings: Word2Vec, FastText, GloVe
4. **Model Development**: 
   - Baseline models: Ridge Regression, Random Forest, Logistic Regression
   - Neural networks: LSTM
   - Transformer models: BERT
5. **Model Optimization**:
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation for robust performance estimation
   - Feature importance analysis
   - Error analysis and residual plots
6. **Ensemble**: Combine multiple models (Ridge, Random Forest, Logistic Regression) for improved performance
7. **Submission**: Generate predictions in required format

## Evaluation

Submissions are evaluated using **Mean Absolute Error**:
- Lower MAE = Better performance
- Predictions can be floats

## Rating Interpretation

- 1 = Very unsatisfied
- 2 = Unsatisfied
- 3 = Neutral
- 4 = Satisfied
- 5 = Very satisfied