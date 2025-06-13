# src/preprocess.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Suicide indicators
SUICIDE_INDICATORS = [
    'kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope', 
    'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad', 
    'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless'
]

# First-person pronouns
FIRST_PERSON_PRONOUNS = ['i', 'me', 'my', 'mine', 'myself']

sid = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Preprocesses text data for suicide risk detection by cleaning and normalizing the input.
    
    This function performs several text cleaning steps:
    1. Converts text to lowercase for consistency
    2. Removes URLs and web links
    3. Removes @mentions
    4. Removes special characters and punctuation
    5. Removes numbers
    6. Normalizes whitespace
    
    Args:
        text (str): The input text to be preprocessed
        
    Returns:
        str: The preprocessed text, or empty string if input is not a string
        
    Example:
        >>> preprocess_text("Hello! Check out https://example.com @user123")
        'hello check out'
    """
    if isinstance(text, str):
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove URLs and web links
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Normalize whitespace (remove extra spaces and trim)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def engineer_features(df):
    """
    Engineers features from text data for suicide risk detection.
    
    This function performs several steps:
    1. Input validation and data cleaning:
       - Checks for empty DataFrame
       - Validates required columns ('text' and 'class')
       - Ensures class values are either 'suicide' or 'non-suicide'
    
    2. Text processing:
       - Applies preprocess_text function to clean and normalize text
       - Drops rows with empty or invalid text after preprocessing
    
    3. Feature engineering:
       - Text metrics:
         * text_length: Number of characters in processed text
         * word_count: Number of words in processed text
       - Sentiment analysis (using VADER):
         * sentiment_score: Compound sentiment score (-1 to 1)
         * sentiment_neg: Negative sentiment score (0 to 1)
         * sentiment_pos: Positive sentiment score (0 to 1)
       - Suicide indicators:
         * suicide_word_count: Count of suicide-related words from SUICIDE_INDICATORS list
       - Personal pronouns:
         * first_person_count: Count of first-person pronouns (I, me, my, mine, myself)
    
    Note: Text preprocessing (lowercase conversion, URL removal, etc.) is handled by the
    preprocess_text function, which is called within this function.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'text': Raw text data (str)
            - 'class': Target labels ('suicide' or 'non-suicide')
        
    Returns:
        pandas.DataFrame: Processed DataFrame with original columns plus:
            - 'processed_text': Cleaned and normalized text (from preprocess_text)
            - 'label': Binary labels (1 for suicide, 0 for non-suicide)
            - 'text_length': Length of processed text
            - 'word_count': Number of words
            - 'sentiment_score': Compound sentiment score
            - 'sentiment_neg': Negative sentiment score
            - 'sentiment_pos': Positive sentiment score
            - 'suicide_word_count': Count of suicide indicator words
            - 'first_person_count': Count of first-person pronouns
        
    Raises:
        ValueError: If input DataFrame is empty, missing required columns,
                   contains invalid class values, or has no valid text after preprocessing
                   
    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['I feel sad and hopeless', 'Hello world'],
        ...     'class': ['suicide', 'non-suicide']
        ... })
        >>> result = engineer_features(df)
        >>> print(result.columns)
        ['text', 'class', 'processed_text', 'label', 'text_length', 'word_count',
         'sentiment_score', 'sentiment_neg', 'sentiment_pos', 'suicide_word_count',
         'first_person_count']
    """
    # Input validation
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check for required columns
    required_columns = ['text', 'class']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate class values
    valid_classes = {'suicide', 'non-suicide'}
    invalid_classes = set(df['class'].unique()) - valid_classes
    if invalid_classes:
        raise ValueError(f"Invalid class values found: {invalid_classes}")
    
    # Apply text preprocessing using preprocess_text function
    df['processed_text'] = df['text'].apply(preprocess_text)
    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'] != ""]
    
    if df.empty:
        raise ValueError("No valid text data after preprocessing")
    
    # Convert class labels to binary (1 for suicide, 0 for non-suicide)
    df['label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})
    
    # Calculate text metrics
    df['text_length'] = df['processed_text'].apply(len)  # Character count
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))  # Word count

    # Calculate sentiment scores using VADER
    # Compound score ranges from -1 (most negative) to 1 (most positive)
    df['sentiment_score'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    # Negative and positive scores range from 0 to 1
    df['sentiment_neg'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['neg'])
    df['sentiment_pos'] = df['processed_text'].apply(lambda x: sid.polarity_scores(x)['pos'])

    # Count occurrences of suicide indicator words
    # Sums up all matches from the SUICIDE_INDICATORS list
    df['suicide_word_count'] = df['processed_text'].apply(
        lambda x: sum(1 for word in x.split() if word in SUICIDE_INDICATORS)
    )

    # Count occurrences of first-person pronouns
    # Sums up all matches from the FIRST_PERSON_PRONOUNS list
    df['first_person_count'] = df['processed_text'].apply(
        lambda x: sum(1 for word in x.split() if word in FIRST_PERSON_PRONOUNS)
    )
    
    return df
