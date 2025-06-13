# src/analyze.py

from src.preprocess import preprocess_text, SUICIDE_INDICATORS, FIRST_PERSON_PRONOUNS, sid

def analyze_text(text):
    """
    Analyzes text for suicide risk assessment using various linguistic and sentiment features.
    
    This function performs a comprehensive analysis of input text by:
    1. Preprocessing the text by cleaning the unrelated words and other noise
    2. Detecting suicide-related keywords
    3. Counting first-person pronouns
    4. Analyzing sentiment by calculating Compound score from VADER
    5. Calculating text statistics like length and word count
    6. Computing a risk score based on multiple factors, such as strong negative sentiment, multiple suicide-related keywords, and high frequency of first-person pronouns
    
    The risk level is determined by a simple scoring system:
    - Low (0 points): No significant risk indicators
    - Medium (1 point): One risk factor present
    - High (2-3 points): Multiple risk factors present
    
    Risk factors include:
    - Strong negative sentiment (score < -0.5)
    - Multiple suicide-related keywords (> 2)
    - High frequency of first-person pronouns (> 3)
    
    Args:
        text (str): The input text to be analyzed
        
    Returns:
        dict: A dictionary containing analysis results:
            - processed_text (str): Cleaned and normalized text
            - suicide_keywords (list): List of detected suicide-related keywords
            - first_person_count (int): Count of first-person pronouns
            - sentiment_score (float): Compound sentiment score (-1 to 1)
            - text_length (int): Length of processed text
            - word_count (int): Number of words
            - risk_level (str): Risk assessment level ('Low', 'Medium', or 'High')
            
    Example:
        >>> result = analyze_text("I feel so alone and hopeless")
        >>> print(result['risk_level'])
        'High'
    """
    # Preprocess the input text
    processed_text = preprocess_text(text)
    words = processed_text.split()

    # Detect suicide-related keywords from the preprocessed text
    indicators_found = [word for word in words if word in SUICIDE_INDICATORS]

    # Count occurrences of first-person pronouns
    first_person_count = sum(1 for word in words if word in FIRST_PERSON_PRONOUNS)

    # Calculate sentiment score using VADER
    # Compound score ranges from -1 (most negative) to 1 (most positive)
    sentiment_score = sid.polarity_scores(processed_text)['compound']

    # Calculate basic text statistics
    text_length = len(processed_text)  # Character count
    word_count = len(words)  # Word count

    # Calculate risk score based on multiple factors
    risk_score = 0
    # Add point for strong negative sentiment
    if sentiment_score < -0.5:
        risk_score += 1
    # Add point for multiple suicide-related keywords
    if len(indicators_found) > 2:
        risk_score += 1
    # Add point for high frequency of first-person pronouns
    if first_person_count > 3:
        risk_score += 1

    # Determine risk level based on total score
    if risk_score == 0:
        risk_level = "Low"
    elif risk_score == 1:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Return comprehensive analysis results
    return {
        'processed_text': processed_text,
        'suicide_keywords': indicators_found,
        'first_person_count': first_person_count,
        'sentiment_score': sentiment_score,
        'text_length': text_length,
        'word_count': word_count,
        'risk_level': risk_level
    }
