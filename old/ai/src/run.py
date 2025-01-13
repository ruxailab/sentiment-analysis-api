import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Sample text
text = "I absolutely love this product! It exceeded my expectations."

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

dialogue = [
    "Did you liked the movie?",
    "Yes, I loved it.",
    "Did you hate the movie?",
    "No,I did not.",
    # "No, I enjoyed it" # Positive
    # "I don't hate it" # Positive
    "Did you enjoy the movie?",
    "It is confusing.",
    "I hated that movie.",
]

for text in dialogue:
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)

    # Interpret sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"The sentiment of the text is: {sentiment}")
