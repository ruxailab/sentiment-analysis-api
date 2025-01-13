import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample texts
texts = ["I love this product", "This is the worst service", "I am very happy", "I am not satisfied with this"]
texts=["Are you still getting the error?",
    "No, I do not get it now."]

# Analyze sentiment for each text
for text in texts:
    sentiment_scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment Scores: {sentiment_scores}")