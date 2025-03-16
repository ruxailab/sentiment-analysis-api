"""
This module contains the service layer for sentiment analysis.
"""
from app.config import Config

# Data Layer for fetching and processing transcripts
from app.data.sentiment_data import SentimentDataLayer

config = Config().config # Load the configuration

class SentimentService:
    def __init__(self):
        self.debug = config.get('debug')

        self.sentiment_data_layer = SentimentDataLayer(config)

    def analyze(self, text: str) -> tuple:
        """
        Perform sentiment analysis on the given text.
        :param text: Input text for sentiment analysis.
        :return: predicted label, and confidence score.
        """
        try:
            result = self.sentiment_data_layer.analyze(text)

            if isinstance(result, dict) and 'error' in result:
                return {
                    'error': result['error']
                }

            # Return the predicted label and confidence score
            return {
                'label': result['label'],
                'confidence': result['confidence']
            }
        
        except Exception as e:
            print(f"[error] [Service Layer] [SentimentService] [analyze] An error occurred during sentiment analysis: {str(e)}")
            return {'error': f'An unexpected error occurred while processing the request.'}  # Generic error message
    
    def analyze_batch(self, texts: list) -> list:
        """
        Perform sentiment analysis on a list of texts.
        :param texts: List of input texts.
        :return: List of dictionaries each with predicted label and confidence
        """
        try:
            results = self.sentiment_data_layer.analyze_batch(texts)
            return results
        
        except Exception as e:
            print(f"[error] [Service Layer] [SentimentService] [analyze_batch] An error occurred: {str(e)}")
            return [{"error": "An unexpected error occurred while processing batch request."} for _ in texts]

        

# if __name__ == "__main__":
#     sentiment_service = SentimentService()

#     result = sentiment_service.analyze("I love this product!")
#     print("result",result)

#     result = sentiment_service.analyze("I hate this product!")
#     print("result",result)

#     result = sentiment_service.analyze("I am neutral about this product.")
#     print("result",result)

# #  Run:
# #  python -m app.services.sentiment_service