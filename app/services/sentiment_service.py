"""
This module contains the service layer for sentiment analysis.
"""
from app.config import Config

from app.utils.logger import logger

# Data Layer for fetching and processing transcripts
from app.data.sentiment_data import SentimentDataLayer
from typing import Union

config = Config().config # Load the configuration

class SentimentService:
    def __init__(self):
        self.debug = config.get('debug')

        self.sentiment_data_layer = SentimentDataLayer(config)

    def analyze(self, texts: Union[str, list]) -> Union[dict, list]:
        """
        Perform sentiment analysis on the given text or list of texts.
        :param texts: Input text or list of texts for sentiment analysis.
        :return: predicted label, and confidence score.
        """
        try:
            results = self.sentiment_data_layer.analyze(texts)

            if isinstance(results, dict) and 'error' in results:
                return {
                    'error': results['error']
                }

            if isinstance(texts, str):
                return self.format_response(results)

            # Batch processing: format each result in the list
            return [self.format_response(res) for res in results]
        
        except Exception as e:
            logger.error(f"[error] [Service Layer] [SentimentService] [analyze] An error occurred during sentiment analysis: {str(e)}")
            # print(f"[error] [Service Layer] [SentimentService] [analyze] An error occurred during sentiment analysis: {str(e)}")
            return {'error': f'An unexpected error occurred while processing the request.'}  # Generic error message
        
    def format_response(self, result: dict) -> dict:
        """
        Format sentiment output into a reusable response structure.
        """
        return {
            'label': result['label'],
            'confidence': result['confidence']
        }

# if __name__ == "__main__":
#     sentiment_service = SentimentService()

#     test_texts = [
#         "I love this product!",
#         "I hate this product!",
#         "I am neutral about this product."
#     ]

#     print("\n--- Testing Batch Inference ---")
#     batch_result = sentiment_service.analyze(test_texts)
#     print("Batch Result:", batch_result)

#     print("\n--- Testing Single Input ---")
#     single_result = sentiment_service.analyze("This is a great day!")
#     print("Single Result:", single_result)

#  Run:
#  python -m app.services.sentiment_service