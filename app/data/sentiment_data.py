"""
This Module is responsible for handling the sentiment analysis data layer.
"""
# Model Layer
from app.models.bertweet_model import BertweetSentiment

from app.utils.logger import logger

class SentimentDataLayer:
    def __init__(self, config: dict):
        """
        Initialize the Sentiment Data Layer.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('sentiment_analysis')
        self.default_model = self.config.get('default_model')

        # Initialize the appropriate model based on the configuration
        if self.default_model == "bertweet":
            self.model = BertweetSentiment(config)
        # elif self.default_model == "another_model":
        #     self.model = AnotherModel(config)  # Replace with your other model class
        else:
            raise ValueError(f"Unsupported sentiment analysis model: {self.default_model}")
        
    def analyze(self, texts: str) -> list:
        """
        Perform sentiment analysis on the given text or list of texts.
        
        :param text: Input text or list of texts for sentiment analysis.
        :return: Dictionary for single input or list of dictionaries for batch input.
        """
        try:
            batch_results = self.model(texts)

            results = []
            for res in batch_results:
                results.append({
                    'label': res['label'],
                    'confidence': round(float(res['confidence']), 2)
                })
            
            
            if isinstance(texts, str):
                return results[0]
            
            return results

        except Exception as e:
            logger.error(f"[error] [Data Layer] [SentimentDataLayer] [analyze] An error occurred during sentiment analysis: {str(e)}")
            # print(f"[error] [Data Layer] [SentimentDataLayer] [analyze] An error occurred during sentiment analysis: {str(e)}")
            return {'error': f'An unexpected error occurred while processing the request.'}  # Generic error message
        

# if __name__ == "__main__":
#     config = {
#         'debug': True,
#         'sentiment_analysis': {
#             'default_model': "bertweet",  # Specify the default sentiment analysis model (e.g., bertweet, another_model)
#             'bertweet': {
#                 'model_name': 'finiteautomata/bertweet-base-sentiment-analysis',
#                 'device': 'cpu'
#             }
#         }
#     }
#     sentiment_data = SentimentDataLayer(config)

#     test_batch = [
#         "I love this product!",
#         "I hate this product!",
#         "I am neutral about this product."
#     ]

#     print("\n--- Testing Batch Inference ---")
#     results = sentiment_data.analyze(test_batch)
#     print(results)

#     print("\n--- Testing Single Inference ---")
#     result = sentiment_data.analyze("I love this product!")
#     print(result)
#  Run:
#  python -m app.data.sentiment_data