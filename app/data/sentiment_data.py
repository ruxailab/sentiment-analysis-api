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
        
    def analyze(self, text: str) -> dict:
        """
        Perform sentiment analysis on the given text.
        :param text: Input text for sentiment analysis.
        :return: Predicted label and confidence score, or an error dict.
        """
        try:
            outputs, probabilities, predicted_label, confidence = self.model(text)
            return {
                # 'outputs': outputs,
                # 'probabilities': probabilities,
                'label': predicted_label,
                'confidence': confidence
            }
        
        except Exception as e:
            logger.error(f"[error] [Data Layer] [SentimentDataLayer] [analyze] An error occurred during sentiment analysis: {str(e)}")
            # print(f"[error] [Data Layer] [SentimentDataLayer] [analyze] An error occurred during sentiment analysis: {str(e)}")
            return {'error': f'An unexpected error occurred while processing the request.'}  # Generic error message

    def analyze_batch(self, texts: list) -> dict:
        """
        Perform sentiment analysis on a batch of texts in a single forward pass.
        :param texts: List of input texts.
        :return: {'results': [{'label', 'confidence'}, ...]} or an error dict.
        """
        try:
            if not texts:
                return {'results': []}

            batch_results = self.model(texts)
            return {
                'results': [
                    {'label': label, 'confidence': confidence}
                    for label, confidence in batch_results
                ]
            }

        except Exception as e:
            logger.error(f"[error] [Data Layer] [SentimentDataLayer] [analyze_batch] An error occurred during sentiment analysis: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}
        

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
#     print("config",config)
#     sentiment_data = SentimentDataLayer(config)
#     print("sentiment_data",sentiment_data)

#     print(sentiment_data.analyze("I love this product!"))
#     print(sentiment_data.analyze("I hate this product!"))
#     print(sentiment_data.analyze("I am neutral about this product."))

# #  Run:
# #  python -m app.data.sentiment_data