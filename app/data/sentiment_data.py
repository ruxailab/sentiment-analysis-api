"""
This Module is responsible for handling the sentiment analysis data layer.
"""
# Model Layer
from app.models.bertweet_model import BertweetSentiment

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
        
    def analyze(self, text: str) -> tuple:
        """
        Perform sentiment analysis on the given text.
        :param text: Input text for sentiment analysis.
        :return: Model outputs, probabilities, predicted label, and confidence score.
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
            print(f"[error] [Data Layer] [SentimentDataLayer] [analyze] An error occurred during sentiment analysis: {str(e)}")
            return {'error': f'An unexpected error occurred while processing the request.'}  # Generic error message
        
    def analyze_batch(self, texts: list) -> list:
        """
        Perform sentiment analysis on a list of texts.
        :param texts: List of input texts.
        :return: List of dictionaries each with predicted label and confidence.
        """
        try:
            # Call the batch_forward method on the underlying model
            results = self.model.batch_forward(texts)
            return results
        
        except Exception as e:
            print(f"[error] [Data Layer] [SentimentDataLayer] [analyze_batch] An error occurred: {str(e)}")
            # Return an error for each text in case of failure
            return [{"error": "An unexpected error occurred while processing batch request."} for _ in texts]
        

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