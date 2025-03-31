"""
This module contains the routes for the sentiment endpoint.
"""

from flask_restx import Namespace, Resource, fields
from flask import request

from app.utils.logger import logger

# Services
from app.services.sentiment_service import SentimentService

service = SentimentService()

def register_routes(api):
    # Define the model for the sentiment analysis request body
    sentiment_analyze_request_model = api.model('SentimentAnalyzeRequestModel', {
        'text': fields.String(required=True, description='Input text for sentiment analysis.', example='I love this product!')
    })

    sentiment_analyze_bad_request_model = api.model('SentimentAnalyzeBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='text is required'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    sentiment_analyze_internal_server_error_model = api.model('SentimentAnalyzeInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during sentiment analysis.'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    sentiment_analyze_success_model = api.model('SentimentAnalyzeSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'data': fields.Nested(api.model('SentimentAnalyzeDataModel', {
            'label': fields.String(required=True, description='Predicted sentiment label.', enum=['POS', 'NEG', 'NEU'], example='POS'),
            'confidence': fields.Float(required=True, description='Confidence score of the prediction.', example=0.95)
        }))  # Embed the data model
    })

    # Define the endpoint for the Analyze sentiment of a text.
    @api.route('/analyze') 
    class SentimentAnalyze(Resource):
        @api.doc(description="Analyze sentiment of a text.")
        @api.expect(sentiment_analyze_request_model)  # Use the model for request validation
        @api.response(200, 'Success', sentiment_analyze_success_model)
        @api.response(400, 'Bad Request', sentiment_analyze_bad_request_model)
        @api.response(500, 'Internal Server Error', sentiment_analyze_internal_server_error_model)
        def post(self):
            """
            Endpoint to analyze sentiment of a text.
                - text (str): Input text for sentiment analysis.
            """
            try:
                # Parse the request body
                data = request.json

                text = data.get('text')

                if not text:
                    return {
                        'status': 'error',
                        'error': 'text is required.',
                        'data': None
                    }, 400
                
                # Call the service to analyze the sentiment of the text
                result = service.analyze(text)

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error': result['error'],
                        'data': None
                    }, 500 # Internal Server Error
                
                # Return the predicted label and confidence score
                return {
                    'status': 'success',
                    'data': {
                        'label': result['label'],
                        'confidence': result['confidence']
                    }
                }
            
            except Exception as e:
                logger.error(f"[error] [Route Layer] [SentimentAnalyze] [post] An error occurred: {str(e)}")
                # print(f"[error] [Route Layer] [SentimentAnalyze] [post] An error occurred: {str(e)}")
                return {
                    'status': 'error',
                    "error": 'An unexpected error occurred while processing the request.', # Generic error message
                    'data': None
                }, 500 # Internal Server Error
            
# Define the namespace for the sentiment endpoint
api = Namespace('Sentiment', description='Sentiment Operations')

# Register the routes
register_routes(api)