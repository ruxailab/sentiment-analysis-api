"""
This module contains the routes for the audio transcript sentiment analysis.
"""

from flask_restx import Namespace, Resource, fields
from flask import request

# Services
from app.services.audio_transcription_sentiment_pipeline import AudioTranscriptionSentimentPipeline

service = AudioTranscriptionSentimentPipeline()

def register_routes(api):
    # Define the model for the audio transcript sentiment analysis request body
    audio_transcript_sentiment_request_model = api.model('AudioTranscriptSentimentRequestModel', {
        'url': fields.String(required=True, description='URL or path of the audio/video file.', example='https://example.com/audio.mp3'),
        'start_time_ms': fields.Integer(required=True, description='Start time in milliseconds.', example=0),
        'end_time_ms': fields.Integer(description='End time in milliseconds.', example=5000),
    })

    audio_transcript_sentiment_bad_request_model = api.model('AudioTranscriptSentimentBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='URL is required'),
        'data': fields.String(description='The data returned by the server (if any)',example=None)
    })

    audio_transcript_sentiment_internal_server_error_model = api.model('AudioTranscriptSentimentInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during audio transcript sentiment analysis.'),
        'data': fields.String(description='The data returned by the server (if any)',example=None)
    })

    audio_transcript_sentiment_success_model = api.model('AudioTranscriptSentimentSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'data': fields.Nested(api.model('AudioTranscriptSentimentDataModel', {
            'audio_path': fields.String(required=True, description='URL of the extracted audio file.', example='https://example.com/extracted_audio.mp3'),
            'start_time_ms': fields.Integer(required=True, description='Start time in milliseconds.', example=0),
            'end_time_ms': fields.Integer(required=True, description='End time in milliseconds.', example=5000),
            'transcription': fields.String(required=True, description='Extracted transcript.', example='Hello, world!'),
            'utterances_sentiment': fields.List(fields.Nested(api.model('AudioTranscriptSentimentUtteranceModel', {
                'timestamp': fields.List(fields.Integer, required=True, description='Start and end time of the utterance.', example=[0, 5000]),
                'text': fields.String(required=True, description='The utterance text.', example='Hello, world!'),
                'label': fields.String(required=True, description='The sentiment of the utterance.', enum= ['POS', 'NEG', 'NEU'], example='POS'),
                'confidence': fields.Float(required=True, description='The confidence score of the sentiment.', example=0.95)
            })))  # Embed the data model
        }))  # Embed the data model
    })


    # Define the endpoint for the Audio Transcript Sentiment Analysis
    @api.route('/process')
    class AudioTranscriptSentiment(Resource):
        @api.doc(description="Perform sentiment analysis on the transcript of an audio or Video file.")
        @api.expect(audio_transcript_sentiment_request_model)  # Use the model for request validation
        @api.response(200, 'Success', audio_transcript_sentiment_success_model)
        @api.response(400, 'Bad Request', audio_transcript_sentiment_bad_request_model)
        @api.response(500, 'Internal Server Error', audio_transcript_sentiment_internal_server_error_model)
        def post(self):
            """
            Endpoint to perform sentiment analysis on the transcript of an audio or Video file.
                - url (str): The URL or path of the audio/video file.
                - start_time_ms (int, optional): Start time in milliseconds (defaults to 0).
                - end_time_ms (int, optional): End time in milliseconds (defaults to full audio if not provided).
            """
            try:
                # Parse the request body
                data = request.json

                url = data.get('url')
                start_time_ms = data.get('start_time_ms',0)
                end_time_ms = data.get('end_time_ms')

                if not url:
                    return {
                        'status': 'error',
                        'error': 'url is required.',
                        'data': None
                    }, 400
                
                if start_time_ms < 0:
                    return {
                        'status': 'error',
                        'error': "'start_time_ms' cannot be negative.",
                        'data': None
                    }, 400
                
                if end_time_ms is not None and end_time_ms < 0:
                    return {
                        'status': 'error',
                        'error': "'end_time_ms' cannot be negative.",
                        'data': None
                    }, 400
                
                if end_time_ms is not None and end_time_ms <= start_time_ms:
                    return {
                        'status': 'error',
                        'error': "'end_time_ms' must be greater than 'start_time_ms'.",
                        'data': None
                    }, 400
                

                # Call the service to perform sentiment analysis on the audio transcript
                # result = service.process(url = url, start_time_ms = start_time_ms, end_time_ms = end_time_ms)
                result = service.process_batch(url = url, start_time_ms = start_time_ms, end_time_ms = end_time_ms)
                

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error': result['error'],
                        'data': None
                    }, 500 # Internal Server Error
                
                return {
                    'status': 'success',
                    'data':{
                        'audio_path': result['audio_path'],
                        'start_time_ms': result['start_time_ms'],
                        'end_time_ms': result['end_time_ms'],
                        'transcription': result['transcription'],
                        'utterances_sentiment': result['utterances_sentiment']
                    }
                }, 200


            except Exception as e:
                return {
                    'status': 'error',
                    'error': 'An unexpected error occurred while processing the request.', # Generic error message
                    'data': None
                }, 500
            
# Define the namespace
api = Namespace('Audio Transcript Sentiment', description='Audio Transcript Sentiment Analysis')

# Register the routes
register_routes(api)