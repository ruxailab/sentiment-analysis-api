"""
This module contains the routes for the audio endpoint.
"""

from flask_restx import Namespace, Resource,fields
from flask import  request

from app.utils.logger import logger

# Services
from app.services.audio_service import AudioService

service = AudioService()

def register_routes(api):
    # Define the model for the audio extraction request body
    audio_extract_request_model = api.model('AudioExtractRequestModel', {
        'url': fields.String(required=True, description='URL or path of the audio/video file.', example='https://example.com/audio.mp3'),
        'start_time_ms': fields.Integer(required=True, description='Start time in milliseconds.', example=0),
        'end_time_ms': fields.Integer(description='End time in milliseconds.', example=5000),
        'user_id': fields.String(description='User ID for creating user-specific subdirectories.', example='user123')
    })

    audio_extract_bad_request_model = api.model('AudioExtractBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='URL is required'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None),
    })

    audio_extract_internal_server_error_model = api.model('AudioExtractInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred while processing the audio.'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None),
    })

    audio_extract_success_model = api.model('AudioExtractSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'data': fields.Nested(api.model('AudioExtractDataModel', {
            'audio_path': fields.String(required=True, description='URL of the extracted audio file.', example='https://example.com/extracted_audio.mp3'),
            'start_time_ms': fields.Integer(required=True, description='Start time in milliseconds.', example=0),
            'end_time_ms': fields.Integer(required=True, description='End time in milliseconds.', example=5000)
        }))  # Embed the data model
    })

    # Define the endpoint for the extraction route
    @api.route('/extract')  # Ensure the route has a proper path
    class AudioExtract(Resource):
        @api.doc(description="Extract audio from a given URL")
        @api.expect(audio_extract_request_model)  # Use the model for request validation
        @api.response(200, 'Success', audio_extract_success_model)
        @api.response(400, 'Bad Request', audio_extract_bad_request_model)
        @api.response(500, 'Internal Server Error', audio_extract_internal_server_error_model)
        def post(self):
            """
            Endpoint to extract audio from a given URL.
            Expected body parameters:
                - url (str): The URL or path of the audio/video file.
                - start_time_ms (int, optional): Start time in milliseconds (defaults to 0).
                - end_time_ms (int, optional): End time in milliseconds (defaults to full audio if not provided).
                - user_id (str, optional): User ID for creating user-specific subdirectories.
            """
            try:
                # Parse the request body
                data = request.json

                url = data.get('url')
                start_time_ms = data.get('start_time_ms',0)
                end_time_ms = data.get('end_time_ms')
                user_id = data.get('user_id')

                if not url :
                    return {
                        "status": "error",
                        'error': 'URL is required',
                        'data': None
                    }, 400
                
                if start_time_ms < 0:
                    return {
                        "status": "error",
                        "error": "'start_time_ms' cannot be negative.",
                        'data': None
                    }, 400
                
                if end_time_ms is not None and end_time_ms < 0:
                    return {
                        "status": "error",
                        "error": "'end_time_ms' cannot be negative.",
                        'data': None
                    }, 400
                
                if end_time_ms  is not None and end_time_ms < start_time_ms:
                    return {
                        "status": "error",
                        "error": "'end_time_ms' must be greater than or equal to 'start_time_ms'.",
                        'data': None
                    }, 400
                
                
                # Call the service method to extract the audio
                result = service.extract_audio(url, start_time_ms, end_time_ms, user_id)

                if 'error' in result:
                    return {
                        "status": "error",
                        "error": result['error'],
                        "data": None
                    }, 500  # Internal server error, as the problem might be with the service layer
            
                return {
                    "status": "success",
                    "data": {
                        "audio_path": result['audio_path'],
                        "start_time_ms": result['start_time_ms'],
                        "end_time_ms": result['end_time_ms']
                    }
                } , 200
            
            except Exception as e:
                # Log the exception (optional)
                logger.error(f"[error] [Route Layer] [AudioExtract] [post] An unexpected error occurred during audio extraction: {str(e)}")
                # print(f"[error] [Route Layer] [AudioExtract] [post] An unexpected error occurred during audio extraction: {str(e)}")
                return {
                    "status": "error",
                    "error": 'An unexpected error occurred while processing the request.', # Generic error message
                    "data": None
                }, 500  # Return 500 in case of unexpected errors
            

# Define the namespace
api = Namespace('Audio', description="Audio operations")

# Register the routes
register_routes(api)