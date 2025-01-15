"""
This module contains the routes for the transcript endpoint.
"""

from flask_restx import Namespace, Resource, fields
from flask import request

# Services
from app.services.transcript_service import TranscriptService

service = TranscriptService()

def register_routes(api):
    # Define the model for the transcript extraction request body
    transcription_transcribe_request_model = api.model('TranscriptionTranscribeRequestModel', {
        'file_path': fields.String(required=True, description='Path of the audio file.', example='audio.mp3')
    })

    transcription_transcribe_bad_request_model = api.model('TranscriptionTranscribeBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='file_path is required')
    })

    transcription_transcribe_internal_server_error_model = api.model('TranscriptionTranscribeInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during transcription.')
    })

    transcription_transcribe_success_model = api.model('TranscriptionTranscribeSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'transcription': fields.String(required=True, description='Extracted transcript.', example='Hello, world!')
    })

    # Define the endpoint for the Transcribe an audio file.
    @api.route('/transcribe') 
    class TranscriptionTranscribe(Resource):
        @api.doc(description="Transcribe an audio file.")
        @api.expect(transcription_transcribe_request_model)  # Use the model for request validation
        @api.response(200, 'Success',transcription_transcribe_success_model)
        @api.response(400, 'Bad Request', transcription_transcribe_bad_request_model)
        @api.response(500, 'Internal Server Error', transcription_transcribe_internal_server_error_model)
        def post(self):
            """
            Endpoint to extract transcript from an audio file.
                - file (str): path of the audio file.
            """
            try:
                # Parse the request body
                data = request.json

                file_path = data.get('file_path')

                if not file_path:
                    return {
                        'status': 'error',
                        'error': 'file_path is required.'
                    }, 400
                
                # Call the service to transcribe the audio file
                result = service.transcribe(audio_file_path = file_path)

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error': "An error occurred during transcription."
                    }, 500 # Internal Server Error
                
                # Return the transcribed text
                return {
                    'status': 'success',
                    'transcription': result['transcription']
                }, 200
            
            except Exception as e:
                print(f"[error] [Route Layer] [TranscriptionTranscribe] [post] An error occurred: {str(e)}")
                return {
                    'status': 'error',
                    'error': 'An unexpected error occurred during transcription.'
                }, 500
            
    # Define the model for the transcript chunks extraction request body
    transcription_chunks_request_model = api.model('TranscriptionChunksRequestModel', {
        'file_path': fields.String(required=True, description='Path of the audio file.', example='audio.mp3')
    })

    transcription_chunks_bad_request_model = api.model('TranscriptionChunksBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='file_path is required')
    })

    transcription_chunks_internal_server_error_model = api.model('TranscriptionChunksInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during transcription.')
    })

    transcription_chunks_success_model = api.model('TranscriptionChunksSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'transcription': fields.String(required=True, description='Extracted transcript.', example='Hello, world!'),
        'chunks': fields.List(fields.String, required=True, description='Extracted chunks.', example=['Hello', 'world'])
    })

    # Define the endpoint for the Transcribe an audio file and return chunks.
    @api.route('/chucks') 
    class TranscriptionChunks(Resource):
        @api.doc(description="Transcribe an audio file and return chunks.")
        @api.expect(transcription_chunks_request_model)  # Use the model for request validation
        @api.response(200, 'Success', transcription_chunks_success_model)
        @api.response(400, 'Bad Request', transcription_chunks_bad_request_model)
        @api.response(500, 'Internal Server Error', transcription_chunks_internal_server_error_model)
        def post(self):
            """
            Endpoint to extract transcript from an audio file and return chunks.
                - file (str): path of the audio file.
            """
            try:
                # Parse the request body
                data = request.json

                file_path = data.get('file_path')

                if not file_path:
                    return {
                        'status': 'error',
                        'error': 'file_path is required.'
                    }, 400
                
                # Call the service to transcribe the audio file
                result = service.transcribe(audio_file_path = file_path)

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error': "An error occurred during transcription."
                    }, 500
                
                # Return the transcribed text and chunks
                return {
                    'status': 'success',
                    'transcription': result['transcription'],
                    'chunks': result['chunks']
                }, 200
            
            except Exception as e:
                print(f"[error] [Route Layer] [TranscriptionChunks] [post] An error occurred: {str(e)}")
                return {
                    'status': 'error',
                    'error': 'An unexpected error occurred during transcription.'
                }, 500

# Define the namespace
api = Namespace('Transcript', description="Transcript operations")

# Register the routes
register_routes(api)