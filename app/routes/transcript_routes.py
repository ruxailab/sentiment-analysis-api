"""
This module contains the routes for the transcription endpoint.
"""

import os
from flask_restx import Namespace, Resource, fields
from flask import request

from app.utils.logger import logger

# Services
from app.services.transcript_service import TranscriptService

service = TranscriptService()

from app.services.audio_service import AudioService
from app.config import Config

audio_service=AudioService()
config = Config().config 
debug = config.get('debug')


def register_routes(api):
    # Define the model for the transcript extraction request body
    transcription_transcribe_request_model = api.model('TranscriptionTranscribeRequestModel', {
        'file_path': fields.String(required=False, description='Path of the audio file.', example='audio.mp3'),
        'audio_url': fields.String(required=False, description='Url of the audio file.', example='https://audio.mp3'),
    })

    transcription_transcribe_bad_request_model = api.model('TranscriptionTranscribeBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='file_path is required'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    transcription_transcribe_internal_server_error_model = api.model('TranscriptionTranscribeInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during transcription.'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    transcription_transcribe_success_model = api.model('TranscriptionTranscribeSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'data': fields.Nested(api.model('TranscriptionTranscribeDataModel', {
            'transcription': fields.String(required=True, description='Extracted transcript.', example='Hello, world!')
        }))  # Embed the data model
        
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
                - file_path (str): path of the audio file.
            """
            try:
                # Parse the request body
                data = request.json

                file_path = data.get('file_path')
                url = data.get('audio_url')

                if not file_path and not url:
                    return {
                        'status': 'error',
                        'error': 'file_path or url is required.',
                        'data': None
                    }, 400
                
                if file_path and url:
                    return {'status': 'error', 'error': 'Provide either file_path or url, not both.', 'data': None}, 400

                if url:
                    audio_result = audio_service.extract_audio(url,start_time_ms=0)

                    if isinstance(audio_result, dict) and 'error' in audio_result:
                        # If there was an error extracting the audio, return it
                        return {
                            'status':'error',
                            'error': audio_result["error"], 
                            'data':None
                        }
                    
                    if debug:
                        logger.debug(f"[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [audio_result]", audio_result)
                    
                    # Parse the audio segment details
                    file_path = audio_result['audio_path']

                should_cleanup = url is not None

                # Call the service to transcribe the audio file
                result = service.transcribe(audio_file_path = file_path)

                if should_cleanup:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass

                if 'error' in result:
                    print(f'error : {result}')
                    return {
                        'status': 'error',
                        'error': result['error'],
                        'data': None
                    }, 500 
                
                return {
                    'status': 'success',
                    'data': {
                        'transcription': result['transcription'],
                        'chunks' : result['chunks']
                    }
                }, 200
            
            except Exception as e:
                logger.error(f"[error] [Route Layer] [TranscriptionTranscribe] [post] An error occurred: {str(e)}")
                # print(f"[error] [Route Layer] [TranscriptionTranscribe] [post] An error occurred: {str(e)}")
                print(f'error : {e}')
                return {
                    'status': 'error',
                    "error": 'An unexpected error occurred while processing the request.', # Generic error message                    
                    'data': None,
                }, 500
            
    # Define the model for the transcript chunks extraction request body
    transcription_chunks_request_model = api.model('TranscriptionChunksRequestModel', {
        'file_path': fields.String(required=True, description='Path of the audio file.', example='audio.mp3')
    })

    transcription_chunks_bad_request_model = api.model('TranscriptionChunksBadRequestModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='file_path is required'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    transcription_chunks_internal_server_error_model = api.model('TranscriptionChunksInternalServerErrorModel', {
        'status': fields.String(required=True, description='The status of the response', example='error'),
        'error': fields.String(required=True, description='The error message', example='An unexpected error occurred during transcription.'),
        'data': fields.Raw(description='Data field will be null for error responses', example=None)
    })

    transcription_chunks_success_model = api.model('TranscriptionChunksSuccessModel', {
        'status': fields.String(required=True, description='The status of the response', example='success'),
        'data': fields.Nested(api.model('TranscriptionChunksDataModel', {
            'transcription': fields.String(required=True, description='Extracted transcript.', example='Hello, world!'),
            'chunks': fields.List(fields.String, required=True, description='Extracted chunks.', example= [{'timestamp': (0.0, 3.0), 'text': " And there's a voice and it's a little quiet voice that goes jump."}, {'timestamp': (3.0, 5.0), 'text': " It's the same voice."}])
        }))  # Embed the data model
    })

    # Define the endpoint for the Transcribe an audio file and return chunks.
    @api.route('/chunks') 
    class TranscriptionChunks(Resource):
        @api.doc(description="Transcribe an audio file and return chunks.")
        @api.expect(transcription_chunks_request_model)  # Use the model for request validation
        @api.response(200, 'Success', transcription_chunks_success_model)
        @api.response(400, 'Bad Request', transcription_chunks_bad_request_model)
        @api.response(500, 'Internal Server Error', transcription_chunks_internal_server_error_model)
        def post(self):
            """
            Endpoint to extract transcript from an audio file and return chunks.
                - file_path (str): path of the audio file.
            """
            try:
                # Parse the request body
                data = request.json

                file_path = data.get('file_path')

                if not file_path:
                    return {
                        'status': 'error',
                        'error': 'file_path is required.',
                        'data': None
                    }, 400
    
                # Call the service to transcribe the audio file
                result = service.transcribe(audio_file_path = file_path)

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error': result['error'],
                        'data': None
                    }, 500
                
                
                # Return the transcribed text and chunks
                return {
                    'status': 'success',
                    'data': {
                        'transcription': result['transcription'],
                        'chunks': result['chunks']
                    }
                }, 200
            
            except Exception as e:
                logger.error(f"[error] [Route Layer] [TranscriptionChunks] [post] An error occurred: {str(e)}")
                # print(f"[error] [Route Layer] [TranscriptionChunks] [post] An error occurred: {str(e)}")
                return {
                    'status': 'error',
                    "error": 'An unexpected error occurred while processing the request.', # Generic error message
                    'data': None,
                }, 500

# Define the namespace
api = Namespace('Transcript', description="Transcript operations")

# Register the routes
register_routes(api)