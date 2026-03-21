"""
This module contains the service layer for transcribing audio files.
"""
import os
from app.config import Config

from app.utils.logger import logger

# Data Layer for fetching and processing transcripts
from app.data.transcript_data import TranscriptDataLayer

config = Config().config # Load the configuration

class TranscriptService:
    def __init__(self):
        self.debug = config.get('debug')

        self.transcript_data_layer = TranscriptDataLayer(config)

    def transcribe(self, audio_file_path: str) -> tuple:
        """
        Transcribe the given audio file.
        :param audio_file_path: Path to the audio file
        :return: Transcribed text and chunks
        """
        try:
            # Check if the audio file exists
            if not os.path.exists(audio_file_path) or not os.path.isfile(audio_file_path):
                return {
                    'error': f'Audio file not found: {audio_file_path}'
                }

            result = self.transcript_data_layer.transcribe(audio_file_path)

            if isinstance(result, dict) and 'error' in result:
                return {
                    'error': result['error']
                }

            # Return the transcribed text and chunks
            return {
                'transcription': result['transcription'],
                'chunks': result['chunks']
            }

        except Exception as e:
            logger.error(
                "[Service Layer] [TranscriptService] [transcribe] An error occurred: %s",
                str(e)
            )
            return {'error': 'An unexpected error occurred while processing the request.'}
