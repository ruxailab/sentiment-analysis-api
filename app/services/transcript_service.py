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
            # Catch any other exceptions
            logger.error(f"[error] [Service Layer] [TranscriptService] [transcribe] An error occurred during transcription: {str(e)}")
            # print(f"[error] [Service Layer] [TranscriptService] [transcribe] An error occurred during transcription: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}  # Generic error message
        
        
# if __name__ == "__main__":
#     transcript_service = TranscriptService()
#     print("transcript_service",transcript_service)

#     # Normal Case
#     result = transcript_service.transcribe("./samples/sample_1.mp3")
#     print("result",result)

    # # File doesn't exist
    # result = transcript_service.transcribe("./samples/non_exist_file.mp3")
    # print("result",result)

# #  Run:
# #  python -m app.services.transcript_service