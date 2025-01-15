import os
from app.config import Config

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
            if not os.path.exists(audio_file_path):
                return {'error': f'Audio file not found: {audio_file_path}'}
            
            result = self.transcript_data_layer.transcribe(audio_file_path)

            if isinstance(result, dict) and 'error' in result:
                return {
                    'error': f'An error occurred during transcription: {result["error"]}'
                }

            # Return the transcribed text and chunks
            return {
                'transcription': result['transcription'],
                'chunks': result['chunks']
            }
        
        except Exception as e:
            print(f"[error] [Service Layer] [TranscriptService] [transcribe] An error occurred during transcription: {str(e)}")
            return {'error': f'An error occurred during transcription: {str(e)}'}
        
# if __name__ == "__main__":
#     transcript_service = TranscriptService()

    # result = transcript_service.transcribe("./samples/sample_1.mp3")
    # print("result",result)

    # result = transcript_service.transcribe("./samples/non_exist_file.mp3")
    # print("result",result)

# #  Run:
# #  python -m app.services.transcript_service