"""
This Module is responsible for handling the transcription data layer.
"""

# Model Layer
from app.models.whisper_model import WhisperTranscript

from app.utils.logger import logger

class TranscriptDataLayer:
    def __init__(self, config: dict):
        """
        Initialize the Transcript Data Layer.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('transcription')
        self.default_model = self.config.get('default_model')

        # Initialize the appropriate model based on the configuration
        if self.default_model == "whisper":
            self.model = WhisperTranscript(config)
        # elif self.default_model == "another_model":
        #     self.model = AnotherModel(config)  # Replace with your other model class
        else:
            raise ValueError(f"Unsupported transcription model: {self.default_model}")


    def transcribe(self, audio_file_path: str) -> tuple:
        """
        Process the audio file and return the transcription.
        :param audio_file_path: Path to the audio file.
        :return: Transcribed text and chunks.
        """
        try:
            transcription, chunks = self.model(audio_file_path)
            return {
                'transcription': transcription,
                'chunks': chunks
            }
        
        except Exception as e:
            # Handle any errors that occur during transcription
            logger.error(f"[error] [Data Layer] [TranscriptDataLayer] [transcribe] An error occurred during transcription: {str(e)}")
            # print(f"[error] [Data Layer] [TranscriptDataLayer] [transcribe] An error occurred during transcription: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}  # Generic error message
    
# if __name__ == "__main__":
#     config = {
#         'debug': True,
#         # Transcription Configuration
#         'transcription':{
#             'default_model': "whisper",  # Specify the default transcription model (e.g., whisper, another_model)
#             'whisper':{                  # Whisper-specific configuration
#                 'model_size': "base" ,   # Choose between tiny, base, small, medium, large
#                 'device': 'cpu'  ,       # -1 for CPU, or the GPU device index (e.g., 0)
#                 'chunk_length_s': 30 
#             }                 
#             # 'another_model':{          # Placeholder for another transcription model's configuration
#                 #   'api_key': "your_api_key"
#                 #   'endpoint': "https://api.example.com/transcribe"
#             # }
#         }
#     }
    # print("config",config)
    # transcript_data = TranscriptDataLayer(config)
    # print("transcript_data",transcript_data)

    # audio_file_path = "./samples/sample_1.mp3"
    # transcript = transcript_data.transcribe(audio_file_path)
    # print("transcript",transcript)

# #  Run:
# #  python -m app.data.transcript_data