# Model Layer
from app.models.whisper_model import WhisperTranscript

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
            print(f"[error] [Data Layer] [TranscriptDataLayer] [transcribe] An error occurred during transcription: {str(e)}")
            return {'error': f'An error occurred during transcription: {str(e)}'}
    
# if __name__ == "__main__":
#     config = {
#         'debug': True,
#         'transcription': {
#             'default_model': "whisper",  # Specify the default transcription model (e.g., whisper, another_model)
#             'whisper': {
#                 'model_size': 'tiny',
#                 'device': 'cpu',
#                 'chunk_length_s': 30
#             }
#         }
#     }
#     print("config",config)
#     transcript_data = TranscriptDataLayer(config)
#     print("transcript_data",transcript_data)
#     print(transcript_data.transcribe("./samples/sample_1.mp3"))

# #  Run:
# #  python -m app.models.bertweet_model