import torch
import torch.nn as nn
from transformers import pipeline


class WhisperTranscript(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Initialize the Whisper model pipeline for transcription.

        Args:
            config (dict): Configuration dictionary.
        """
        self.debug = config.get('debug')

        self.config = config.get('transcription').get('whisper')
        self.model_size = self.config.get('model_size')
        self.device = self.config.get('device')
        self.chunk_length_s = self.config.get('chunk_length_s')

        model_name = f"openai/whisper-{self.model_size}"  # Dynamically set model size

        super(WhisperTranscript, self).__init__()
        # Initialize the pipeline
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=self.chunk_length_s,
            device=self.device,  # Use the device from the configuration
        )


    def forward(self, audio_file: str) -> tuple:
        """
        Perform transcription on the given audio file.

        Args:
            audio_file (str): Path to the audio file.

        Returns:
            tuple: Transcribed text and timestamped chunks.
        """
        # Forward pass
        out = self.pipeline(audio_file, return_timestamps=True)
        
        return out["text"], out["chunks"]
    
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
#     model = WhisperTranscript(config)
#     print("model",model)

#     audio_file = "./samples/sample_1.mp3"
#     print("audio_file",audio_file)
#     transcription, chunks = model(audio_file)
#     print("transcription",transcription)
#     print("chunks",chunks)