"""
This module defines the WhisperTranscript class, which is a PyTorch model for transcribing audio files using the OpenAI Whisper model.
"""
import torch
import torch.nn as nn

from transformers import pipeline
from typing import Dict, Any


class WhisperTranscript(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Initialize the Whisper model pipeline for transcription.
        :param config: The configuration object containing model and device info.      
        """
        self.debug = config.get('debug')
        transcription_config = config.get('transcription')
        if not transcription_config:
            raise ValueError("'transcription' not found in config")
        
        self.config = transcription_config.get('whisper')
        if not self.config:
            raise ValueError("'whisper' not found in transcription config")
            
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


    def forward(self, audio_file: str) -> Dict[str, Any]:
        """
        Perform transcription on the given audio file.

        Args:
            audio_file (str): Path to the audio file.

        Returns:
            Dict: Transcribed text and timestamped chunks.
        """
        # Forward pass
        out = self.pipeline(audio_file, return_timestamps=True)

         # Initialize to avoid "possibly unbound" error
        text = ""
        chunks = []   
        
        
        # Extract text and chunks safely
        if isinstance(out, dict):
            text = out.get("text", "")
            chunks = out.get("chunks", [])
        else:
            # For dict-like objects (not necessarily dict type)
             text = getattr(out, "text", "")
             chunks = getattr(out, "chunks", [])
            
        return {
               "text": text,
               "chunks": chunks
}
  
    
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
#     print("config",config)
#     model = WhisperTranscript(config)
#     print("model",model)

#     audio_file = "./samples/sample_1.mp3"
#     print("audio_file",audio_file)
#     transcription, chunks = model(audio_file)
#     print("transcription",transcription)
#     print("chunks",chunks)

# #  Run:
# # python -m app.models.whisper_model
    