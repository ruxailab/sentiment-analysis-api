import torch
import torch.nn as nn
from transformers import pipeline

class WhisperTranscript(nn.Module):
    def __init__(self) -> None:
        super(WhisperTranscript, self).__init__()

        model_name = "openai/whisper-tiny"

        # Pipeline        
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device='cpu',
        )

    def forward(self, audio_file:str)->tuple:
        # Forward pass
        out = self.pipeline(audio_file, return_timestamps=True)

        return out["text"],out["chunks"]

