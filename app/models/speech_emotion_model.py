"""
Speech Emotion Model
Uses HuggingFace audio classification pipeline
"""

import torch
import torch.nn as nn
import logging
from transformers import pipeline
from typing import Dict , Any , List


class SpeechEmotionModel(nn.Module):

    def __init__(self, config: dict) -> None:
        super(SpeechEmotionModel, self).__init__()

        self.debug = config.get('debug')

        # added a null check
        emotion_config = config.get('speech_emotion')
        if not emotion_config:
            raise ValueError("'speech_emotion' not found in config")

        self.config = emotion_config.get('default')
        if not self.config:
            raise ValueError("'default' speech_emotion config missing")

        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        # Use logging for structured and production-ready output instead of print
        logger = logging.getLogger(__name__)
        logger.info(f"Loading SpeechEmotionModel: {self.model_name}")

        print(f"Loading Speech Emotion Model: {self.model_name}")

        # Initializing HuggingFace pipeline for audio classification.
        # This abstracts preprocessing, model inference, and postprocessing
        self.pipeline = pipeline(
            task="audio-classification",
            model=self.model_name,
            device=self.device
        )

    def forward(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform emotion classification
        """

        # Running inference using Huggingface pipelines.
        # The pipeline internally handles feature extraction + model prediction.
        outputs = self.pipeline(audio_path)

        # Ensure output is valid and non-empty.
        # This prevents runtime errors in case of unexpected model behavior/ safe handling.
        if not isinstance(outputs, list) or len(outputs) == 0:
            return {
                "emotion": {
                    "label": "unknown",
                    "score": 0.0
                }
            }
        
# Most pipelines return sorted results, so index 0 is the best prediction
        top = outputs[0]

# Return structured output consistent with other models in the system.
# it ensures that the easy integration  with downstream pipelines and API's.
        return {
            "emotion": {
                "label": top.get("label", "unknown"),
                "score": float(top.get("score", 0.0))
            }
        }