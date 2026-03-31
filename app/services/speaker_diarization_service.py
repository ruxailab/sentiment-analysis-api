"""
This module contains the service layer for speaker diarization.
"""
import os

from app.config import Config
from app.data.speaker_diarization_data import SpeakerDiarizationDataLayer
from app.utils.logger import logger


class SpeakerDiarizationService:
    def __init__(self, config: dict | None = None):
        self.root_config = config or Config().config
        self.debug = self.root_config.get('debug')

        self.config = self.root_config.get('speaker_diarization', {})
        self.enabled = self.config.get('enabled', True)

        self.speaker_diarization_data_layer = SpeakerDiarizationDataLayer(self.root_config)

    def diarize(self, audio_file_path: str) -> dict:
        """
        Perform speaker diarization on the given audio file.
        :param audio_file_path: Path to the audio file.
        :return: Speaker diarization segments or an error.
        """
        try:
            if not self.enabled:
                return {'segments': []}

            if not os.path.exists(audio_file_path) or not os.path.isfile(audio_file_path):
                return {'error': f'Audio file not found: {audio_file_path}'}

            result = self.speaker_diarization_data_layer.diarize(audio_file_path)
            if isinstance(result, dict) and 'error' in result:
                return {
                    'error': result['error']
                }

            return {
                'segments': result['segments']
            }

        except Exception as e:
            logger.warning(
                f"[warning] [Service Layer] [SpeakerDiarizationService] [diarize] "
                f"Speaker diarization unavailable: {str(e)}"
            )
            return {'error': 'Speaker diarization is unavailable.'}
