"""
This module is responsible for handling the speaker diarization data layer.
"""
from app.utils.logger import logger


class SpeakerDiarizationDataLayer:
    def __init__(self, config: dict):
        """
        Initialize the Speaker Diarization Data Layer.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('speaker_diarization', {})
        self.default_backend = self.config.get('backend', 'pyannote')

        self.root_config = config
        self.model = None

    def _get_model(self):
        if self.model is not None:
            return self.model

        if self.default_backend == "pyannote":
            from app.models.speaker_diarization_model import PyannoteSpeakerDiarization

            self.model = PyannoteSpeakerDiarization(self.root_config)
            return self.model

        raise ValueError(f"Unsupported speaker diarization backend: {self.default_backend}")

    @staticmethod
    def _normalize_segments(segments: list) -> list:
        """
        Normalize backend speaker labels into stable speaker_1..N labels.
        """
        normalized_segments = []
        label_mapping = {}

        for segment in sorted(segments, key=lambda item: (item['start'], item['end'], item['speaker'])):
            speaker = segment['speaker']
            if speaker not in label_mapping:
                label_mapping[speaker] = f"speaker_{len(label_mapping) + 1}"

            normalized_segments.append({
                'speaker': label_mapping[speaker],
                'start': float(segment['start']),
                'end': float(segment['end']),
            })

        return normalized_segments

    def diarize(self, audio_file_path: str) -> dict:
        """
        Process the audio file and return speaker diarization segments.
        :param audio_file_path: Path to the audio file.
        :return: Speaker diarization segments.
        """
        try:
            raw_segments = self._get_model()(audio_file_path)
            return {
                'segments': self._normalize_segments(raw_segments)
            }

        except Exception as e:
            logger.warning(
                f"[warning] [Data Layer] [SpeakerDiarizationDataLayer] [diarize] "
                f"Speaker diarization unavailable: {str(e)}"
            )
            return {'error': 'Speaker diarization is unavailable.'}
