"""
This module contains the service layer for extracting audio segments.
"""
import os
import uuid

from app.config import Config
from app.utils.logger import logger
from app.utils.ffmpeg_audio import FFmpegError, extract_audio_segment, get_duration_ms

# Data layer for fetching audio files
from app.data.audio_data import AudioDataLayer

config = Config().config  # Load the configuration

DEFAULT_OUTPUT_SAMPLE_RATE = 16000


class AudioService:
    def __init__(self, static_folder="static/audio"):
        self.debug = config.get('debug')
        audio_config = config.get('audio') or {}
        self.output_sample_rate = audio_config.get(
            'output_sample_rate',
            DEFAULT_OUTPUT_SAMPLE_RATE,
        )

        self.audio_data_layer = AudioDataLayer(config)
        self.static_folder = static_folder

    def extract_audio(self, url: str, start_time_ms: int, end_time_ms: int = None, user_id: str = None):
        """
        Extract a segment from the audio/video file using ffmpeg.
        :param url: URL or local file path to the audio file.
        :param start_time_ms: Start time of the segment to extract (in milliseconds).
        :param end_time_ms: End time of the segment to extract (in milliseconds).
        :param user_id: (Optional) User ID for creating user-specific subdirectories.
        :return: Path to the saved audio file or error message
        """
        source = None
        try:
            if not isinstance(start_time_ms, (int, float)) or start_time_ms < 0:
                return {
                    'error': 'Start time must be a non-negative number.'
                }

            if end_time_ms is not None and end_time_ms < start_time_ms:
                return {
                    'error': 'End time must not be less than start time.'
                }

            source = self.audio_data_layer.resolve_audio_source(url)
            if isinstance(source, dict) and 'error' in source:
                return {
                    'error': source['error']
                }

            source_path = source['path']
            duration_ms = get_duration_ms(source_path)

            resolved_end_time_ms = end_time_ms
            if resolved_end_time_ms is None or resolved_end_time_ms > duration_ms:
                resolved_end_time_ms = duration_ms

            if resolved_end_time_ms < start_time_ms:
                return {
                    'error': 'End time must not be less than start time.'
                }

            output_path = self._build_output_path(user_id)
            extract_audio_segment(
                input_path=source_path,
                output_path=output_path,
                start_time_ms=start_time_ms,
                end_time_ms=resolved_end_time_ms,
                sample_rate=self.output_sample_rate,
            )

            return {
                "audio_path": output_path,
                "start_time_ms": start_time_ms,
                "end_time_ms": resolved_end_time_ms,
            }

        except FFmpegError as e:
            logger.error(
                f"[error] [Service Layer] [AudioService] [extract_audio] FFmpeg error: {str(e)}"
            )
            return {'error': 'An unexpected error occurred while processing the request.'}
        except Exception as e:
            logger.error(
                f"[error] [Service Layer] [AudioService] [extract_audio] "
                f"An error occurred during the audio extraction: {str(e)}"
            )
            return {'error': 'An unexpected error occurred while processing the request.'}
        finally:
            if isinstance(source, dict) and source.get('is_temporary') and source.get('path'):
                self._safe_remove(source['path'])

    def _build_output_path(self, user_id: str = None) -> str:
        """
        Build a unique WAV output path under the static audio folder.
        """
        unique_filename = f"{str(uuid.uuid4())}_audio.wav"

        if user_id:
            user_folder = os.path.join(self.static_folder, user_id).replace("\\", "/")
            os.makedirs(user_folder, exist_ok=True)
            return f"{user_folder}/{unique_filename}"

        os.makedirs(self.static_folder, exist_ok=True)
        return f"{self.static_folder}/{unique_filename}"

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
