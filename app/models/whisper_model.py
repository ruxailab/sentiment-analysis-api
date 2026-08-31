"""
This module defines the WhisperTranscript class for transcribing audio files
using faster-whisper (CTranslate2).
"""
import threading

from faster_whisper import WhisperModel


class WhisperTranscript:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: dict = None):
        """Return the singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: dict) -> None:
        """
        Initialize the faster-whisper model for transcription.
        Heavy model weights are loaded only once for the process lifetime.
        :param config: The configuration object containing model and device info.
        """
        if self._initialized:
            return

        self.debug = config.get('debug')

        self.config = config.get('transcription').get('whisper')
        self.model_size = self.config.get('model_size', 'base')
        self.device = self.config.get('device', 'cpu')
        self.compute_type = self.config.get('compute_type', 'int8')
        self.beam_size = self.config.get('beam_size', 5)
        self.vad_filter = self.config.get('vad_filter', True)
        self.language = self.config.get('language')  # None => auto-detect

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._initialized = True

    def __call__(self, audio_file: str) -> tuple:
        """
        Perform transcription on the given audio file.

        Args:
            audio_file (str): Path to the audio file.

        Returns:
            tuple: Transcribed text and timestamped chunks.
        """
        segments_iter, _info = self.model.transcribe(
            audio_file,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            language=self.language,
        )
        segments = list(segments_iter)

        chunks = [
            {
                "timestamp": (segment.start, segment.end),
                "text": segment.text,
            }
            for segment in segments
        ]
        transcription = "".join(segment.text for segment in segments)

        return transcription, chunks
