"""
This module defines the PyannoteSpeakerDiarization model wrapper.
"""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment


class PyannoteSpeakerDiarization:
    def __init__(self, config: dict) -> None:
        """
        Initialize the pyannote speaker diarization pipeline lazily.
        :param config: The configuration object containing speaker diarization settings.
        """
        self.debug = config.get('debug')

        self.config = config.get('speaker_diarization', {})
        self.local_model_path = self.config.get('local_model_path')
        self.model_name = self.config.get('model_name', 'pyannote/speaker-diarization-community-1')
        self.token_env = self.config.get('token_env', 'HUGGINGFACE_TOKEN')

        self.pipeline = None

    @staticmethod
    def _resolve_local_model_path(local_model_path: str | None) -> Path | None:
        """
        Resolve a configured local model path relative to the repository root when needed.
        """
        if not local_model_path:
            return None

        path = Path(local_model_path)
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / path

        if path.is_dir():
            config_path = path / "config.yaml"
            if config_path.exists():
                path = config_path

        return path.resolve()

    def _load_pipeline(self):
        """
        Lazy-load the pyannote pipeline so app startup does not fail when diarization is unavailable.
        """
        if self.pipeline is not None:
            return self.pipeline

        self._patch_torchaudio_for_speechbrain()

        try:
            from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError(
                "pyannote.audio is not installed or could not be imported."
            ) from exc

        local_model_path = self._resolve_local_model_path(self.local_model_path)
        if local_model_path and local_model_path.exists():
            with self._legacy_torch_load():
                self.pipeline = Pipeline.from_pretrained(local_model_path)
            return self.pipeline

        token = os.getenv(self.token_env)
        if not token:
            if local_model_path:
                raise RuntimeError(
                    f"Local speaker diarization model not found at: {local_model_path}. "
                    f"Either restore that directory or set {self.token_env} for Hugging Face loading."
                )
            raise RuntimeError(
                f"Missing Hugging Face token for speaker diarization. "
                f"Set the {self.token_env} environment variable."
            )

        try:
            self.pipeline = Pipeline.from_pretrained(self.model_name, token=token)
        except TypeError:
            # Backward compatibility with older pyannote releases.
            self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=token)

        return self.pipeline

    @staticmethod
    @contextmanager
    def _legacy_torch_load():
        """
        Newer torch defaults to weights_only=True, but the trusted local
        pyannote checkpoints still require full checkpoint deserialization.
        """
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if kwargs.get("weights_only") is None:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load

    @staticmethod
    def _patch_torchaudio_for_speechbrain() -> None:
        """
        SpeechBrain still expects torchaudio.list_audio_backends on import, but
        newer torchaudio builds can omit it on Windows. Provide a minimal shim
        so local pyannote loading can proceed.
        """
        try:
            import torchaudio
        except Exception:
            return

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    @staticmethod
    def _load_audio_input(audio_file_path: str) -> dict:
        """
        Preload audio into memory so pyannote does not rely on torchaudio/torchcodec
        decoding on this machine.
        """
        audio = AudioSegment.from_file(audio_file_path).set_sample_width(2)

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).T
        else:
            samples = samples.reshape((1, -1))

        waveform = torch.from_numpy(samples / 32768.0)
        return {
            "waveform": waveform,
            "sample_rate": audio.frame_rate,
        }

    def __call__(self, audio_file_path: str) -> list:
        """
        Perform speaker diarization on the given audio file.
        :param audio_file_path: Path to the audio file.
        :return: Raw speaker diarization segments.
        """
        pipeline = self._load_pipeline()
        output = pipeline(self._load_audio_input(audio_file_path))

        diarization = getattr(output, 'exclusive_speaker_diarization', None)
        if diarization is None:
            diarization = getattr(output, 'speaker_diarization', output)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue

            segments.append({
                'speaker': str(speaker),
                'start': start,
                'end': end,
            })

        return segments
