"""
Local-only audio -> speaker -> transcript -> sentiment pipeline used by test.py.
This keeps the original Flask pipeline untouched.
"""
import copy
import os

from app.config import Config
from app.services.audio_service import AudioService
from app.services.sentiment_service import SentimentService
from app.services.speaker_diarization_service import SpeakerDiarizationService
from app.services.transcript_service import TranscriptService
from app.utils.logger import logger


class LocalAudioTranscriptionSentimentPipeline:
    def __init__(self, local_model_path: str = "speaker-diarization-community-1"):
        root_config = copy.deepcopy(Config().config)
        speaker_config = root_config.setdefault("speaker_diarization", {})
        speaker_config.setdefault("enabled", True)
        speaker_config.setdefault("backend", "pyannote")
        speaker_config.setdefault("local_model_path", local_model_path)
        speaker_config.setdefault("model_name", "pyannote/speaker-diarization-community-1")
        speaker_config.setdefault("token_env", "HUGGINGFACE_TOKEN")

        self.debug = root_config.get("debug")
        self.remove_audio = root_config.get("audio_transcription_sentiment_pipeline", {}).get("remove_audio", False)

        self.audio_service = AudioService()
        self.speaker_diarization_service = SpeakerDiarizationService(root_config)
        self.transcript_service = TranscriptService()
        self.sentiment_service = SentimentService()

    @staticmethod
    def _resolve_chunk_speaker(chunk_timestamp, speaker_segments: list) -> str:
        if not chunk_timestamp or len(chunk_timestamp) != 2:
            return "UNKNOWN"

        chunk_start = float(chunk_timestamp[0])
        chunk_end = float(chunk_timestamp[1])

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        best_start = float("inf")

        for segment in speaker_segments:
            overlap = max(
                0.0,
                min(chunk_end, float(segment["end"])) - max(chunk_start, float(segment["start"]))
            )
            segment_start = float(segment["start"])

            if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and segment_start < best_start):
                best_overlap = overlap
                best_speaker = segment["speaker"]
                best_start = segment_start

        return best_speaker if best_overlap > 0 else "UNKNOWN"

    def _assign_speakers_to_chunks(self, chunks: list, speaker_segments: list) -> list:
        for chunk in chunks:
            chunk["speaker"] = self._resolve_chunk_speaker(chunk.get("timestamp"), speaker_segments)
        return chunks

    def process(self, url: str, start_time_ms: int, end_time_ms: int = None, user_id: str = None) -> dict:
        try:
            audio_result = self.audio_service.extract_audio(url, start_time_ms, end_time_ms, user_id)
            if isinstance(audio_result, dict) and "error" in audio_result:
                return {"error": audio_result["error"]}

            audio_path = audio_result["audio_path"]
            start_time_ms = audio_result["start_time_ms"]
            end_time_ms = audio_result["end_time_ms"]

            speaker_result = self.speaker_diarization_service.diarize(audio_path)
            speaker_segments = []
            if isinstance(speaker_result, dict) and "error" in speaker_result:
                logger.warning(
                    "[warning] [LocalAudioTranscriptionSentimentPipeline] [process] "
                    "Speaker diarization unavailable. Falling back to UNKNOWN speaker labels. "
                    f"Details: {speaker_result['error']}"
                )
            else:
                speaker_segments = speaker_result.get("segments", [])

            transcription_result = self.transcript_service.transcribe(audio_path)
            if isinstance(transcription_result, dict) and "error" in transcription_result:
                return {"error": transcription_result["error"]}

            transcription = transcription_result["transcription"]
            chunks = self._assign_speakers_to_chunks(transcription_result["chunks"], speaker_segments)

            if self.remove_audio:
                os.remove(audio_path)

            for chunk in chunks:
                sentiment_result = self.sentiment_service.analyze(chunk["text"])
                if isinstance(sentiment_result, dict) and "error" in sentiment_result:
                    chunk["error"] = sentiment_result["error"]
                    continue

                chunk["label"] = sentiment_result["label"]
                chunk["confidence"] = sentiment_result["confidence"]

            return {
                "audio_path": audio_path,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "transcription": transcription,
                "utterances_sentiment": chunks,
            }
        except Exception as exc:
            logger.error(
                "[error] [LocalAudioTranscriptionSentimentPipeline] [process] "
                f"An error occurred during processing: {str(exc)}"
            )
            return {"error": "An unexpected error occurred while processing the request."}
