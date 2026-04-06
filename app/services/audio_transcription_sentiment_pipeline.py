import os

from app.config import Config

from app.utils.logger import logger

# Services
from app.services.audio_service import AudioService
from app.services.transcript_service import TranscriptService
from app.services.sentiment_service import SentimentService

config = Config().config # Load the configuration


from pydantic import BaseModel
from typing import List, Union

class TranscriptionChunk(BaseModel):
    timestamp: List[int]  # [start_time_ms, end_time_ms]
    text: str  # Text from the chunk
    label: Union[str, None] = None  # Sentiment label (optional)
    confidence: Union[float, None] = None  # Sentiment confidence score (optional)

class AudioTranscriptionSentimentResult(BaseModel):
    audio_path: str  # Path to the extracted audio segment
    start_time_ms: int  # Start time of the segment (in milliseconds)
    end_time_ms: int  # End time of the segment (in milliseconds)
    transcription: str  # Full transcription of the audio segment
    utterances_sentiment: List[TranscriptionChunk]  # Sentiment analysis for each chunk

class ErrorResponse(BaseModel):
    error: str  # Error message describing what went wrong

# Union type to handle both successful and error responses
ProcessResponse = Union[AudioTranscriptionSentimentResult, ErrorResponse]


class AudioTranscriptionSentimentPipeline:
    def __init__(self):
        self.debug = config.get('debug')

        self.config = config.get('audio_transcription_sentiment_pipeline')
        self.remove_audio = self.config.get('remove_audio')

        self.audio_service = AudioService()
        self.transcript_service = TranscriptService()
        self.sentiment_service = SentimentService()

    def process(self, url: str, start_time_ms: int, end_time_ms: int = None, user_id: str = None) -> ProcessResponse:
        """
        Process the Video/Audio file by extracting a segment, transcribing it, and performing sentiment analysis.
        :param url: URL or local file path to the audio file.
        :param start_time_ms: Start time of the segment to extract (in milliseconds).
        :param end_time_ms: End time of the segment to extract (in milliseconds).
        :param user_id: (Optional) User ID for creating user-specific subdirectories
        :return: Transcription, sentiment analysis, and audio segment details
        """
        try:
            # Step(1) Extract the audio segment
            audio_result = self.audio_service.extract_audio(url, start_time_ms, end_time_ms, user_id)

            if isinstance(audio_result, dict) and 'error' in audio_result:
                return {
                    'error': audio_result["error"]
                }

            if self.debug:
                logger.debug(
                    "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] audio_result: %s",
                    audio_result
                )

            # Parse the audio segment details
            audio_path = audio_result['audio_path']
            start_time_ms = audio_result['start_time_ms']
            end_time_ms = audio_result['end_time_ms']

            # Step(2) Transcribe the audio segment
            transcription_result = self.transcript_service.transcribe(audio_path)

            if isinstance(transcription_result, dict) and 'error' in transcription_result:
                return {
                    'error': transcription_result['error']
                }

            if self.debug:
                logger.debug(
                    "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] transcription_result: %s",
                    transcription_result
                )

            # Parse the transcription details
            transcription = transcription_result['transcription']
            chunks = transcription_result['chunks']

            # Remove the audio file after processing
            if self.remove_audio:
                logger.debug(
                    "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] Removing audio file: %s",
                    audio_path
                )
                os.remove(audio_path)

            # Step(3) Perform sentiment analysis per chunk
            for chunk in chunks:
                timestamp = chunk['timestamp']
                text = chunk['text']

                sentiment_result = self.sentiment_service.analyze(text)
                if isinstance(sentiment_result, dict) and 'error' in sentiment_result:
                    logger.error(
                        "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] sentiment error: %s",
                        sentiment_result
                    )
                    chunk['error'] = sentiment_result['error']
                    continue

                if self.debug:
                    logger.debug(
                        "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] sentiment_result: %s",
                        sentiment_result
                    )

                chunk['label'] = sentiment_result['label']
                chunk['confidence'] = sentiment_result['confidence']

            return {
                'audio_path': audio_path,
                'start_time_ms': start_time_ms,
                'end_time_ms': end_time_ms,
                'transcription': transcription,
                'utterances_sentiment': chunks,
            }
        except Exception as e:
            logger.error(
                "[Service Layer] [AudioTranscriptionSentimentPipeline] [process] An error occurred: %s",
                str(e)
            )
            return {'error': 'An unexpected error occurred while processing the request.'}
