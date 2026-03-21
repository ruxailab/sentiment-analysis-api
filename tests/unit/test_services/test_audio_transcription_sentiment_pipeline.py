"""
This Module contains the unit tests for the AudioTranscriptionSentimentPipeline class.
"""

import pytest
from unittest.mock import MagicMock, patch

# Service to be tested
from app.services.audio_transcription_sentiment_pipeline import AudioTranscriptionSentimentPipeline

class TestAudioTranscriptionSentimentPipeline:
    @pytest.fixture
    def audio_transcription_sentiment_pipeline(self):
        """
        Fixture to set up AudioTranscriptionSentimentPipeline instance for testing.
        remove_audio is False by default — tests that require True set it explicitly.
        """
        pipeline = AudioTranscriptionSentimentPipeline()
        pipeline.remove_audio = False
        return pipeline

    # Grouped tests for the `process` method
    class TestProcess:
        def setup_method(self):
            """Set up before each test for Process class."""
            self.args = {
                "url": "https://example.com/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20,
                "user_id": "user123"
            }

        # Mocking Services
        @pytest.fixture
        def mock_audio_service__extract_audio(self):
            with patch("app.services.audio_transcription_sentiment_pipeline.AudioService.extract_audio") as mock:
                yield mock

        @pytest.fixture
        def mock_transcript_service__transcribe(self):
            with patch("app.services.audio_transcription_sentiment_pipeline.TranscriptService.transcribe") as mock:
                yield mock

        @pytest.fixture
        def mock_sentiment_service__analyze(self):
            with patch("app.services.audio_transcription_sentiment_pipeline.SentimentService.analyze") as mock:
                yield mock

        @pytest.fixture
        def mock_os__remove(self):
            with patch("app.services.audio_transcription_sentiment_pipeline.os.remove") as mock:
                mock.return_value = True
                yield mock

        # --- Existing tests (unchanged) ---

        def test_process__extract_audio_failure(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {"error": "Mocked error message"}

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {'error': "Mocked error message"}
            mock_audio_service__extract_audio.assert_called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )

        def test_process__extract_audio_exception(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio):
            payload = self.args.copy()
            mock_audio_service__extract_audio.side_effect = Exception("Mocked exception")

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {'error': "An unexpected error occurred while processing the request."}
            mock_audio_service__extract_audio.assert_called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )

        def test_process__transcribe_audio_failure(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio, mock_transcript_service__transcribe):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {"error": "Mocked error message"}

            result = audio_transcription_sentiment_pipeline.process(**self.args)

            assert result == {'error': "Mocked error message"}
            mock_audio_service__extract_audio.assert_called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )
            mock_transcript_service__transcribe.assert_called_once_with("/path/to/audio.mp3")

        def test_process__transcribe_audio_exception(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio, mock_transcript_service__transcribe):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.side_effect = Exception("Mocked exception")

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {'error': "An unexpected error occurred while processing the request."}
            mock_audio_service__extract_audio.assert_called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )
            mock_transcript_service__transcribe.assert_called_once_with("/path/to/audio.mp3")

        def test_process__sentiment_analysis_failure(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze
        ):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }
            mock_sentiment_service__analyze.side_effect = [
                {"label": "POS", "confidence": 0.9},
                {"error": "Mocked sentiment analysis failure"}
            ]

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {
                'audio_path': '/path/to/audio.mp3',
                'start_time_ms': 10,
                'end_time_ms': 20,
                'transcription': 'This is a test transcription.',
                'utterances_sentiment': [
                    {'timestamp': [10, 15], 'text': 'First chunk', 'label': 'POS', 'confidence': 0.9},
                    {'timestamp': [15, 20], 'text': 'Second chunk', 'error': 'Mocked sentiment analysis failure'}
                ]
            }

        def test_process__sentiment_analysis_exception(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze
        ):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }
            mock_sentiment_service__analyze.side_effect = [
                Exception("Mocked sentiment analysis exception"),
                {"label": "POS", "confidence": 0.9}
            ]

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {'error': 'An unexpected error occurred while processing the request.'}
            assert mock_audio_service__extract_audio.called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )
            assert mock_transcript_service__transcribe.called_once_with("/path/to/audio.mp3")
            assert mock_sentiment_service__analyze.call_once_with("First chunk")

        def test_process_success(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze,
            mock_os__remove
        ):
            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }
            mock_sentiment_service__analyze.side_effect = [
                {"label": "POS", "confidence": 0.9},
                {"label": "NEG", "confidence": 0.8}
            ]

            result = audio_transcription_sentiment_pipeline.process(**payload)

            assert result == {
                'audio_path': '/path/to/audio.mp3',
                'start_time_ms': 10,
                'end_time_ms': 20,
                'transcription': 'This is a test transcription.',
                'utterances_sentiment': [
                    {'timestamp': [10, 15], 'text': 'First chunk', 'label': 'POS', 'confidence': 0.9},
                    {'timestamp': [15, 20], 'text': 'Second chunk', 'label': 'NEG', 'confidence': 0.8}
                ]
            }
            assert mock_audio_service__extract_audio.called_once_with(
                payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id']
            )
            assert mock_transcript_service__transcribe.called_once_with("/path/to/audio.mp3")
            assert mock_sentiment_service__analyze.call_count == 2
            assert mock_os__remove.called == False

        # --- NEW: remove_audio=True coverage ---

        def test_process__remove_audio_called_when_enabled(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze,
            mock_os__remove
        ):
            """
            Test that os.remove() is called with the correct audio path
            when remove_audio is set to True.
            Previously untested — the shared fixture hardcodes remove_audio=False.
            """
            # Enable remove_audio for this test only
            audio_transcription_sentiment_pipeline.remove_audio = True

            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"}
                ]
            }
            mock_sentiment_service__analyze.return_value = {"label": "POS", "confidence": 0.9}

            result = audio_transcription_sentiment_pipeline.process(**payload)

            # os.remove must be called exactly once with the audio path
            mock_os__remove.assert_called_once_with("/path/to/audio.mp3")

            assert result == {
                'audio_path': '/path/to/audio.mp3',
                'start_time_ms': 10,
                'end_time_ms': 20,
                'transcription': 'This is a test transcription.',
                'utterances_sentiment': [
                    {'timestamp': [10, 15], 'text': 'First chunk', 'label': 'POS', 'confidence': 0.9}
                ]
            }

        def test_process__remove_audio_not_called_when_disabled(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze,
            mock_os__remove
        ):
            """
            Test that os.remove() is NOT called when remove_audio is False.
            Complements test_process__remove_audio_called_when_enabled.
            """
            audio_transcription_sentiment_pipeline.remove_audio = False

            payload = self.args.copy()
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"}
                ]
            }
            mock_sentiment_service__analyze.return_value = {"label": "NEU", "confidence": 0.7}

            audio_transcription_sentiment_pipeline.process(**payload)

            mock_os__remove.assert_not_called()
