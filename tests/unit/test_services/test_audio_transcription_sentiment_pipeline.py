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
        """
        pipeline = AudioTranscriptionSentimentPipeline()

        # Override the remove_audio attribute to prevent deletion of audio files
        pipeline.remove_audio = False
        return  pipeline
    
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
            """
            Fixture to mock the AudioService class.
            """
            with patch("app.services.audio_transcription_sentiment_pipeline.AudioService.extract_audio") as mock_audio_service__extract_audio:
                yield mock_audio_service__extract_audio

        @pytest.fixture
        def mock_transcript_service__transcribe(self):
            """
            Fixture to mock the TranscriptService class.
            """
            with patch("app.services.audio_transcription_sentiment_pipeline.TranscriptService.transcribe") as mock_transcript_service__transcribe:
                yield mock_transcript_service__transcribe

        @pytest.fixture
        def mock_sentiment_service__analyze(self):
            """
            Fixture to mock the SentimentService class.
            """
            with patch("app.services.audio_transcription_sentiment_pipeline.SentimentService.analyze") as mock_sentiment_service__analyze:
                yield mock_sentiment_service__analyze

        @pytest.fixture
        def mock_os__remove(self):
            """
            Fixture to mock the os.remove method.
            """
            with patch("os.remove") as mock_os__remove:

                # Override the remove_audio attribute to prevent deletion of audio files
                mock_os__remove.return_value = True
                yield mock_os__remove

        
        def test_process__extract_audio_failure(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio):
            """
            Test the process method when the extract_audio method fails.
            """
            payload = self.args.copy()
            # Setup
            mock_audio_service__extract_audio.return_value = {
                "error": "Mocked error message"
            }

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
            assert result == {
                'error': "Mocked error message"
            }
            mock_audio_service__extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])


        def test_process__extract_audio_exception(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio):
            """
            Test the process method when the extract_audio method raises an exception.
            """
            payload = self.args.copy()
            # Setup
            mock_audio_service__extract_audio.side_effect = Exception("Mocked exception")

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
            assert result == {
                'error': "An unexpected error occurred while processing the request."
            }
            mock_audio_service__extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])


        def test_process__transcribe_audio_failure(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio, mock_transcript_service__transcribe):
            """
            Test the process method when the transcribe method fails.
            """
            payload = self.args.copy()
            # Setup
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.return_value = {
                "error": "Mocked error message"
            }

            # Run
            result = audio_transcription_sentiment_pipeline.process(**self.args)

            # Assert
            assert result == {
                'error': "Mocked error message"
            }
            mock_audio_service__extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])
            mock_transcript_service__transcribe.assert_called_once_with("/path/to/audio.mp3")

        def test_process__transcribe_audio_exception(self, audio_transcription_sentiment_pipeline, mock_audio_service__extract_audio, mock_transcript_service__transcribe):
            """
            Test the process method when the transcribe method raises an exception.
            """
            payload = self.args.copy()
            # Setup
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }
            mock_transcript_service__transcribe.side_effect = Exception("Mocked exception")

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
            assert result == {
                'error': "An unexpected error occurred while processing the request."
            }
            mock_audio_service__extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])
            mock_transcript_service__transcribe.assert_called_once_with("/path/to/audio.mp3")

        
        def test_process__sentiment_analysis_failure(
            self,
            audio_transcription_sentiment_pipeline,
            mock_audio_service__extract_audio,
            mock_transcript_service__transcribe,
            mock_sentiment_service__analyze
        ):
            """
            Test the process method when the sentiment analysis service fails for one or more chunks.
            """
            payload = self.args.copy()

            # Mock extract_audio success
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }

            # Mock transcribe success with multiple chunks
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }

            # Mock sentiment analysis failure for one chunk
            mock_sentiment_service__analyze.side_effect = [
                {"label": "POS", "confidence": 0.9},  # First chunk succeeds
                {"error": "Mocked sentiment analysis failure"}  # Second chunk fails
            ]

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
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
            """
            Test the process method when the sentiment analysis service raises an exception.
            """
            payload = self.args.copy()

            # Mock extract_audio success
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }

            # Mock transcribe success with multiple chunks
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }

            # Mock sentiment analysis failure for one chunk
            mock_sentiment_service__analyze.side_effect = [
                Exception("Mocked sentiment analysis exception"),
                {"label": "POS", "confidence": 0.9}  # Second chunk succeeds
            ]

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
            assert result == {
                'error': 'An unexpected error occurred while processing the request.'
            }
            assert mock_audio_service__extract_audio.called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])
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
            """
            Test the process method when all services succeed.
            """
            payload = self.args.copy()

            # Mock extract_audio success
            mock_audio_service__extract_audio.return_value = {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20
            }

            # Mock transcribe success with multiple chunks
            mock_transcript_service__transcribe.return_value = {
                "transcription": "This is a test transcription.",
                "chunks": [
                    {"timestamp": [10, 15], "text": "First chunk"},
                    {"timestamp": [15, 20], "text": "Second chunk"}
                ]
            }

            # Mock sentiment analysis success for all chunks
            mock_sentiment_service__analyze.side_effect = [
                {"label": "POS", "confidence": 0.9},
                {"label": "NEG", "confidence": 0.8}
            ]

            # Run
            result = audio_transcription_sentiment_pipeline.process(**payload)

            # Assert
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
            assert mock_audio_service__extract_audio.called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])
            assert mock_transcript_service__transcribe.called_once_with("/path/to/audio.mp3")
            assert mock_sentiment_service__analyze.call_count == 2
            assert mock_os__remove.called == False


# # Run:
# coverage run  -m pytest .\tests\unit\test_services\test_audio_transcription_sentiment_pipeline.py