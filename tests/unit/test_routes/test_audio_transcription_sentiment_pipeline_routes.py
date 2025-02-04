"""
This Module contains the unit tests for the Audio Transcript Sentiment routes.
"""

import pytest
from unittest.mock import patch

class TestAudioTranscriptionSentimentPipelineProcess:
    """Test suite for the AudioTranscriptionSentimentPipeline process route."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/audio-transcript-sentiment/process"
        self.payload = {
            "url": "https://example.com/audio.mp3", 
            "start_time_ms": 0,
            "end_time_ms": 5000
        }

    @pytest.fixture
    def mock_process(self):
        """Fixture to mock the process method."""
        with patch('app.routes.audio_transcript_sentiment_routes.AudioTranscriptionSentimentPipeline.process') as mock_process:
            yield mock_process

    def test_audio_transcript_sentiment_process_missing_url(self, mock_process):
        """Test missing URL in the request."""
        payload = self.payload.copy()
        del payload['url']
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "url is required.",
            "data": None
        }
        mock_process.assert_not_called()  # Ensure the method was not called


    def test_audio_transcript_sentiment_negative_start_time(self, mock_process):
        """Test negative start time in the request."""
        payload = self.payload.copy()
        payload['start_time_ms'] = -100
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "'start_time_ms' cannot be negative.",
            "data": None
        }
        mock_process.assert_not_called()  # Ensure the method was not called

    def test_audio_transcript_sentiment_negative_end_time(self, mock_process):
        """Test negative end time in the request."""
        payload = self.payload.copy()
        payload['end_time_ms'] = -100
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "'end_time_ms' cannot be negative.",
            "data": None
        }
        mock_process.assert_not_called()  # Ensure the method was not called


    def test_audio_transcript_sentiment_end_time_less_than_start_time(self, mock_process):
        """Test end time less than start time in the request."""
        payload = self.payload.copy()
        payload['start_time_ms'] = 5000
        payload['end_time_ms'] = 1000

        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "'end_time_ms' must be greater than 'start_time_ms'.",
            "data": None
        }


    def test_audio_transcript_sentiment_failure(self, mock_process):
        """Test when the service layer returns an error."""
        payload = self.payload.copy()
        mock_process.return_value = {
            "error": "Mocked error"
        }
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Mocked error",
            "data": None
        }
        mock_process.assert_called_once_with(url=payload['url'], start_time_ms=payload['start_time_ms'], end_time_ms=payload['end_time_ms'])

    def test_audio_transcript_sentiment_exception(self, mock_process):
        """Test when the service layer raises an exception."""
        # Mock the service method to raise an exception
        mock_process.side_effect = Exception("Mocked exception")

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred while processing the request.",
            "data": None
        }
        mock_process.assert_called_once_with(url=payload['url'], start_time_ms=payload['start_time_ms'], end_time_ms=payload['end_time_ms'])


    def test_audio_transcript_sentiment_success(self, mock_process):
        """Test successful audio transcript sentiment analysis."""
        payload = self.payload.copy()
        mock_process.return_value = {
            "audio_path": "/path/to/audio",
            "start_time_ms": 0,
            "end_time_ms": 5000,
            "transcription": "Hello World",
            "utterances_sentiment": [
                {
                    "utterance": "Hello",
                    "sentiment": "POS",
                    "confidence": 0.95
                },
                {
                    "utterance": "World",
                    "sentiment": "NEG",
                    "confidence": 0.90
                }
            ]
        }

        response = self.client.post(self.endpoint, json=payload)
        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "audio_path": "/path/to/audio",
                "start_time_ms": 0,
                "end_time_ms": 5000,
                "transcription": "Hello World",
                "utterances_sentiment": [
                    {
                        "utterance": "Hello",
                        "sentiment": "POS",
                        "confidence": 0.95
                    },
                    {
                        "utterance": "World",
                        "sentiment": "NEG",
                        "confidence": 0.90
                    }
                ]
            }
        }
        mock_process.assert_called_once_with(url=payload['url'], start_time_ms=payload['start_time_ms'], end_time_ms=payload['end_time_ms'])


# # Run:
# coverage run  -m pytest .\tests\unit\test_routes\test_audio_transcription_sentiment_pipeline_routes.py


