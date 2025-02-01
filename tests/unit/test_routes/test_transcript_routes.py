"""
This Module contains the unit tests for the transcript routes.
"""

import pytest
from unittest.mock import patch


class TestTranscriptionTranscribe:
    """Test suite for the TranscriptionTranscribe route."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/transcription/transcribe"
        self.payload = {
            "file_path": "audio.mp3"
        }

    @pytest.fixture
    def mock_transcribe(self):
        """Fixture to mock the transcribe_audio method."""
        with patch('app.routes.transcript_routes.TranscriptService.transcribe') as mock:
            yield mock

    def test_transcription_transcribe_missing_file_path(self,mock_transcribe):
        """Test missing file_path in the request."""
        payload = self.payload.copy()
        del payload['file_path']  # Remove file_path to simulate the error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }
        mock_transcribe.assert_not_called()  # Ensure the method was not called


    def test_transcription_transcribe_failure(self, mock_transcribe):
        """Test the transcribe_audio method returns an error."""
        # Mock the service method to return an error
        mock_transcribe.return_value = { 'error': 'Mocked service error' }

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Mocked service error",
            "data": None
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])

    def test_transcription_transcribe_success(self, mock_transcribe):
        """Test a successful transcription."""
        # Mock the service method to return a successful response
        mocked_transcription = "Hello, world!"
        mocked_chunks = [{'timestamp': (0.0, 3.0), 'text': "Hello, world!"}]
        mock_transcribe.return_value = { 
            'transcription': mocked_transcription,
            'chunks': mocked_chunks
            }

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "transcription": mocked_transcription,
            }
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])


    def test_transcription_transcribe_exception(self, mock_transcribe):
        """Test an unexpected exception during transcription."""
        # Mock the service method to raise an exception
        mock_transcribe.side_effect = Exception('Mocked exception')

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred during transcription.",
            "data": None
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])

class TestTranscriptionChunks:
    """Test suite for the TranscriptionChunks route."""
    
    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/transcription/chunks"
        self.payload = {
            "file_path": "audio.mp3"
        }

    @pytest.fixture
    def mock_transcribe(self):
        """Fixture to mock the transcribe_audio method."""
        with patch('app.routes.transcript_routes.TranscriptService.transcribe') as mock:
            yield mock

    def test_transcription_chunks_missing_file_path(self, mock_transcribe):
        """Test missing file_path in the request."""
        payload = self.payload.copy()
        del payload['file_path']  # Remove file_path to simulate the error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }
        mock_transcribe.assert_not_called() # Ensure the method was not called

    def test_transcription_chunks_failure(self, mock_transcribe):
        """Test the transcribe_audio method returns an error."""
        # Mock the service method to return an error
        mock_transcribe.return_value = { 'error': 'Mocked service error' }

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Mocked service error",
            "data": None
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])


    def test_transcription_chunks_success(self, mock_transcribe):
        """Test a successful transcription."""
        # Mock the service method to return a successful response
        mocked_transcription = "Hello, world!"
        mocked_chunks = [{'timestamp': 'Mocked_timestamp', 'text': "Hello, world!"}]
        mock_transcribe.return_value = { 
            'transcription': mocked_transcription,
            'chunks': mocked_chunks
        }

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "transcription": mocked_transcription,
                "chunks": mocked_chunks
            }
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])

    def test_transcription_chunks_exception(self, mock_transcribe):
        """Test an unexpected exception during transcription."""
        # Mock the service method to raise an exception
        mock_transcribe.side_effect = Exception('Mocked exception')

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred during transcription.",
            "data": None
        }
        mock_transcribe.assert_called_once_with(audio_file_path=payload['file_path'])


# # Run:
# coverage run  -m pytest .\tests\unit\test_routes\test_transcript_routes.py




