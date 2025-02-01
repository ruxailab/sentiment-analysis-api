"""
This Module contains the unit tests for the audio routes.
"""

import pytest
from unittest.mock import patch


class TestAudioExtract:
    """Test suite for the AudioExtract route."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/audio/extract"
        self.payload = {
            "url": "https://example.com/audio.mp3",
            "start_time_ms": 0,
            "end_time_ms": 5000,
            "user_id": "user123"
        }

    @pytest.fixture
    def mock_extract_audio(self):
        """Fixture to mock the extract_audio method."""
        with patch('app.routes.audio_routes.AudioService.extract_audio') as mock:
            yield mock



    def test_audio_extract_missing_url(self, mock_extract_audio):
        """Test missing URL in the request."""
        payload = self.payload.copy()
        del payload['url']  # Remove URL to simulate the error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "URL is required",
            "data": None
        }
        mock_extract_audio.assert_not_called()  # Ensure the method was not called

    def test_audio_extract_negative_start_time(self, mock_extract_audio):
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
        mock_extract_audio.assert_not_called()  # Ensure the method was not called

    def test_audio_extract_negative_end_time(self, mock_extract_audio):
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
        mock_extract_audio.assert_not_called()  # Ensure the method was not called

    def test_audio_extract_end_time_less_than_start_time(self, mock_extract_audio):
        """Test end time less than start time in the request."""
        payload = self.payload.copy()
        payload['start_time_ms'] = 1000
        payload['end_time_ms'] = 0
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "'end_time_ms' must be greater than or equal to 'start_time_ms'.",
            "data": None
        }
        mock_extract_audio.assert_not_called()  # Ensure the method was not called

    def test_audio_extract_failure(self, mock_extract_audio):
        """Test failed audio extraction (mocking service failure)."""
        # Mock the service to return an error
        mock_extract_audio.return_value = {'error': 'Mocked service error'}

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Mocked service error",
            "data": None
        }
        mock_extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])

    def test_audio_extract_exception(self, mock_extract_audio):
        """Test unexpected exception during audio extraction (mocking service exception)."""
        # Mock the service to raise an exception
        mock_extract_audio.side_effect = Exception('Mocked exception')

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred while processing the request.",
            "data": None
        }
        mock_extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])


    def test_audio_extract_success(self, mock_extract_audio):
        """Test successful audio extraction (mocking service response)."""
        # Mock the service response
        mock_extract_audio.return_value = {
            'audio_path': '/path/to/audio.mp3',
            'start_time_ms': 0,
            'end_time_ms': 5000
        }
    
        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "audio_path": "/path/to/audio.mp3",
                "start_time_ms": 0,
                "end_time_ms": 5000
            }
        }
        mock_extract_audio.assert_called_once_with(payload['url'], payload['start_time_ms'], payload['end_time_ms'], payload['user_id'])

# # Run:
# coverage run  -m pytest .\tests\unit\test_routes\test_audio_routes.py