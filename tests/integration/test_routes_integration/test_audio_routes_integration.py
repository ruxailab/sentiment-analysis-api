"""
Integration test for the /audio route
"""

import pytest

class TestAudioExtract:
    """Test suite for the /audio/extract endpoint."""
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

    def test_audio_extract_missing_url(self):
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

    def test_audio_extract_negative_start_time(self):
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

    def test_audio_extract_negative_end_time(self):
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

    def test_audio_extract_end_time_less_than_start_time(self):
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

    def test_audio_extract_invalid_url(self):
        """Test invalid URL in the request."""
        payload = self.payload.copy()
        payload['url'] = "https://example.com/audio.mp3"
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json["status"] == "error"
        assert "An error occurred during the HTTP request" in response.json["error"]

    def test_audio_extract_file_not_exist(self):
        """Test file does not exist"""
        payload = self.payload.copy()
        payload['url'] = "./non-exist.mp4"
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Provided url is neither a valid URL nor a valid file path.",
            "data": None
        }

    def test_audio_extract_success_url(self):
        """Test successful extraction with a valid URL."""
        payload = self.payload.copy()
        payload['url'] = "https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v"
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"

        # Ensure no other keys are present in the response
        expected_keys = {"audio_path", "start_time_ms", "end_time_ms"}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"
        
        assert "audio_path" in response.json["data"]
        assert isinstance(response.json["data"]["audio_path"], str)
        assert response.json["data"]["start_time_ms"] == 0
        assert response.json["data"]["end_time_ms"] == 5000

    def test_audio_extract_success_file_path(self):
        """Test successful extraction with a valid file path."""
        payload = self.payload.copy()
        payload['url'] = "./samples/sample_0.mp4"
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"
        
        # Ensure no other keys are present in the response
        expected_keys = {"audio_path", "start_time_ms", "end_time_ms"}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"
        
        assert "audio_path" in response.json["data"]
        assert isinstance(response.json["data"]["audio_path"], str)
        assert response.json["data"]["start_time_ms"] == 0
        assert response.json["data"]["end_time_ms"] == 5000

# # Run:
# coverage run  -m pytest .\tests\integration\test_routes_integration\test_audio_routes_integration.py 