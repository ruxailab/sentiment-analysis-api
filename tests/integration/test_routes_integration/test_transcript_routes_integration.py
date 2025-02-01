"""
Integration tests for:
    - /transcription/transcribe
    - /transcription/chunks
"""

import pytest

class TestTranscriptionTranscribe:
    """Test suite for the /transcription/transcribe endpoint."""
    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/transcription/transcribe"
        self.payload = {
            "file_path": "./samples/sample_1.mp3"
        }

    def test_transcribe_missed_file_path(self):
        """Test missing file_path in the request."""
        payload = self.payload.copy()
        del payload['file_path']
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }

    def test_transcribe_empty_file_path(self):
        """Test empty file path in the request."""
        payload = self.payload.copy()
        payload['file_path'] = ""  # Empty file_path to simulate error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }

    def test_transcribe_non_exists_file_path(self):
        """Test non-existing file path in the request."""
        payload = self.payload.copy()
        payload['file_path'] = "non-exist.mp3"  # Empty file_path to simulate error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": f"Audio file not found: {payload['file_path']}",
            "data": None
        }

    def test_transcribe_success(self):
        """Test successful transcription of the audio."""
        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"

        # Ensure no other keys are present in the response
        expected_keys = {"status", "data"}
        actual_keys = set(response.json.keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"

        # Ensure no other keys are present in the response
        expected_keys = {"transcription"}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response (data): {actual_keys}"

        assert isinstance(response.json["data"]["transcription"], str)   

class TestTranscriptionChunks:
    """Test suite for the /transcription/chunks endpoint."""
    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/transcription/chucks"
        self.payload = {
            "file_path": "./samples/sample_1.mp3"
        }

    def test_chunks_missed_file_path(self):
        """Test missing file_path in the request."""
        payload = self.payload.copy()
        del payload['file_path']
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }

    def test_chunks_empty_file_path(self):
        """Test empty file path in the request."""
        payload = self.payload.copy()
        payload['file_path'] = ""  # Empty file_path to simulate error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "file_path is required.",
            "data": None
        }

    def test_chunks_non_exists_file_path(self):
        """Test non-existing file path in the request."""
        payload = self.payload.copy()
        payload['file_path'] = "non-exist.mp3"  # Empty file_path to simulate error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": f"Audio file not found: {payload['file_path']}",
            "data": None
        }

    def test_chunks_success(self):
        """Test successful extraction of chunks from the audio."""
        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"

        # Ensure no other keys are present in the response
        expected_keys = {"status", "data"}
        actual_keys = set(response.json.keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"

        # Ensure no other keys are present in the response
        expected_keys = {'transcription', 'chunks'}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response (data): {actual_keys}"
        
        assert isinstance(response.json["data"]["transcription"], str) 
        assert isinstance(response.json["data"]["chunks"], list)
        for chunk in response.json["data"]["chunks"]:
        # [{'timestamp': (0.0, 3.0), 'text': " And there's a voice and it's a little quiet voice that goes jump."},
            assert isinstance(chunk, dict)
            assert isinstance(chunk["timestamp"], list)
            assert len(chunk["timestamp"]) == 2
            assert isinstance(chunk["timestamp"][0], float)
            assert isinstance(chunk["timestamp"][1], float)
            assert isinstance(chunk["text"], str)
        
    

# # Run:
# coverage run  -m pytest .\tests\integration\test_routes_integration\test_transcript_routes_integration.py