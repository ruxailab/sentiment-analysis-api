"""
Integration tests for:
    - /audio-transcript-sentiment/process
"""

import pytest

class TestAudioTranscriptSentimentProcess:
    """Test suite for the /audio-transcript-sentiment/process endpoint."""
    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/audio-transcript-sentiment/process"
        self.payload = {
            "url": "https://example.com/audio.mp3",
            "start_time_ms": 0,
            "end_time_ms": 5000,
            "user_id": "user123"
        }
    
    def test_audio_transcript_sentiment_process_missing_url(self):
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

    def test_audio_transcript_sentiment_negative_start_time(self):
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

    def test_audio_transcript_sentiment_negative_end_time(self):
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

    def test_audio_transcript_sentiment_end_time_less_than_start_time(self):
        """Test end time less than start time in the request."""
        payload = self.payload.copy()
        payload['end_time_ms'] = 1000
        payload['start_time_ms'] = 2000
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "'end_time_ms' must be greater than 'start_time_ms'.",
            "data": None
        }

    def test_audio_transcript_sentiment_process_success(self):
        """Test successful audio transcription and sentiment analysis."""
        payload = self.payload.copy()
        payload['url'] = "./samples/sample_0.mp4"
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"

        # Ensure no other keys are present in the response
        expected_keys = {"status", "data"}
        actual_keys = set(response.json.keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"

        # Ensure no other keys are present in the response
        expected_keys = {"audio_path", "start_time_ms", "end_time_ms", "transcription", "utterances_sentiment"}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response (data): {actual_keys}"

        assert isinstance(response.json["data"]["transcription"], str)
        assert isinstance(response.json["data"]["utterances_sentiment"], list)
        for chunk in response.json["data"]["utterances_sentiment"]:
            # [{'timestamp': (0.0, 3.0), 'text': " And there's a voice and it's a little quiet voice that goes jump.", 'sentiment': 'POS', 'confidence': 0.99}]
            # OR 
            # [{'timestamp': (0.0, 3.0), 'text': " And there's a voice and it's a little quiet voice that goes jump.",'error': 'error message'}]
            assert isinstance(chunk, dict)
            assert isinstance(chunk["timestamp"], list)
            assert len(chunk["timestamp"]) == 2
            assert isinstance(chunk["text"], str)

            if 'error' not in chunk:
                assert isinstance(chunk["label"], str)
                assert isinstance(chunk["confidence"], float)
            else:
                assert isinstance(chunk["error"], str)


# # Run:
# coverage run  -m pytest .\tests\integration\test_routes_integration\test_audio_transcription_sentiment_pipeline_routes_integration.py