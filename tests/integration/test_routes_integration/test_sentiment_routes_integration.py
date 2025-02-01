"""
Integration tests for:
    - /sentiment/analyze
"""

import pytest

class TestSentimentAnalyze:
    """Test suite for the /sentiment/analyze endpoint."""
    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/sentiment/analyze"
        self.payload = {
            "text": "I love this product!"
        }

    def test_sentiment_analyze_missing_text(self):
        """Test missing text in the request."""
        payload = self.payload.copy()
        del payload['text']  # Remove text to simulate the error
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 400
        assert response.json == {
            "status": "error",
            "error": "text is required.",
            "data": None
        }


    def test_sentiment_analyze_success(self):
        """Test successful analysis of the text."""
        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json["status"] == "success"

        # Ensure no other keys are present in the response
        expected_keys = {"status", "data"}
        actual_keys = set(response.json.keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response: {actual_keys}"

        # Ensure no other keys are present in the response
        expected_keys = {"label", "confidence"}
        actual_keys = set(response.json["data"].keys())
        assert actual_keys == expected_keys, f"Unexpected keys in response (data): {actual_keys}"

        assert isinstance(response.json["data"]["label"], str)
        assert isinstance(response.json["data"]["confidence"], float)


# # Run:
# coverage run  -m pytest .\tests\integration\test_routes_integration\test_sentiment_routes_integration.py