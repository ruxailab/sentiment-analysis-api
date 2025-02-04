"""
This Module contains the unit tests for the sentiment routes.
"""

import pytest
from unittest.mock import patch

class TestSentimentAnalyze:
    """Test suite for the SentimentAnalyze route."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/sentiment/analyze"
        self.payload = {
            "text": "I love this product!"
        }

    @pytest.fixture
    def mock_analyze(self):
        """Fixture to mock the analyze method."""
        with patch('app.routes.sentiment_routes.SentimentService.analyze') as mock_analyze:
            yield mock_analyze

    def test_sentiment_analyze_missing_text(self, mock_analyze):
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
        mock_analyze.assert_not_called()  # Ensure the method was not called

    def test_sentiment_analyze_failure(self, mock_analyze):
        """Test when the service layer returns an error."""
        payload = self.payload.copy()
        mock_analyze.return_value = {
            "error": "Mocked error"
        }
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "Mocked error",
            "data": None
        }
        mock_analyze.assert_called_once_with(payload['text'])

    def test_sentiment_analyze_exception(self, mock_analyze):
        """Test when the service layer raises an exception."""
        # Mock the service method to raise an exception
        mock_analyze.side_effect = Exception("Mocked exception")

        payload = self.payload.copy()
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred while processing the request.",
            "data": None
        }
        mock_analyze.assert_called_once_with(payload['text'])


    def test_sentiment_analyze_success(self, mock_analyze):
        """Test successful sentiment analysis."""
        payload = self.payload.copy()
        mock_analyze.return_value = {
            "label": "POS",
            "confidence": 0.95
        }
        response = self.client.post(self.endpoint, json=payload)

        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "label": "POS",
                "confidence": 0.95
            }
        }
        mock_analyze.assert_called_once_with(payload['text'])

# # Run:
# coverage run  -m pytest .\tests\unit\test_routes\test_sentiment_routes.py