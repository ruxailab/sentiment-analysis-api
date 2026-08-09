"""
This Module contains the unit tests for the sentiment service.
"""

import pytest
from unittest.mock import patch
from app.services.sentiment_service import SentimentService



@pytest.fixture
def sentiment_service():
    """Fixture to provide a SentimentService instance."""
    return SentimentService()

@pytest.fixture
def mock_sentiment_data_layer__analyze():
    """Fixture to mock the analyze method of SentimentDataLayer."""
    with patch('app.services.sentiment_service.SentimentDataLayer.analyze') as mock_analyze:
        yield mock_analyze



class TestAnalyze:
    def setup_method(self):
        """Setup method for each test."""
        self.single_text = 'I love this product!'
        self.batch_texts = ['I love it', 'I hate it']

    def test_analyze_success_single(self, sentiment_service, mock_sentiment_data_layer__analyze):
        """Test for successful single text analysis."""
        mock_sentiment_data_layer__analyze.return_value = {
            'label': 'POS',
            'confidence': 0.95
        }

        result = sentiment_service.analyze(self.single_text)

        assert result == {'label': 'POS', 'confidence': 0.95}
        mock_sentiment_data_layer__analyze.assert_called_once_with(self.single_text)

    def test_analyze_success_batch(self, sentiment_service, mock_sentiment_data_layer__analyze):
        """Test for successful batch text analysis."""
        mock_sentiment_data_layer__analyze.return_value = [
            {'label': 'POS', 'confidence': 0.99},
            {'label': 'NEG', 'confidence': 0.88}
        ]

        result = sentiment_service.analyze(self.batch_texts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['label'] == 'POS'
        assert result[1]['label'] == 'NEG'
        mock_sentiment_data_layer__analyze.assert_called_once_with(self.batch_texts)

    def test_analyze_failure(self, sentiment_service, mock_sentiment_data_layer__analyze):
        """Test for handling error response from Data Layer."""
        mock_sentiment_data_layer__analyze.return_value = {'error': 'Mocked error'}

        result = sentiment_service.analyze(self.single_text)

        assert result == {'error': 'Mocked error'}

    def test_analyze_exception(self, sentiment_service, mock_sentiment_data_layer__analyze):
        """Test for handling unexpected exceptions."""
        mock_sentiment_data_layer__analyze.side_effect = Exception("Mocked exception")

        result = sentiment_service.analyze(self.single_text)

        assert result == {'error': 'An unexpected error occurred while processing the request.'}

# # Run the tests
# coverage run  -m pytest .\tests\unit\test_services\test_sentiment_service.py