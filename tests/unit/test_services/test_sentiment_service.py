"""
This Module contains the unit tests for the sentiment service.
"""

import pytest
from unittest.mock import patch

from app.services.sentiment_service import SentimentService


class TestSentimentService:
    """Test suite for the SentimentService class."""
    @pytest.fixture()
    def sentiment_service(self):
        """Fixture for creating an instance of the SentimentService class."""
        return SentimentService()
    
    # Grouped tests for the 'analyze' method
    class TestAnalyze:
        def setup_method(self):
            """Setup method for each test."""
            self.args = {
                'text': 'I love this product!'
            }

        # Mock Methods of SentimentDataLayer
        @pytest.fixture
        def mock_sentiment_data_layer__analyze(self):
            """Fixture for mocking the SentimentDataLayer.analyze method."""
            with patch('app.services.sentiment_service.SentimentDataLayer.analyze') as mock:
                yield mock
        
        def test_analyze__sentiment_data_layer_analyze_failure(self, sentiment_service, mock_sentiment_data_layer__analyze):
            """
            Test for when the SentimentDataLayer.analyze method fails.
            """
            args = self.args.copy()
            mock_sentiment_data_layer__analyze.return_value = {
                'error': 'Mocked error'
            }

            result = sentiment_service.analyze(**args)

            assert result == {
                'error': "Mocked error"
            }
            mock_sentiment_data_layer__analyze.assert_called_once_with(args['text'])

        def test_analyze__sentiment_data_layer_analyze_exception(self, sentiment_service, mock_sentiment_data_layer__analyze):
            """
            Test for when the SentimentDataLayer.analyze method raises an exception.
            """
            args = self.args.copy()
            mock_sentiment_data_layer__analyze.side_effect = Exception("Mocked exception")

            result = sentiment_service.analyze(**args)

            assert result == {
                'error': 'An unexpected error occurred while processing the request.'
            }
            mock_sentiment_data_layer__analyze.assert_called_once_with(args['text'])

        def test_analyze_success(self, sentiment_service, mock_sentiment_data_layer__analyze):
            """
            Test for when the SentimentDataLayer.analyze method succeeds.
            """
            args = self.args.copy()
            mock_sentiment_data_layer__analyze.return_value = {
                'label': 'POS',
                'confidence': 0.95
            }

            result = sentiment_service.analyze(**args)

            assert result == {
                'label': 'POS',
                'confidence': 0.95
            }
            mock_sentiment_data_layer__analyze.assert_called_once_with(args['text'])

# # Run the tests
# coverage run  -m pytest .\tests\unit\test_services\test_sentiment_service.py