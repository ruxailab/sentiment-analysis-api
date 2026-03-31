"""
This module contains unit tests for the SentimentDataLayer class.
"""

import pytest
from unittest.mock import MagicMock, patch

# Import the class to be tested
from app.data.sentiment_data import SentimentDataLayer

# Grouped tests for initialization and analyze method
class TestSentimentDataLayer:

    # Grouped tests for initialization (__init__ method)
    class TestInitialization:
        @pytest.fixture
        def sentiment_data_layer__bertweet(self):
            """
            Fixture to set up SentimentDataLayer instance for testing.
            """
            config = {
                'debug': True,
                'sentiment_analysis': {
                    'default_model': "bertweet",
                    'bertweet': {
                        'model_name': 'finiteautomata/bertweet-base-sentiment-analysis',
                        'device': 'cpu'
                    }
                }
            }
            return SentimentDataLayer(config)


        @pytest.fixture
        def mock_bertweet_sentiment(self):
            """
            Fixture to mock the 'BertweetSentiment' class.
            """
            with patch('app.data.sentiment_data.BertweetSentiment') as mock_bertweet_sentiment:
                yield mock_bertweet_sentiment

        def test_init_bertweet_model(self,mock_bertweet_sentiment,sentiment_data_layer__bertweet):
            """
            Test that SentimentDataLayer initializes the Bertweet model.
            """
            # Ensure the BertweetSentiment is initialized with the correct configuration
            mock_bertweet_sentiment.assert_called_once_with({
                'debug': True,
                'sentiment_analysis': {
                    'default_model': "bertweet",
                    'bertweet': {
                        'model_name': 'finiteautomata/bertweet-base-sentiment-analysis',
                        'device': 'cpu'
                    }
                }
            })

            # Ensure the model is set to the BertweetSentiment instance
            assert isinstance(sentiment_data_layer__bertweet.model, mock_bertweet_sentiment.return_value.__class__)


        def test_init_unsupported_model(self):
            """
            Test that an exception is raised for an unsupported model.
            """
            config = {
                'debug': True,
                'sentiment_analysis': {'default_model': 'unsupported_model'}
            }
            with pytest.raises(ValueError) as e:
                SentimentDataLayer(config)

            assert str(e.value) == "Unsupported sentiment analysis model: unsupported_model"


    # Define a fixture to initialize the SentimentDataLayer instance for testing
    @pytest.fixture
    def sentiment_data_layer(self):
        """
        Fixture to set up SentimentDataLayer instance for testing.
        """
        config = {
            'debug': True,
            'sentiment_analysis': {
                'default_model': "bertweet",
                'bertweet': {
                    'model_name': 'finiteautomata/bertweet-base-sentiment-analysis',
                    'device': 'cpu'
                }
            }
        }
        return SentimentDataLayer(config)
    

    # Grouped tests for analyze method
    class TestAnalyze:
        def setup_method(self):
            """
            Setup method to create a SentimentDataLayer instance for testing.
            """
            self.single_text = "This is awesome!"
            self.batch_texts = ["I love it", "I hate it"]

        @pytest.fixture
        def mock_model(self,sentiment_data_layer):
            """
            Fixture to mock the self.model
            """
            with patch.object(sentiment_data_layer,'model',MagicMock()) as mock_model:
                yield mock_model

        def test_analyze_exception(self, sentiment_data_layer, mock_model):
            """
            Test that the analyze method handles exceptions.
            """
            # Mock the model to raise an exception
            mock_model.side_effect = Exception("An error occurred")

            result = sentiment_data_layer.analyze(self.single_text)

            # Ensure the method returns an error message
            assert result == {'error': 'An unexpected error occurred while processing the request.'}
            mock_model.assert_called_once_with(self.single_text)


        
        def test_analyze_success_single(self, sentiment_data_layer, mock_model):
            """
            Test analyzing a single text input and ensure it returns a dictionary.
            """
            # The model now returns a list of dicts even for single input
            mock_model.return_value = [{'text': self.single_text, 'label': 'POS', 'confidence': 0.9}]

            result = sentiment_data_layer.analyze(self.single_text)

            # Ensure Data Layer extracts the dict from the list for backward compatibility
            assert result == {'label': 'POS', 'confidence': 0.9}
            mock_model.assert_called_once_with(self.single_text)

        def test_analyze_success_batch(self, sentiment_data_layer, mock_model):
            """
            Test analyzing a batch of texts and ensure it returns a list of dictionaries.
            """
            mock_model.return_value = [
                {'text': "I love it", 'label': 'POS', 'confidence': 0.99},
                {'text': "I hate it", 'label': 'NEG', 'confidence': 0.88}
            ]

            result = sentiment_data_layer.analyze(self.batch_texts)

            # Verify the result is a list with the correct mapped values
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]['label'] == 'POS'
            assert result[1]['label'] == 'NEG'
            mock_model.assert_called_once_with(self.batch_texts)

# # Run the tests
# # coverage run  -m pytest .\tests\unit\test_data\test_sentiment_data.py