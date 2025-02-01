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
            self.args = {
                'text': "This is awesome!",
            }

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

            result = sentiment_data_layer.analyze(self.args['text'])

            # Ensure the method returns an error message
            assert result == {'error': 'An unexpected error occurred while processing the request.'}
            mock_model.assert_called_once_with(self.args['text'])


        def test_analyze_success(self, sentiment_data_layer, mock_model):
            """
            Test that the analyze method returns expected results.
            """
            # Mock the model to return expected results
            mock_model.return_value = ("output_mocked","probabilities_mocked",'POS', 0.9)

            result = sentiment_data_layer.analyze(**self.args)

            # Ensure the method returns the expected results
            assert result == {
                # 'outputs': "output_mocked", # Not Used
                # 'probabilities': "probabilities_mocked",  # Not Used
                'label': 'POS',
                'confidence': 0.9
            }
            mock_model.assert_called_once_with(self.args['text'])
        

# # Run the tests
# # coverage run  -m pytest .\tests\unit\test_data\test_sentiment_data.py