"""
This module contains unit tests for the TranscriptData class.
"""

import pytest
from unittest.mock import MagicMock, patch

# Import the class to be tested
from app.data.transcript_data import TranscriptDataLayer

# Grouped tests for initialization and transcribe method
class TestTranscriptDataLayer:

    # Grouped tests for initialization (__init__ method)
    class TestInitialization:
        @pytest.fixture
        def transcript_data_layer__whisper(self):
            """
            Fixture to set up TranscriptDataLayer instance for testing.
            """
            config = {
                'debug': True,
                'transcription': {
                    'default_model': "whisper",
                    'whisper': {
                        'model_size': "base",
                        'device': 'cpu',
                        'chunk_length_s': 30
                    }
                }
            }
            return TranscriptDataLayer(config)
        
        @pytest.fixture
        def mock_whisper_transcript(self):
            """
            Fixture to mock the 'WhisperTranscript' class.
            """
            with patch('app.data.transcript_data.WhisperTranscript') as mock_whisper_transcript:
                yield mock_whisper_transcript


        def test_init_whisper_model(self,mock_whisper_transcript,transcript_data_layer__whisper):
            """
            Test that TranscriptDataLayer initializes the Whisper model.
            """
            # Ensure the WhisperTranscript is initialized with the correct configuration
            mock_whisper_transcript.assert_called_once_with({
                'debug': True,
                'transcription': {
                    'default_model': "whisper",
                    'whisper': {
                        'model_size': "base",
                        'device': 'cpu',
                        'chunk_length_s': 30
                    }
                }
            })

            # Ensure the model is set to the WhisperTranscript instance
            assert isinstance(transcript_data_layer__whisper.model, mock_whisper_transcript.return_value.__class__)

        def test_init_unsupported_model(self):
            """
            Test that an exception is raised for an unsupported model.
            """
            config = {
                'debug': True,
                'transcription': {'default_model': 'unsupported_model'}
            }
            with pytest.raises(ValueError) as e:
                TranscriptDataLayer(config)

            assert str(e.value) == "Unsupported transcription model: unsupported_model"



    # Define a fixture to initialize the TranscriptDataLayer instance for testing
    @pytest.fixture
    def transcript_data_layer(self):
        """
        Fixture to set up TranscriptDataLayer instance for testing.
        """
        config = {
            'debug': True,
            'transcription': {
                'default_model': "whisper",
                'whisper': {
                    'model_size': "base",
                    'device': 'cpu',
                    'chunk_length_s': 30
                }
            }
        }
        return TranscriptDataLayer(config)
    
    # Grouped tests for the 'transcribe' method
    class TestTranscribe:
        def setup_method(self):
            """
            setup method to prepare the test environment.
            """
            self.args = {
                'audio_file_path': 'test_audio.mp3',
            }

        @pytest.fixture
        def mock_model(self,transcript_data_layer):
            """
            Fixture to mock the self.model
            """
            with patch.object(transcript_data_layer, 'model', MagicMock()) as mock_model:
                yield mock_model

        def test_transcribe_exception(self, transcript_data_layer, mock_model):
            """
            Test that the transcribe method handles errors properly.
            """
            # Mock the model layer to raise an exception
            mock_model.side_effect = Exception("Mocked error")

            result = transcript_data_layer.transcribe(**self.args)

            assert result == {'error': 'An unexpected error occurred while processing the request.'}
            mock_model.assert_called_once_with(self.args['audio_file_path'])

        def test_transcribe_success(self, transcript_data_layer, mock_model):
            """
            Test that the transcribe method returns expected results on success.
            """
            # Mock the model layer to return expected results
            mock_model.return_value = ("Test transcription", [{'timestamp': (0.0, 3.0), 'text': "Test transcription"}])

            result = transcript_data_layer.transcribe(**self.args)

            assert result == {
                'transcription': "Test transcription",
                'chunks': [{'timestamp': (0.0, 3.0), 'text': "Test transcription"}]
            }
            mock_model.assert_called_once_with(self.args['audio_file_path'])

# # Run the tests
# # coverage run  -m pytest .\tests\unit\test_data\test_transcript_data.py    