"""
This Module contains the unit tests for the transcript service.
"""

import pytest
from unittest.mock import patch

from app.services.transcript_service import TranscriptService


class TestTranscriptService:
    """Test suite for the TranscriptService class."""
    @pytest.fixture()
    def transcript_service(self):
        """Fixture for creating an instance of the TranscriptService class."""
        return TranscriptService()
    
    # Grouped tests for the 'transcribe' method
    class TestTranscribe:
        def setup_method(self):
            """Setup method for each test."""
            self.args = {
                'audio_file_path': 'audio.mp3'
            }

        # Mock Methods of TranscriptDataLayer
        @pytest.fixture
        def mock_transcript_data_layer__transcribe(self):
            """Fixture for mocking the TranscriptDataLayer.transcribe method."""
            with patch('app.services.transcript_service.TranscriptDataLayer.transcribe') as mock:
                yield mock

        # Mock Methods
        @pytest.fixture
        def mock_os__path_exists(self):
            """Fixture for mocking the os.path.exists method."""
            with patch('os.path.exists') as mock:
                yield mock

        @pytest.fixture
        def mock_os__path_isfile(self):
            """Fixture for mocking the os.path.isfile method."""
            with patch('os.path.isfile') as mock:
                yield mock
        

        def test_transcribe_file_not_exist(self, transcript_service, mock_os__path_exists):
            """
            Test for when the file does not exist.
            """
            args = self.args.copy()
            mock_os__path_exists.return_value = False

            result = transcript_service.transcribe(**args)

            assert result == {
                'error': f'Audio file not found: {args["audio_file_path"]}'
            }
            mock_os__path_exists.assert_called_once_with(args['audio_file_path'])

        def test_transcribe_file_not_a_file(self, transcript_service, mock_os__path_exists, mock_os__path_isfile):
            """
            Test for when the path is not a file.
            """
            args = self.args.copy()
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = False

            result = transcript_service.transcribe(**args)

            assert result == {
                'error': f'Audio file not found: {args["audio_file_path"]}'
            }
            mock_os__path_exists.assert_called_once_with(args['audio_file_path'])
            mock_os__path_isfile.assert_called_once_with(args['audio_file_path'])

        def test_transcribe__transcript_data_layer_transcribe_failure(self, transcript_service, mock_os__path_exists, mock_os__path_isfile, mock_transcript_data_layer__transcribe):
            """
            Test for when the TranscriptDataLayer.transcribe method returns an error.
            """
            args = self.args.copy()
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = True

            mock_transcript_data_layer__transcribe.return_value = {
                'error': 'Mocked service error'
            }

            result = transcript_service.transcribe(**args)

            assert result == {
                'error': 'Mocked service error'
            }
            mock_os__path_exists.assert_called_once_with(args['audio_file_path'])
            mock_os__path_isfile.assert_called_once_with(args['audio_file_path'])
            mock_transcript_data_layer__transcribe.assert_called_once_with(args['audio_file_path'])

        def test_transcribe__transcript_data_layer_transcribe_exception(self, transcript_service, mock_os__path_exists, mock_os__path_isfile, mock_transcript_data_layer__transcribe):
            """
            Test for when the TranscriptDataLayer.transcribe method raises an exception.
            """
            args = self.args.copy()
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = True

            mock_transcript_data_layer__transcribe.side_effect = Exception('Mocked exception')

            result = transcript_service.transcribe(**args)

            assert result == {
                'error': 'An unexpected error occurred while processing the request.'
            }

        def test_transcribe_success(self, transcript_service, mock_os__path_exists, mock_os__path_isfile, mock_transcript_data_layer__transcribe):
            """
            Test for a successful transcription.
            """
            args = self.args.copy()
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = True

            mocked_transcription = "Hello, world!"
            mocked_chunks = [{'timestamp': 'Mocked_timestamp', 'text': "Hello, world!"}]
            mock_transcript_data_layer__transcribe.return_value = {
                'transcription': mocked_transcription,
                'chunks': mocked_chunks
            }

            result = transcript_service.transcribe(**args)

            assert result == {
                'transcription': mocked_transcription,
                'chunks': mocked_chunks
            }
            mock_os__path_exists.assert_called_once_with(args['audio_file_path'])
            mock_os__path_isfile.assert_called_once_with(args['audio_file_path'])
            mock_transcript_data_layer__transcribe.assert_called_once_with(args['audio_file_path'])

# # Run:
# coverage run  -m pytest .\tests\unit\test_services\test_transcript_service.py



            







      
            