"""
This module contains the unit tests for the AudioDataLayer class.
"""

import pytest
from unittest.mock import MagicMock, patch

import requests

# Import the class to be tested
from app.data.audio_data import AudioDataLayer

class TestAudioDataLayer:
    @pytest.fixture
    def audio_data_layer(self):
        """
        Fixture to set up AudioDataLayer instance for testing.
        """
        config = {'debug': True}
        return AudioDataLayer(config)
    

    # Grouped tests for the 'fetch_audio' method
    class TestFetchAudio:
        def setup_method(self):
            """
            Setup method to prepare the test environment.
            """
            self.args = {
                'url': 'http://example.com/audio.mp3',
            }

        # Mock Methods
        @pytest.fixture
        def mock_requests__get(self):
            """
            Fixture to mock the 'requests.get' method.
            """
            with patch('app.data.audio_data.requests.get') as mock_requests__get:
                yield mock_requests__get

        @pytest.fixture
        def mock_io__BytesIO(self):
            """
            Fixture to mock the 'io.BytesIO' method.
            """
            with patch('app.data.audio_data.BytesIO') as mock_io_BytesIO:
                mock_io_BytesIO.return_value = 'mock_bytes_io'
                yield mock_io_BytesIO

        @pytest.fixture
        def mock_audio_segment__from_file(self):
            """
            Fixture to mock the 'AudioSegment.from_file' method.
            """
            with patch('app.data.audio_data.AudioSegment.from_file') as mock_audio_segment__from_file:
                mock_audio_segment__from_file.return_value = 'mock_audio_data'
                yield mock_audio_segment__from_file
        
        @pytest.fixture
        def mock_os__path_exists(self):
            """
            Fixture to mock the 'os.path.exists' method.
            """
            with patch('app.data.audio_data.os.path.exists') as mock_os__path_exists:
                yield mock_os__path_exists
        
        @pytest.fixture
        def mock_os__path_isfile(self):
            """
            Fixture to mock the 'os.path.isfile' method.
            """
            with patch('app.data.audio_data.os.path.isfile') as mock_os__path_isfile:
                yield mock_os__path_isfile


        def test_fetch_audio_from_url_failure(self, audio_data_layer, mock_requests__get):
            """
            Test failed fetch from a URL due to an HTTP error.
            """
            # Mock the HTTP request response
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_requests__get.return_value = mock_response


            payload = self.args
            result = audio_data_layer.fetch_audio(**payload)

            # Assert that mock_requests__get is called with the correct URL
            mock_requests__get.assert_called_once_with(payload['url'])
            
            assert result == {'error': f'An error occurred during the HTTP request: HTTP status: {mock_response.status_code}'}

            def test_fetch_audio_exception(self, audio_data_layer, mock_requests__get):
                """
                Test an exception during audio fetch.
                """
                # Mock the 'requests.get' method to raise a RequestException
                mock_requests__get.side_effect = requests.exceptions.RequestException('mock exception')

                payload = self.args
                result = audio_data_layer.fetch_audio(**payload)

                # Assert that mock_requests__get is called with the correct URL
                mock_requests__get.assert_called_once_with(payload['url'])

                assert result == {'error': 'An error occurred during the HTTP request: mock exception'}

        def test_fetch_audio_from_url_success(self, audio_data_layer, mock_requests__get,mock_io__BytesIO,mock_audio_segment__from_file):
            """
            Test successful fetch from a URL.
            """
            # Mock the HTTP request response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'audio_data'
            mock_requests__get.return_value = mock_response

            payload = self.args
            result = audio_data_layer.fetch_audio(**payload)

            # Assert that mock_requests_get is called with the correct URL
            mock_requests__get.assert_called_once_with(payload['url'])

            # Assert the mock_io__BytesIO is called with the correct audio data
            mock_io__BytesIO.assert_called_once_with(mock_response.content)

            # Assert that mock_audio_segment__from_file is called with the correct BytesIO object
            mock_audio_segment__from_file.assert_called_once_with('mock_bytes_io')

            assert result == 'mock_audio_data'

        

        def test_fetch_audio_from_local_path_success(self, audio_data_layer, mock_audio_segment__from_file,mock_os__path_exists,mock_os__path_isfile):
            """
            Test successful fetch from a local file path.
            """
            # Mock the 'os.path.exists' and 'os.path.isfile' methods
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = True

            payload= self.args
            payload['url'] = '/dummy/path/to/audio.mp3'
            result = audio_data_layer.fetch_audio(**payload)

            # Assert that mock_os__path_exists is called with the correct URL
            mock_os__path_exists.assert_called_once_with(payload['url'])

            # Assert that mock_os__path_isfile is called with the correct URL
            mock_os__path_isfile.assert_called_once_with(payload['url'])

            # Assert that mock_audio_segment__from_file is called with the correct URL
            mock_audio_segment__from_file.assert_called_once_with(payload['url'])

            assert result == 'mock_audio_data'

        def test_fetch_audio_from_invalid_path(self, audio_data_layer, mock_os__path_exists,mock_os__path_isfile):
            """
            Test failed fetch from an invalid path.
            """
            # Mock the 'os.path.exists' and 'os.path.isfile' methods
            mock_os__path_exists.return_value = False
            mock_os__path_isfile.return_value = False

            payload = self.args
            payload['url'] = '/dummy/path/to/audio.mp3'
            result = audio_data_layer.fetch_audio(**payload)

            # Assert that mock_os__path_exists is called with the correct URL
            mock_os__path_exists.assert_called_once_with(payload['url'])

            # Assert that mock_os__path_isfile is called with the correct URL
            mock_os__path_isfile.assert_not_called()

            # Assert that mock_os__path_exists is called with the correct URL
            mock_os__path_exists.assert_called_once_with(payload['url'])

            # Assert that mock_os__path_isfile is called with the correct URL
            mock_os__path_isfile.assert_not_called()

            assert result == {'error': 'Provided url is neither a valid URL nor a valid file path.'}


        def test_fetch_audio_exception(self, audio_data_layer, mock_os__path_exists):
            """
            Test an exception during audio fetch. raised by any method (urlparse, os.path.exists, os.path.isfile)
            """
            # Mock the 'os.path.exists' method to raise an exception
            mock_os__path_exists.side_effect = Exception('mock exception')

            payload = self.args
            payload['url'] = '/dummy/path/to/audio.mp3'
            result = audio_data_layer.fetch_audio(**payload)

            # Assert that mock_os__path_exists is called with the correct URL
            mock_os__path_exists.assert_called_once_with(payload['url'])

            assert result == {'error': 'An unexpected error occurred while processing the request.'}

    
# # Run the tests
# coverage run  -m pytest .\tests\unit\test_data\test_audio_data_.py