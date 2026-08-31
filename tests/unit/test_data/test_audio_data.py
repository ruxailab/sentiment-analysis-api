"""
This module contains the unit tests for the AudioDataLayer class.
"""

import pytest
from unittest.mock import MagicMock, mock_open, patch

import requests

from app.data.audio_data import AudioDataLayer


class TestAudioDataLayer:
    @pytest.fixture
    def audio_data_layer(self):
        config = {
            'debug': True,
            'audio': {
                'download_timeout_seconds': 30,
                'max_download_bytes': 1024,
            },
        }
        return AudioDataLayer(config)

    class TestResolveAudioSource:
        def setup_method(self):
            self.args = {
                'url': 'http://example.com/audio.mp3',
            }

        @pytest.fixture
        def mock_requests__get(self):
            with patch('app.data.audio_data.requests.get') as mock_requests__get:
                yield mock_requests__get

        @pytest.fixture
        def mock_os__path_exists(self):
            with patch('app.data.audio_data.os.path.exists') as mock_os__path_exists:
                yield mock_os__path_exists

        @pytest.fixture
        def mock_os__path_isfile(self):
            with patch('app.data.audio_data.os.path.isfile') as mock_os__path_isfile:
                yield mock_os__path_isfile

        @pytest.fixture
        def mock_mkstemp(self):
            with patch('app.data.audio_data.tempfile.mkstemp') as mock_mkstemp:
                mock_mkstemp.return_value = (3, '/tmp/mock_audio.mp3')
                yield mock_mkstemp

        @pytest.fixture
        def mock_os__close(self):
            with patch('app.data.audio_data.os.close') as mock_os__close:
                yield mock_os__close

        def test_resolve_audio_from_url_failure(self, audio_data_layer, mock_requests__get, mock_mkstemp, mock_os__close):
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_requests__get.return_value = mock_response

            with patch.object(audio_data_layer, '_safe_remove') as mock_safe_remove:
                result = audio_data_layer.resolve_audio_source(**self.args)

            mock_requests__get.assert_called_once_with(
                self.args['url'],
                stream=True,
                timeout=30,
            )
            mock_safe_remove.assert_called_once_with('/tmp/mock_audio.mp3')
            assert result == {
                'error': 'An error occurred during the HTTP request: HTTP status: 404'
            }

        def test_resolve_audio_from_url_request_exception(
            self,
            audio_data_layer,
            mock_requests__get,
            mock_mkstemp,
            mock_os__close,
        ):
            mock_requests__get.side_effect = requests.exceptions.RequestException('mock exception')

            with patch.object(audio_data_layer, '_safe_remove') as mock_safe_remove:
                result = audio_data_layer.resolve_audio_source(**self.args)

            mock_requests__get.assert_called_once_with(
                self.args['url'],
                stream=True,
                timeout=30,
            )
            mock_safe_remove.assert_called_once_with('/tmp/mock_audio.mp3')
            assert result == {
                'error': 'An error occurred during the HTTP request: mock exception'
            }

        def test_resolve_audio_from_url_success(
            self,
            audio_data_layer,
            mock_requests__get,
            mock_mkstemp,
            mock_os__close,
        ):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_content.return_value = [b'audio', b'_data']
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_requests__get.return_value = mock_response

            with patch('app.data.audio_data.open', mock_open()) as mocked_open:
                result = audio_data_layer.resolve_audio_source(**self.args)

            mock_requests__get.assert_called_once_with(
                self.args['url'],
                stream=True,
                timeout=30,
            )
            mocked_open.assert_called_once_with('/tmp/mock_audio.mp3', 'wb')
            assert result == {
                'path': '/tmp/mock_audio.mp3',
                'is_temporary': True,
            }

        def test_resolve_audio_from_url_exceeds_max_size(
            self,
            audio_data_layer,
            mock_requests__get,
            mock_mkstemp,
            mock_os__close,
        ):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_content.return_value = [b'x' * 2048]
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = False
            mock_requests__get.return_value = mock_response

            with patch('app.data.audio_data.open', mock_open()), \
                    patch.object(audio_data_layer, '_safe_remove') as mock_safe_remove:
                result = audio_data_layer.resolve_audio_source(**self.args)

            mock_safe_remove.assert_called_once_with('/tmp/mock_audio.mp3')
            assert result == {
                'error': 'Downloaded file exceeds max size of 1024 bytes.'
            }

        def test_resolve_audio_from_local_path_success(
            self,
            audio_data_layer,
            mock_os__path_exists,
            mock_os__path_isfile,
        ):
            mock_os__path_exists.return_value = True
            mock_os__path_isfile.return_value = True

            payload = {'url': '/dummy/path/to/audio.mp3'}
            result = audio_data_layer.resolve_audio_source(**payload)

            mock_os__path_exists.assert_called_once_with(payload['url'])
            mock_os__path_isfile.assert_called_once_with(payload['url'])
            assert result == {
                'path': payload['url'],
                'is_temporary': False,
            }

        def test_resolve_audio_from_invalid_path(
            self,
            audio_data_layer,
            mock_os__path_exists,
            mock_os__path_isfile,
        ):
            mock_os__path_exists.return_value = False
            mock_os__path_isfile.return_value = False

            payload = {'url': '/dummy/path/to/audio.mp3'}
            result = audio_data_layer.resolve_audio_source(**payload)

            mock_os__path_exists.assert_called_once_with(payload['url'])
            mock_os__path_isfile.assert_not_called()
            assert result == {
                'error': 'Provided url is neither a valid URL nor a valid file path.'
            }

        def test_resolve_audio_exception(self, audio_data_layer, mock_os__path_exists):
            mock_os__path_exists.side_effect = Exception('mock exception')

            payload = {'url': '/dummy/path/to/audio.mp3'}
            result = audio_data_layer.resolve_audio_source(**payload)

            mock_os__path_exists.assert_called_once_with(payload['url'])
            assert result == {
                'error': 'An unexpected error occurred while processing the request.'
            }
