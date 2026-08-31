"""
This Module contains the unit tests for the AudioService class.
"""

import pytest
from unittest.mock import patch

from app.services.audio_service import AudioService
from app.utils.ffmpeg_audio import FFmpegError


class TestAudioService:
    @pytest.fixture
    def audio_service(self):
        return AudioService(static_folder="mock_static")

    class TestExtractAudio:
        def setup_method(self):
            self.args = {
                "url": "https://example.com/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20,
                "user_id": "user123"
            }

        @pytest.fixture
        def mock_resolve_audio_source(self):
            with patch('app.services.audio_service.AudioDataLayer.resolve_audio_source') as mock:
                yield mock

        @pytest.fixture
        def mock_get_duration_ms(self):
            with patch('app.services.audio_service.get_duration_ms') as mock:
                mock.return_value = 10000
                yield mock

        @pytest.fixture
        def mock_extract_audio_segment(self):
            with patch('app.services.audio_service.extract_audio_segment') as mock:
                yield mock

        @pytest.fixture
        def mock_build_output_path(self):
            with patch('app.services.audio_service.AudioService._build_output_path') as mock:
                mock.return_value = "mock_static/user123/mock_uuid_audio.wav"
                yield mock

        def test_extract_audio_negative_start_time(self, audio_service, mock_resolve_audio_source):
            args = self.args.copy()
            args['start_time_ms'] = -1000
            result = audio_service.extract_audio(**args)

            mock_resolve_audio_source.assert_not_called()
            assert result == {
                "error": "Start time must be a non-negative number.",
            }

        def test_extract_audio_end_before_start_without_probe(self, audio_service, mock_resolve_audio_source):
            payload = self.args.copy()
            payload['start_time_ms'] = 2000
            payload['end_time_ms'] = 1000
            result = audio_service.extract_audio(**payload)

            mock_resolve_audio_source.assert_not_called()
            assert result == {
                "error": "End time must not be less than start time."
            }

        def test_extract_audio__resolve_failure(self, audio_service, mock_resolve_audio_source):
            mock_resolve_audio_source.return_value = {"error": "Failed to fetch audio."}

            payload = self.args.copy()
            result = audio_service.extract_audio(**payload)

            mock_resolve_audio_source.assert_called_once_with(payload['url'])
            assert result == {
                "error": "Failed to fetch audio."
            }

        def test_extract_audio__resolve_exception(self, audio_service, mock_resolve_audio_source):
            mock_resolve_audio_source.side_effect = Exception("An error occurred.")

            payload = self.args.copy()
            result = audio_service.extract_audio(**payload)

            mock_resolve_audio_source.assert_called_once_with(payload['url'])
            assert result == {
                "error": "An unexpected error occurred while processing the request."
            }

        def test_extract_audio_end_time_ms_is_none(
            self,
            audio_service,
            mock_resolve_audio_source,
            mock_get_duration_ms,
            mock_extract_audio_segment,
            mock_build_output_path,
        ):
            mock_resolve_audio_source.return_value = {
                'path': '/tmp/source.mp4',
                'is_temporary': False,
            }

            payload = self.args.copy()
            del payload['end_time_ms']
            result = audio_service.extract_audio(**payload)

            mock_resolve_audio_source.assert_called_once_with(payload['url'])
            mock_get_duration_ms.assert_called_once_with('/tmp/source.mp4')
            mock_build_output_path.assert_called_once_with(payload['user_id'])
            mock_extract_audio_segment.assert_called_once_with(
                input_path='/tmp/source.mp4',
                output_path='mock_static/user123/mock_uuid_audio.wav',
                start_time_ms=self.args['start_time_ms'],
                end_time_ms=10000,
                sample_rate=audio_service.output_sample_rate,
            )
            assert result == {
                "audio_path": "mock_static/user123/mock_uuid_audio.wav",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": 10000,
            }

        def test_extract_audio_success_end_time_ms_gt_len_audio(
            self,
            audio_service,
            mock_resolve_audio_source,
            mock_get_duration_ms,
            mock_extract_audio_segment,
            mock_build_output_path,
        ):
            mock_resolve_audio_source.return_value = {
                'path': '/tmp/source.mp4',
                'is_temporary': False,
            }

            payload = self.args.copy()
            payload['end_time_ms'] = 200000
            result = audio_service.extract_audio(**payload)

            mock_extract_audio_segment.assert_called_once_with(
                input_path='/tmp/source.mp4',
                output_path='mock_static/user123/mock_uuid_audio.wav',
                start_time_ms=self.args['start_time_ms'],
                end_time_ms=10000,
                sample_rate=audio_service.output_sample_rate,
            )
            assert result == {
                "audio_path": "mock_static/user123/mock_uuid_audio.wav",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": 10000,
            }

        def test_extract_audio_success(
            self,
            audio_service,
            mock_resolve_audio_source,
            mock_get_duration_ms,
            mock_extract_audio_segment,
            mock_build_output_path,
        ):
            mock_resolve_audio_source.return_value = {
                'path': '/tmp/source.mp4',
                'is_temporary': True,
            }

            with patch.object(audio_service, '_safe_remove') as mock_safe_remove:
                payload = self.args.copy()
                result = audio_service.extract_audio(**payload)

            mock_extract_audio_segment.assert_called_once_with(
                input_path='/tmp/source.mp4',
                output_path='mock_static/user123/mock_uuid_audio.wav',
                start_time_ms=self.args['start_time_ms'],
                end_time_ms=self.args['end_time_ms'],
                sample_rate=audio_service.output_sample_rate,
            )
            mock_safe_remove.assert_called_once_with('/tmp/source.mp4')
            assert result == {
                "audio_path": "mock_static/user123/mock_uuid_audio.wav",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": self.args['end_time_ms'],
            }

        def test_extract_audio_ffmpeg_error(
            self,
            audio_service,
            mock_resolve_audio_source,
            mock_get_duration_ms,
            mock_extract_audio_segment,
            mock_build_output_path,
        ):
            mock_resolve_audio_source.return_value = {
                'path': '/tmp/source.mp4',
                'is_temporary': False,
            }
            mock_extract_audio_segment.side_effect = FFmpegError("ffmpeg failed")

            result = audio_service.extract_audio(**self.args.copy())
            assert result == {
                "error": "An unexpected error occurred while processing the request."
            }

    class TestBuildOutputPath:
        @pytest.fixture
        def mock_uuid(self):
            with patch('uuid.uuid4') as mock:
                mock.return_value = "mock_uuid"
                yield mock

        @pytest.fixture
        def mock_os__makedirs(self):
            with patch('os.makedirs') as mock:
                yield mock

        def test_build_output_path_with_user_id(self, audio_service, mock_uuid, mock_os__makedirs):
            result = audio_service._build_output_path(user_id="user123")

            mock_os__makedirs.assert_called_once_with(f"{audio_service.static_folder}/user123", exist_ok=True)
            assert result == f"{audio_service.static_folder}/user123/mock_uuid_audio.wav"

        def test_build_output_path_without_user_id(self, audio_service, mock_uuid, mock_os__makedirs):
            result = audio_service._build_output_path()

            mock_os__makedirs.assert_called_once_with(audio_service.static_folder, exist_ok=True)
            assert result == f"{audio_service.static_folder}/mock_uuid_audio.wav"

        def test_build_output_path_directory_creation_failure(self, audio_service, mock_os__makedirs):
            mock_os__makedirs.side_effect = OSError("Failed to create directory")

            with pytest.raises(OSError, match="Failed to create directory"):
                audio_service._build_output_path(user_id="user123")

            mock_os__makedirs.assert_called_once_with(f"{audio_service.static_folder}/user123", exist_ok=True)
