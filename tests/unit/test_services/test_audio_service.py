import pytest
from unittest.mock import MagicMock, patch

# Service to be tested
from app.services.audio_service import AudioService

class TestAudioService:
    @pytest.fixture
    def audio_service(self):
        """
        Fixture to set up AudioService instance for testing.
        """
        return AudioService(static_folder="mock_static")
    
    # Grouped tests for the 'extract_audio' method
    class TestExtractAudio:
        def setup_method(self):
            """Set up before each test for ExtractAudio class."""
            self.args = {
                "url": "https://example.com/audio.mp3",
                "start_time_ms": 10,
                "end_time_ms": 20,
                "user_id": "user123"
            }

        # Mock Methods of AudioDataLayer
        @pytest.fixture
        def mock_audio_data_layer__fetch_audio(self):
            """Fixture to mock the fetch_audio method."""
            with patch('app.services.audio_service.AudioDataLayer.fetch_audio') as mock:
                yield mock

        # Mock Private Methods
        @pytest.fixture
        def mock__save_audio(self):
            """Fixture to mock the _save_audio method."""
            with patch('app.services.audio_service.AudioService._save_audio') as mock:
                mock.return_value = "mock_audio_path"
                yield mock


        def test_extract_audio_negative_start_time(self, audio_service, mock_audio_data_layer__fetch_audio):
            """
            Test for extracting audio with a negative start time.
            """
            args = self.args.copy()
            args['start_time_ms'] = -1000
            result = audio_service.extract_audio(**args)
        
            # Verify fetch_audio was not called
            mock_audio_data_layer__fetch_audio.assert_not_called()

            assert result == {
                "error": "Start time must be a non-negative integer.",
            }
            

        def test_extract_audio__fetch_audio_failure(self,audio_service, mock_audio_data_layer__fetch_audio):
            """
            Test for extracting audio when fetch_audio fails.
            """
            mock_audio_data_layer__fetch_audio.return_value = {"error": "Failed to fetch audio."}

            payload = self.args.copy()
            result = audio_service.extract_audio(**payload)


            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            assert result == {
                "error": "Failed to fetch audio."
            }

        def test_extract_audio__fetch_audio_exception(self,audio_service, mock_audio_data_layer__fetch_audio):
            """
            Test for extracting audio when fetch_audio raises an exception.
            """
            mock_audio_data_layer__fetch_audio.side_effect = Exception("An error occurred.")

            payload = self.args.copy()
            result = audio_service.extract_audio(**payload)

            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            assert result == {
                "error": "An unexpected error occurred while processing the request."
            }

        
        def test_extract_audio_end_time_ms_is_none(self,audio_service, mock_audio_data_layer__fetch_audio, mock__save_audio):
            """
            Test for extracting audio when end_time_ms is not passed.
            """
            # Create a mock object
            mock_audio = MagicMock()
            mocked_audio_length = 10000
            
            # Mock fetch_audio to return the mock audio object
            mock_audio_data_layer__fetch_audio.return_value = mock_audio
            
            # Mock the length of the audio
            mock_audio_data_layer__fetch_audio.return_value.__len__.return_value = mocked_audio_length
            
            # mock_audio_data_layer__fetch_audio.__getitem__.side_effect = lambda s: f"mocked_audio[{s.start}:{s.stop}]"
            expected_audio_slice = f"mocked_audio[{self.args['start_time_ms']}:{mocked_audio_length}]"
            mock_audio.__getitem__.return_value  = expected_audio_slice
               
            payload = self.args.copy()
            del payload['end_time_ms']
            result = audio_service.extract_audio(**payload)

            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            # Verify the slice was called with the expected arguments
            mock_audio.__getitem__.assert_called_once_with(slice(self.args['start_time_ms'], mocked_audio_length))

            # Verify Save Audio was called with the mocked audio segment
            mock__save_audio.assert_called_once_with(expected_audio_slice, payload['user_id'])
   
            assert result == {
                "audio_path": "mock_audio_path",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": mocked_audio_length
            }

        def test_extract_audio_success_end_time_ms_gt_len_audio(self,audio_service, mock_audio_data_layer__fetch_audio, mock__save_audio):
            """
            Test for extracting audio when end_time_ms > len(audio).
            """
            # Create a mock object
            mock_audio = MagicMock()
            mocked_audio_length = 10000
            
            # Mock fetch_audio to return the mock audio object
            mock_audio_data_layer__fetch_audio.return_value = mock_audio
            
            # Mock the length of the audio
            mock_audio_data_layer__fetch_audio.return_value.__len__.return_value = mocked_audio_length


            # mock_audio_data_layer__fetch_audio.__getitem__.side_effect = lambda s: f"mocked_audio[{s.start}:{s.stop}]"
            expected_audio_slice = f"mocked_audio[{self.args['start_time_ms']}:temp_end_time]"
            mock_audio.__getitem__.return_value  = expected_audio_slice
               
            payload = self.args.copy()
            payload['end_time_ms'] = 200000
            result = audio_service.extract_audio(**payload)

            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            # Verify the slice was called with the expected arguments
            mock_audio.__getitem__.assert_called_once_with(slice(self.args['start_time_ms'], mocked_audio_length))

            # Verify Save Audio was called with the mocked audio segment
            mock__save_audio.assert_called_once_with(expected_audio_slice, payload['user_id'])

            assert result == {
                "audio_path": "mock_audio_path",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": mocked_audio_length
            }

        def test_extract_audio_success_end_time_ms_lt_start_time(self,audio_service, mock_audio_data_layer__fetch_audio):
            """
            Test for extracting audio when end_time_ms < start_time_ms.
            """
            # Create a mock object
            mock_audio = MagicMock()
            mocked_audio_length = 10000
            
            # Mock fetch_audio to return the mock audio object
            mock_audio_data_layer__fetch_audio.return_value = mock_audio
            
            # Mock the length of the audio
            mock_audio_data_layer__fetch_audio.return_value.__len__.return_value = mocked_audio_length

            payload = self.args.copy()
            payload['start_time_ms'] = 2000
            payload['end_time_ms'] = 1000
            result = audio_service.extract_audio(**payload)

            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            assert result == {
                "error": "End time must not be less than start time."
            }

        def test_extract_audio_success(self,audio_service, mock_audio_data_layer__fetch_audio, mock__save_audio):
            """
            Test for extracting audio successfully with all & valid arguments. :D
            """
            # Create a mock object
            mock_audio = MagicMock()
            mocked_audio_length = 10000
            
            # Mock fetch_audio to return the mock audio object
            mock_audio_data_layer__fetch_audio.return_value = mock_audio
            
            # Mock the length of the audio
            mock_audio_data_layer__fetch_audio.return_value.__len__.return_value = mocked_audio_length

            # mock_audio_data_layer__fetch_audio.__getitem__.side_effect = lambda s: f"mocked_audio[{s.start}:{s.stop}]"
            expected_audio_slice = f"mocked_audio[{self.args['start_time_ms']}:{self.args['end_time_ms']}]"
            mock_audio.__getitem__.return_value  = expected_audio_slice
               
            payload = self.args.copy()
            result = audio_service.extract_audio(**payload)

            # Verify fetch_audio was called with the correct URL
            mock_audio_data_layer__fetch_audio.assert_called_once_with(payload['url'])

            # Verify the slice was called with the expected arguments
            mock_audio.__getitem__.assert_called_once_with(slice(self.args['start_time_ms'], self.args['end_time_ms']))

            # Verify Save Audio was called with the mocked audio segment
            mock__save_audio.assert_called_once_with(expected_audio_slice, payload['user_id'])

            assert result == {
                "audio_path": "mock_audio_path",
                "start_time_ms": self.args['start_time_ms'],
                "end_time_ms": self.args['end_time_ms']
            }

    # Grouped tests for the '_save_audio' method
    class TestSaveAudio:
        """
        Test cases for the _save_audio method.
        """
        @pytest.fixture(autouse=True)
        def setup_method(self):
            """
            Set up before each test.
            """
            self.mock_audio_segment = MagicMock()
            self.mock_audio_segment.export = MagicMock()

            self.args = {
                "audio": self.mock_audio_segment,
                "user_id": "user123"
            }

        # Mock Methods
        @pytest.fixture
        def mock_uuid(self):
            """Fixture to mock the uuid module."""
            with patch('uuid.uuid4') as mock:
                mock.return_value = "mock_uuid"
                yield mock

        @pytest.fixture
        def mock_os__makedirs(self):
            """Fixture to mock the os.makedirs method."""
            with patch('os.makedirs') as mock:
                yield mock

        def test_save_audio_with_user_id(self,audio_service,mock_uuid, mock_os__makedirs):
            """
            Test saving audio with a user_id.
            """
            payload = self.args.copy()
            mock_audio_segment = payload['audio']
            result = audio_service._save_audio(**payload)

            # Verify the makedirs was called with the correct arguments
            mock_os__makedirs.assert_called_once_with(f"{audio_service.static_folder}/user123", exist_ok=True)

            # Verify the export was called with the correct arguments
            mock_audio_segment.export.assert_called_once_with(f"{audio_service.static_folder}/user123/mock_uuid_audio.mp3", format="mp3")

            assert result == f"{audio_service.static_folder}/user123/mock_uuid_audio.mp3"

        def test_save_audio_without_user_id(self,audio_service,mock_uuid,mock_os__makedirs):
            """
            Test saving audio without a user_id.
            """
            payload = self.args.copy()
            del payload['user_id']
            mock_audio_segment = payload['audio']
            result = audio_service._save_audio(**payload)

            # Verify the makedirs was called with the correct arguments
            mock_os__makedirs.assert_called_once_with(audio_service.static_folder, exist_ok=True)

            # Verify the export was called with the correct arguments
            mock_audio_segment.export.assert_called_once_with(f"{audio_service.static_folder}/mock_uuid_audio.mp3", format="mp3")

            assert result == f"{audio_service.static_folder}/mock_uuid_audio.mp3"

        def test_save_audio_directory_creation_failure(self,audio_service,mock_os__makedirs):
            """
            Test saving audio when directory creation fails. (OSError) (with user_id) :D
            """
            mock_os__makedirs.side_effect = OSError("Failed to create directory")

            payload = self.args.copy()
            mock_audio_segment = payload['audio']

            with pytest.raises(OSError, match="Failed to create directory"):
                audio_service._save_audio(**payload)

            # Assert the directory creation was attempted
            mock_os__makedirs.assert_called_once_with(f"{audio_service.static_folder}/user123", exist_ok=True)

        def test_save_audio_export_failure(self,audio_service,mock_uuid,mock_os__makedirs):
            """
            Test saving audio when exporting the audio fails. (Exception) with user_id :D
            """
            payload = self.args.copy()
            mock_audio_segment = payload['audio']
            mock_audio_segment.export.side_effect = Exception("Failed to export audio")

            with pytest.raises(Exception, match="Failed to export audio"):
                audio_service._save_audio(**payload)

            # Assert the export was called
            mock_audio_segment.export.assert_called_once_with(f"{audio_service.static_folder}/user123/mock_uuid_audio.mp3", format="mp3")



# # Run:
# coverage run  -m pytest .\tests\unit\test_services\test_audio_service.py