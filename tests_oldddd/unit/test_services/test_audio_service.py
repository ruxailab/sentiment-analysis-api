import pytest
from unittest.mock import MagicMock
from unittest import mock

import os
from pydub import AudioSegment

# Import the class to be tested
from app.services.audio_service import AudioService

# Test setup
@pytest.fixture
def audio_service():
    """
    Fixture for initializing the AudioService.
    """
    return AudioService()


@pytest.fixture
def mock_audio_segment():
    # Create a mock for AudioSegment
    return MagicMock(spec=AudioSegment)


def test_extract_audio_non_int_start_time(audio_service):
    """
    Test extracting audio with a non-integer start time.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = "1000"
    mock_end_time_ms = 4000

    # Call the extract_audio method
    result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

    # Assert the result
    assert 'error' in result
    assert result['error'] == 'Start time must be a non-negative integer.'

def test_extract_audio_negative_start_time(audio_service):
    """
    Test extracting audio with a negative start time.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = -1000
    mock_end_time_ms = 4000

    # Call the extract_audio method
    result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

    # Assert the result
    assert 'error' in result
    assert result['error'] == 'Start time must be a non-negative integer.'

def test_extract_audio_success(audio_service,mock_audio_segment):
    """
    Test extracting audio successfully.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = 1000
    mock_end_time_ms = 4000
    mock_audio_length = 2000

    # Mock the fetch_audio method to return a mock audio segment
    audio_service.audio_data_layer.fetch_audio = mock.MagicMock(return_value=mock_audio_segment)

    # Mock the _save_audio method
    with mock.patch.object(audio_service, '_save_audio', return_value="path/to/file.mp3") as mock_save_audio:      
        # Mock the length of the audio
        mock_audio_segment.__len__.return_value = mock_audio_length

        # Call the method under test
        result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

        # Assert the fetch_audio method was called with the correct arguments
        audio_service.audio_data_layer.fetch_audio.assert_called_once_with(mock_url)

        # Assert the _save_audio method was called
        mock_save_audio.assert_called_once_with(mock_audio_segment[mock_start_time_ms:mock_end_time_ms], None)

        print("Result: ",result)

        # Assert the result
        assert result == {
            "audio_path": "path/to/file.mp3",
            "start_time_ms": mock_start_time_ms,
            "end_time_ms": mock_audio_length
        }


def test_extract_audio_success_end_time_ms_gt_len_audio(audio_service,mock_audio_segment):
    """
    Test extracting audio successfully when end_time_ms > len(audio).
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = 1000
    mock_end_time_ms = 4000
    mock_audio_length = 2000

    # Mock the fetch_audio method to return a mock audio segment
    audio_service.audio_data_layer.fetch_audio = mock.MagicMock(return_value=mock_audio_segment)

    # Mock the _save_audio method
    with mock.patch.object(audio_service, '_save_audio', return_value="path/to/file.mp3") as mock_save_audio:      
        # Mock the length of the audio
        mock_audio_segment.__len__.return_value = mock_audio_length

        # Call the method under test
        result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

        # Assert the fetch_audio method was called with the correct arguments
        audio_service.audio_data_layer.fetch_audio.assert_called_once_with(mock_url)

        # Assert the _save_audio method was called with the correct arguments
        mock_save_audio.assert_called_once_with(mock_audio_segment[mock_start_time_ms:mock_audio_length], None)

        # Assert the result
        assert result == {
            "audio_path": "path/to/file.mp3",
            "start_time_ms": mock_start_time_ms,
            "end_time_ms": mock_audio_length
        }
 
def test_extract_audio_success_end_time_non_passed(audio_service,mock_audio_segment):
    """
    Test extracting audio successfully when end_time_ms is not passed.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = 1000
    # mock_end_time_ms = None
    mock_audio_length = 2000

    # Mock the fetch_audio method to return a mock audio segment
    audio_service.audio_data_layer.fetch_audio = mock.MagicMock(return_value=mock_audio_segment)

    # Mock the _save_audio method
    with mock.patch.object(audio_service, '_save_audio', return_value="path/to/file.mp3") as mock_save_audio:      
        # Mock the length of the audio
        mock_audio_segment.__len__.return_value = mock_audio_length

        # Call the method under test
        result = audio_service.extract_audio(mock_url, mock_start_time_ms,)

        # Assert the fetch_audio method was called with the correct arguments
        audio_service.audio_data_layer.fetch_audio.assert_called_once_with(mock_url)

        # Assert the _save_audio method was called with the correct arguments
        mock_save_audio.assert_called_once_with(mock_audio_segment[mock_start_time_ms:mock_audio_length], None)

        # Assert the result
        assert result == {
            "audio_path": "path/to/file.mp3",
            "start_time_ms": mock_start_time_ms,
            "end_time_ms": mock_audio_length
        }

def test_extract_audio_error_fetching(audio_service,mock_audio_segment):
    """
    Test extracting audio when an error occurs while fetching the audio.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = 1000
    mock_end_time_ms = 4000
    mock_error = {"error": "An error occurred while fetching the audio"}

    # Mock the fetch_audio method to return an error
    audio_service.audio_data_layer.fetch_audio = mock.MagicMock(return_value=mock_error)

    # Call the extract_audio method
    result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

    # Assert the result
    assert 'error' in result
    assert result['error'] == f'An error occurred while fetching the audio: {mock_error["error"]}'

def test_extract_audio_exception_handling(audio_service):
    """
    Test extracting audio when an exception occurs.
    """
    mock_url = "https://valid-url.com/audio.mp3"
    mock_start_time_ms = 1000
    mock_end_time_ms = 4000

    # Mock the fetch_audio method to raise an exception
    audio_service.audio_data_layer.fetch_audio = mock.MagicMock(side_effect=Exception("Test exception"))

    # Call the extract_audio method
    result = audio_service.extract_audio(mock_url, mock_start_time_ms, mock_end_time_ms)

    # Assert the result
    assert 'error' in result
    assert result['error'] == 'An error occurred during the audio extraction: Test exception'
