import pytest
from unittest.mock import patch, MagicMock
from unittest import mock

import requests
from pydub import AudioSegment

# Import the class to be tested
from app.data.audio_data import AudioDataLayer  # Import your class

# Test setup
@pytest.fixture
def audio_layer():
    """
    Fixture for initializing the AudioDataLayer.
    """
    config = {
        'debug': True
    }
    return AudioDataLayer(config=config)


########################################################## From Local File ##########################################################
# Test for fetching audio from a local file path
def test_fetch_audio_from_local_file(audio_layer):
    """
    GIVEN a valid local file path
    WHEN the fetch_audio method is called
    THEN it should return an AudioSegment object.
    """
    mock_local_path = "mock_audio.mp3"
    mock_audio_content = b"fake_audio_data"

    # Mock os.path.isfile to return True, indicating it's a local file
    with mock.patch('os.path.isfile', return_value=True) as mock_isfile:
        # Mock AudioSegment.from_file to return a mock AudioSegment object
        with mock.patch.object(AudioSegment, 'from_file', return_value = mock_audio_content) as mock_from_file:
        
            # Call the fetch_audio method
            result = audio_layer.fetch_audio(mock_local_path)

            # Assert that the result is the mock audio object
            assert result == mock_audio_content
            
            # Ensure that os.path.isfile was called with the correct path
            mock_isfile.assert_called_with(mock_local_path)

            # Ensure that AudioSegment.from_file was called with the correct path
            mock_from_file.assert_called_with(mock_local_path)

# Test for failed fetching audio from a local file path does not exist
def test_fetch_audio_from_nonexistent_local_file(audio_layer):
    """
    GIVEN a non-existent local file path
    WHEN the fetch_audio method is called
    THEN it should return an error message indicating the file does not exist.
    """
    mock_local_path = "non_existent_audio.mp3"

    # Mock os.path.isfile to return False, indicating the file does not exist
    with mock.patch('os.path.isfile', return_value=False) as mock_isfile:
        # Call the method
        result = audio_layer.fetch_audio(mock_local_path)
        
        # Assert that the result contains an error message
        assert 'error' in result

########################################################## From URL ##########################################################
def test_fetch_audio_from_url(audio_layer):
    """
    GIVEN a valid audio URL
    WHEN the fetch_audio method is called
    THEN it should download the file and return an AudioSegment object.
    """
    mock_url = "http://example.com/audio.mp3"
    mock_audio_content = b"fake_audio_data"

    # Patch requests.get to simulate a successful HTTP response
    with mock.patch("requests.get") as mock_get, mock.patch.object(AudioSegment, 'from_file',return_value = mock_audio_content) as mock_from_file:
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = mock_audio_content
        mock_get.return_value = mock_response

        # Call the method
        result = audio_layer.fetch_audio(mock_url)

        # Assert that requests.get was called with the correct URL
        mock_get.assert_called_with(mock_url)

        # Alternatively, you can assert based on the content inside the BytesIO object:
        # Assert that the content of the BytesIO object passed to AudioSegment.from_file is the same as mock_audio_content
        mock_from_file.assert_called_once()
        called_stream = mock_from_file.call_args[0][0]  # Get the first argument passed to from_file
        assert called_stream.getvalue() == mock_audio_content

        # Assert that the result is the mocked audio object
        assert result == mock_audio_content


# Test for failed download due to invalid URL (HTTP 404)
def test_fetch_audio_from_invalid_url(audio_layer):
    """
    GIVEN an invalid audio URL (404 error)
    WHEN the fetch_audio method is called
    THEN it should return an error message indicating failure to download.
    """
    mock_url = "http://example.com/invalid_audio.mp3"

    # Patch requests.get to simulate a failed HTTP response
    with patch("requests.get") as mock_get:
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Call the method
        result = audio_layer.fetch_audio(mock_url)

        # Assert that the result is an error message
        assert 'error' in result
        assert result['error'] == 'Failed to download audio file, HTTP status: 404'

# Test for handling requests.exceptions.RequestException
def test_fetch_audio_from_request_exception(audio_layer):
    """
    GIVEN a network-related error (requests.exceptions.RequestException)
    WHEN the fetch_audio method is called
    THEN it should return an error message indicating the HTTP request failure.
    """
    mock_url = "http://example.com/audio.mp3"


    # Patch requests.get to raise a RequestException
    with patch("requests.get") as mock_get:
        # Simulate a RequestException when making the HTTP request
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        # Call the method
        result = audio_layer.fetch_audio(mock_url)

        # Assert that the result contains an error message
        assert 'error' in result
        assert result['error'] == 'Error occurred during the HTTP request: Network error'


# ########################################################## Exception ##########################################################
# Test for handling unexpected exceptions (os.path.isfile() throws an exception)
def test_fetch_audio_unexpected_exception(audio_layer):
    """
    GIVEN an unexpected exception (os.path.isfile() raises an exception)
    WHEN the fetch_audio method is called
    THEN it should return an error message indicating the unexpected error.
    """
    mock_local_path = "mock_audio.mp3"

    # Patch os.path.isfile to raise an exception
    with patch('os.path.isfile', side_effect=Exception("Unexpected error")) as mock_isfile:
        # Call the method
        result = audio_layer.fetch_audio(mock_local_path)

        # Assert that the result contains an error message
        assert 'error' in result
        assert result['error'] == 'An unexpected error occurred: Unexpected error'