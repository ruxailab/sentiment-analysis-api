"""
Unit tests for the Audio route module.
"""

# Global variable for the endpoint path
ENDPOINT = '/audio/extract'

def test_audio_extract_route_bad_request_no_url(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with a missing URL
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route
    response = client.post(ENDPOINT, json={})

    # Check the response
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "error": "URL is required"
    }

    # Verify that the extract_audio method of the service was not called
    mock_audio_service.extract_audio.assert_not_called()

def test_audio_extract_route_bad_request_negative_start_time(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with a negative start time
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with a negative start time
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": -100
    })

    # Check the response
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "error": "'start_time_ms' cannot be negative."
    }

    # Verify that the extract_audio method of the service was not called
    mock_audio_service.extract_audio.assert_not_called()

def test_audio_extract_route_bad_request_negative_end_time(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with a negative end time
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with a negative end time
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 0,
        "end_time_ms": -100
    })

    # Check the response
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "error": "'end_time_ms' cannot be negative."
    }

    # Verify that the extract_audio method of the service was not called
    mock_audio_service.extract_audio.assert_not_called()

def test_audio_extract_route_bad_request_end_time_less_than_start_time(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with an end time less than the start time
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with end time less than start time
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 100,
        "end_time_ms": 50
    })

    # Check the response
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "error": "'end_time_ms' must be greater than or equal to 'start_time_ms'."
    }

    # Verify that the extract_audio method of the service was not called
    mock_audio_service.extract_audio.assert_not_called()

def test_audio_extract_route_success(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with valid parameters
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with valid parameters
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 0,
        "end_time_ms": 1000
    })

    # Check the response
    assert response.status_code == 200
    assert response.json == {
        "status": "success",
        "audio_path": "mocked_audio_path",
        "start_time_ms": "mocked_start_time_ms",
        "end_time_ms": "mocked_end_time_ms"
    }

    # Verify that the extract_audio method of the service was called with the correct parameters
    mock_audio_service.extract_audio.assert_called_once_with("test.mp3", 0, 1000, None)

def test_audio_extract_route_internal_server_error(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made and the service layer returns an error
    THEN check the response status code and message
    """
    # Mock the extract_audio method to return an error
    mock_audio_service.extract_audio.return_value = {
        "error": "An error occurred while processing the audio."
    }

    # Make a POST request to the extract audio route
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 0,
        "end_time_ms": 1000
    })

    # Check the response
    assert response.status_code == 500
    assert response.json == {
        "status": "error",
        "error": "An error occurred while processing the audio."
    }

    # Verify that the extract_audio method of the service was called with the correct parameters
    mock_audio_service.extract_audio.assert_called_once_with("test.mp3", 0, 1000, None)

def test_audio_extract_route_unexpected_error(client, mock_audio_service):
    """
    GIVEN the extract audio route
    WHEN a POST request is made and an unexpected error occurs
    THEN check the response status code and message
    """
    # Mock the extract_audio method to raise an exception
    mock_audio_service.extract_audio.side_effect = Exception("Unexpected error")

    # Make a POST request to the extract audio route
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 0,
        "end_time_ms": 1000
    })

    # Check the response
    assert response.status_code == 500
    assert response.json == {
        "status": "error",
        "error": "An unexpected error occurred while processing the audio."
    }

    # Verify that the extract_audio method of the service was called with the correct parameters
    mock_audio_service.extract_audio.assert_called_once_with("test.mp3", 0, 1000, None)
