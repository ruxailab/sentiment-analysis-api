"""
Integration test for the /audio route
"""

# Global variable for the endpoint path
ENDPOINT = '/audio/extract'

def test_audio_extract_route_bad_request_no_url(client):
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

def test_audio_extract_route_bad_request_negative_start_time(client):
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

def test_audio_extract_route_bad_request_negative_end_time(client):
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

def test_audio_extract_route_bad_request_end_time_less_than_start_time(client):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with an end time less than the start time
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with an end time less than the start time
    response = client.post(ENDPOINT, json={
        "url": "test.mp3",
        "start_time_ms": 1000,
        "end_time_ms": 500
    })

    # Check the response
    assert response.status_code == 400
    assert response.json == {
        "status": "error",
        "error": "'end_time_ms' must be greater than or equal to 'start_time_ms'."
    }

def test_audio_extract_route_bad_request_invalid_url(client):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with an invalid URL
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with an invalid URL
    response = client.post(ENDPOINT, json={
        "url": "invalid_url.mp3"
    })

    # Check the response
    assert response.status_code == 500
    assert response.json ["status"] == "error"
    assert response.json ["error"] == "An error occurred while processing the audio."


def test_audio_extract_route_success_local(client):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with valid data
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with valid data
    response = client.post(ENDPOINT, json={
        "url": "./data/video.mp4",
        "start_time_ms": 0,
        "end_time_ms": 1000
    })

    # Check the response
    assert response.status_code == 200
    assert response.json ["status"] == "success"
    assert "audio_path" in response.json
    assert response.json ["start_time_ms"] == 0
    assert response.json ["end_time_ms"] == 1000

def test_audio_extract_route_success_url(client):
    """
    GIVEN the extract audio route
    WHEN a POST request is made with valid data
    THEN check the response status code and message
    """
    # Make a POST request to the extract audio route with valid data
    response = client.post(ENDPOINT, json={
        "url": "https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v&export=download",
        "start_time_ms": 0,
        "end_time_ms": 1000
    })

    # Check the response
    assert response.status_code == 200
    assert response.json ["status"] == "success"
    assert "audio_path" in response.json
    assert response.json ["start_time_ms"] == 0
    assert response.json ["end_time_ms"] == 1000
