"""
Unit tests for the Ping route module.
"""

# Global variable for the endpoint path
ENDPOINT = '/ping/'

def test_ping_route(client, mock_ping_service):
    """
    GIVEN the ping route
    WHEN a GET request is made
    THEN check the response status code and message
    """
    # Make a GET request to the ping route
    response = client.get(ENDPOINT)

    # Check the response
    assert response.status_code == 200
    assert response.json == {
        "status": "success",
        "data": {
            "message": "Mocked Pong!"
        }
    }

    # Verify that the ping method of the service was called
    mock_ping_service.ping.assert_called_once()