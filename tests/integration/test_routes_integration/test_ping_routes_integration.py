"""
Integration test for the /ping route
"""

# Global variable for the endpoint path
ENDPOINT = '/ping/'

def test_ping_route(client):
    """
    GIVEN a Flask app instance
    WHEN the GET request is made to the /ping/ endpoint
    THEN check that the response is valid and the status code is 200
    """
    response = client.get(ENDPOINT)  # Ensure the path matches the registered route

    # Assert that the status code is 200
    assert response.status_code == 200

    # Assert that the returned data is in the expected format
    assert response.json['status'] == 'success'
    assert response.json['data']['message'] == 'Pong!'