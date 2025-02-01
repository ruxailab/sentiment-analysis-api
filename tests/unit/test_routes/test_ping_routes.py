"""
This Module contains the unit tests for the ping routes.
"""

import pytest
from unittest.mock import patch

class TestPing:
    """Test suite for the Ping route."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Set up before each test."""
        self.client = client
        self.endpoint = "/ping/"

    @pytest.fixture
    def mock_ping(self):
        """Fixture to mock the ping method."""
        with patch('app.routes.ping_routes.PingService.ping') as mock_ping:
            yield mock_ping

    def test_ping_success(self, mock_ping):
        """Test a successful ping."""
        mock_ping.return_value = {
            "message": "Pong!"
        }
        response = self.client.get(self.endpoint)

        assert response.status_code == 200
        assert response.json == {
            "status": "success",
            "data": {
                "message": "Pong!"
            }
        }
        mock_ping.assert_called_once()

    def test_ping_exception(self, mock_ping):
        """Test when the service layer raises an exception."""
        # Mock the service method to raise an exception
        mock_ping.side_effect = Exception("Mocked exception")
        response = self.client.get(self.endpoint)

        assert response.status_code == 500
        assert response.json == {
            "status": "error",
            "error": "An unexpected error occurred while processing the request.",
            "data": None
        }
        mock_ping.assert_called_once()

# # Run:
# coverage run  -m pytest .\tests\unit\test_routes\test_ping_routes.py
