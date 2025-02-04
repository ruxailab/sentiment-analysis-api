"""
This Module contains the unit tests for the ping service.
"""

import pytest
from unittest.mock import patch

from app.services.ping_service import PingService


class TestPingService:
    """Test suite for the PingService class."""
    @pytest.fixture()
    def ping_service(self):
        """Fixture for creating an instance of the PingService class."""
        return PingService()
    
    # Grouped tests for the 'ping' method
    class TestPing:
        def setup_method(self):
            """Setup method for each test."""
            self.args = {
            }

        def test_ping(self, ping_service):
            """
            Test for the 'ping' method.
            """
            args = self.args.copy()

            result = ping_service.ping()

            assert result == {
                'message': 'Pong!'
            }

# # Run the tests
# coverage run  -m pytest .\tests\unit\test_services\test_ping_service.py