"""
Unit tests for the ping service module :D
"""
from app.services.ping_service import PingService

def test_ping_service():
    """
    GIVEN a PingService object
    WHEN the ping method is called
    THEN check the return value
    """
    # Create a PingService object
    service = PingService()

    # Call the ping method
    result = service.ping()

    # Check the result
    assert result['message'] == "Pong!"