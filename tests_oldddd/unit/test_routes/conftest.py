"""
Shared configurations and fixtures for unit route tests.
"""

import pytest
from unittest.mock import MagicMock

from app import create_app
from app.config import Config



@pytest.fixture(scope="session")
def config():
    """
    Load the configuration using the singleton instance.
    This fixture ensures the config is loaded once during the test session.
    """
    config_path = "config.yaml"
    config = Config(config_path=config_path)  # Singleton instance will load config once
    return config

@pytest.fixture
def app(config):
    """
    Create and configure a Flask app instance for unit tests.
    """
    # Create the app using the loaded config
    flask_app_config = config.config.get('flask')
    app = create_app(flask_app_config)

    # Optionally update the config for testing (like setting TESTING, DEBUG flags)
    app.config.update({
        'TESTING': True,
        'DEBUG': False
    })
    
    return app

@pytest.fixture
def client(app):
    """
    Create a test client for the Flask app.
    """
    return app.test_client()


@pytest.fixture
def mock_ping_service(monkeypatch):
    """
    Mock the PingService for unit tests.
    """
    mock_service = MagicMock()
    # Mock the ping method to return a predefined response
    mock_service.ping.return_value = {
        "message": "Mocked Pong!"
    }

    monkeypatch.setattr("app.routes.ping_routes.service", mock_service)
    return mock_service

@pytest.fixture
def mock_audio_service(monkeypatch):
    """
    Mock the AudioService for unit tests.
    """
    mock_service = MagicMock()

    # Mock the extract_audio method to return a predefined response
    mock_service.extract_audio.return_value = {
        "audio_path": "mocked_audio_path",
        "start_time_ms": "mocked_start_time_ms",
        "end_time_ms": "mocked_end_time_ms"
    }

    monkeypatch.setattr("app.routes.audio_routes.service", mock_service)
    return mock_service