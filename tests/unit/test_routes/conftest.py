"""
Shared configurations and fixtures for unit route tests.
"""

import pytest
from unittest.mock import MagicMock

import yaml
from app import create_app



@pytest.fixture(scope="session")
def config():
    """
    Load the configuration from the config.yaml file.
    """
    with open("config.yaml", "r") as config_file:
        return yaml.safe_load(config_file)

@pytest.fixture
def app(config):
    """
    Create and configure a Flask app instance for unit tests.
    """
    # Create the app using the loaded config
    app = create_app(config)

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
    mock_service.ping.return_value = {"message": "Mocked Pong!"}
    monkeypatch.setattr("app.routes.ping_routes.service", mock_service)
    return mock_service