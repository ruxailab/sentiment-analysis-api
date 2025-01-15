import pytest
from app import create_app  # Assuming your create_app function is in the app module
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
    Create and configure a Flask app instance for tests.
    """
    # Create the app using the loaded config
    flask_app_config = config.config.get('flask')
    app = create_app(flask_app_config)

    # Optionally update the config for testing (like setting TESTING, DEBUG flags)
    app.config.update({
        'TESTING': True,
        'DEBUG': True
    })
    
    yield app  # Return the app for the test to use

    # Clean up or teardown if necessary after each test

@pytest.fixture
def client(app):
    """
    Provides a test client for the Flask app.
    """
    return app.test_client()