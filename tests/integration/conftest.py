import pytest
import yaml
from app import create_app  # Assuming your create_app function is in the app module

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
    Create and configure a Flask app instance for tests.
    """
    # Create the app using the loaded config
    app = create_app(config)

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