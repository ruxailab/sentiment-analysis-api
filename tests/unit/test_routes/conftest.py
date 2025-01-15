import pytest
from app import create_app  # Assuming your create_app function is in the app module
from app.config import Config

@pytest.fixture(scope="session")
def app():
    """
    Creates a Flask app instance for testing.
    """
    try:
        print("Setting up Flask app for testing...")
        
        # Create the Config instance (Singleton will ensure it's only loaded once)
        config = Config(config_path='config.yaml')

        # Create Flask app using the config loaded
        flask_app_config = config.config.get('flask')
        app = create_app(flask_app_config)

        # Optionally update the config for testing
        app.config.update({
            'TESTING': True,  # Enable testing mode
            'DEBUG': True      # Enable debug mode for detailed error messages
        })

        yield app  # Yield the app to the test

    except Exception as e:
        print(f"Error setting up Flask app: {e}")
        pytest.fail(f"App setup failed: {e}")

    finally:
        print("Cleaning up Flask app after tests...")

@pytest.fixture
def client(app):
    """
    Provides a test client for the Flask app.
    """
    # Create a test client using the Flask app passed in the fixture
    test_client = app.test_client() # This creates a test client instance, which you can use to send HTTP requests to your Flask app.
    return test_client

# Why is the client fixture function scoped?
# - If you have multiple test functions that each need a clean test client instance, then you would use this scope.
# client.get('/'): This simulates an HTTP GET request to the route '/'. It behaves like an actual HTTP client accessing the app.