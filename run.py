from app import create_app

from app.config import Config  # Import Config class


if __name__ == '__main__':
    try:
        print("Hello from the Sentiment Analysis Back End Server :D")

        # Create the Config instance (Singleton will ensure it's only loaded once)
        config = Config(config_path='config.yaml')

        # Create Flask APP
        flask_app_config = config.config.get('flask')
        app = create_app(flask_app_config)

        # Run the APP
        app.run(host=flask_app_config['host'], port=flask_app_config['port'], debug=flask_app_config['debug'])

    except Exception as e:
        print(f"Unexpected Error in the Back End Server: {e}")

        