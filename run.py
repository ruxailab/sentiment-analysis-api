from app import create_app


import yaml

def load_config(config_file: str):
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration file '{config_file}': {e}")
        return {}


if __name__ == '__main__':
    try:
        print("Hello from the Sentiment Analysis Back End Server!")

        # Load configuration from config.yaml
        config = load_config("config.yaml")

        # Create Flask APP
        app = create_app(config)

        # Run the APP
        app.run(host=config['host'], port=config['port'], debug=config['debug'])

    except Exception as e:
        print(f"Unexpected Error in the Back End Server: {e}")

        