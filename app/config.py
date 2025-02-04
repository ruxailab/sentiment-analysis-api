import yaml
import os

class Config:
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls, config_path=None):
        """Create and return a singleton instance."""
        if cls._instance is None:
            # If no instance exists, create a new one and store it in _instance
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.config_path = config_path or 'config.yaml'
            cls._instance.config = cls._instance._load_config(cls._instance.config_path)
        return cls._instance

    def _load_config(self, config_path):
        """Load configuration from the YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The configuration file {config_path} was not found.")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """Retrieve a configuration value by key."""
        return self.config.get(key, default)