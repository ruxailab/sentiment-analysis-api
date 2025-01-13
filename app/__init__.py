from flask import Flask
from flask_cors import CORS
from flask_restx import Api 

from app.routes import register_routes

def create_app(config:dict):
    # Init Flask APP
    app = Flask(__name__)

    # Enable CORS with detailed settings
    CORS(app, resources={r"/*": {
        "origins": "*",  # Only allow specific origin
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allow these HTTP methods
        "allow_headers": ["Content-Type", "Authorization"],  # Allow these headers
        "supports_credentials": True  # If you need cookies/auth headers
    }})


    # Initialize Flask-RESTx Api
    api = Api(
        app,
        version="1.0",
        title=config['app_name'],
        description=f'API documentation for the {config["app_name"]} app',
        doc="/docs"  # Swagger UI documentation will be available at /docs
    )

    # Register routes
    register_routes(api)
    

    # print("Registered Routes:")
    # for rule in app.url_map.iter_rules():
    #     print(rule)

    return app