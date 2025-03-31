from flask import Flask,request
from flask_cors import CORS
from flask_restx import Api 

from app.utils.logger import requests_logger,logger

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

    @app.before_request
    def before_request():
        # Log incoming requests
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            body = request.get_json()
        else:
            body = None  # GET and other requests typically don't have a body
        
        requests_logger.info(f"Incoming Request: {request.method} {request.url} | Headers: {dict(request.headers)} | Body: {body}")


    @app.after_request
    def after_request(response):
        # Skip logging if the response is related to Swagger's JSON
        if '/swagger.json' in request.url:
            return response
        
        # Log the outgoing response
        requests_logger.info(f"Outgoing Response: {response.status_code} | Data: {response.get_json()}")

        # # Flush the log queue
        # requests_logger.handlers[0].flush()

        # # Flush the log queue
        # logger.handlers[0].flush()
        
        return response



    # Register routes
    register_routes(api)
    

    # print("Registered Routes:")
    # for rule in app.url_map.iter_rules():
    #     print(rule)

    return app