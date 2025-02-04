"""
This module contains the routes for the ping endpoint.
"""
from flask_restx import Namespace, Resource, fields

# Service
from app.services.ping_service import PingService

service = PingService()


def register_routes(api):
    # Define the Response model
    response_model = api.model('PingResponseModel', {
        'status': fields.String(required=True, description='The status of the response',example='success'),
        'data': fields.Nested(api.model('PingDataModel', {
            'message': fields.String(required=True, description='The message returned by the server',example='Pong')
        }))  # Embed the data model
    })

    # Define the Error model
    error_model = api.model('PingErrorModel', {
        'status': fields.String(required=True, description='The status of the response',example='error'),
        'error': fields.String(required=True, description='The error message',example='Internal Server Error'),
        'data': fields.String(description='The data returned by the server (if any)',example=None)
    })


    # Define the endpoint for the ping route
    @api.route('/')  # Ensure the route has a proper path
    class Ping(Resource):
        @api.doc(description="Ping the server to check if it is alive")
        @api.response(200, 'Success', response_model)
        @api.response(500, 'Internal Server Error', error_model)
        def get(self):
            """
            Endpoint to ping the server.
            """
            try:
                # Call the service
                result = service.ping()

                # Return the result
                return {
                    "status": "success",
                    "data": {
                        "message": result['message']
                    }
                }, 200

            except Exception as e:
                print(f"Route Error: Failed to ping the server: {str(e)}")
                return {
                    "status": "error",
                    "error": "An unexpected error occurred while processing the request.",
                    "data": None
                }, 500
            

# Define the namespace
api = Namespace('Ping', description="Ping operations")

register_routes(api)