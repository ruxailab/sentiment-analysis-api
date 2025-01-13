from flask_restx import Api

# Routes
from app.routes.ping_routes import api as ping_api

def register_routes(api: Api):
    api.add_namespace(ping_api, path='/ping')

    return 