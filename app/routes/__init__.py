from flask_restx import Api

# Routes
from app.routes.ping_routes import api as ping_api
from app.routes.audio_routes import api as audio_api
from app.routes.transcript_routes import api as transcript_api
from app.routes.sentiment_routes import api as sentiment_api
from app.routes.audio_transcript_sentiment_routes import api as audio_transcription_sentiment_api

def register_routes(api: Api):
    # Initialize the AudioService with the configuration

    # Register the routes
    api.add_namespace(ping_api, path='/ping')
    api.add_namespace(audio_api, path='/audio')
    api.add_namespace(transcript_api, path='/transcription')
    api.add_namespace(sentiment_api, path='/sentiment')

    # Pipelines
    api.add_namespace(audio_transcription_sentiment_api, path='/audio-transcript-sentiment')
    return 