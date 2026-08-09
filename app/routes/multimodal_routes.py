"""
Routes for the multimodal sentiment analysis endpoint.

Exposes POST /multimodal/analyze — accepts any combination of text,
pre-computed facial emotions, and audio path, and returns a fused
sentiment prediction from the MultimodalSentimentEngine.

Thread safety note:
    The engine is a module-level singleton. Per-request weight overrides
    are passed as arguments to engine.analyze() and are never written back
    to engine.weights. Concurrent requests cannot overwrite each other's
    weight configurations under gunicorn or any threaded WSGI server.
"""

from flask_restx import Namespace, Resource, fields
from flask import request
from app.services.multimodal_engine import MultimodalSentimentEngine
from app.utils.logger import logger

engine = MultimodalSentimentEngine()


def register_routes(api):

    multimodal_request_model = api.model('MultimodalRequest', {
        'text': fields.String(
            description='Transcribed text from the audio segment.',
            example='This interface is really confusing.'
        ),
        'facial_emotions': fields.Raw(
            description=(
                'Dict of {emotion_label: percentage} from the '
                'facial-sentiment-analysis-api /process_video endpoint. '
                'Example: {"Happy": 12.5, "Neutral": 61.0, "Sad": 26.5}'
            )
        ),
        'audio_path': fields.String(
            description='Server-side path to the audio file for prosody extraction.',
            example='static/audio/session_01_task_03.mp3'
        ),
        'start_ms': fields.Integer(
            default=0,
            description='Segment start time in milliseconds.'
        ),
        'end_ms': fields.Integer(
            description='Segment end time in milliseconds.'
        ),
        'weights': fields.Raw(
            description=(
                'Optional per-request modality weight override. '
                'Example: {"text": 0.6, "facial": 0.3, "prosody": 0.1}'
            )
        )
    })

    multimodal_success_model = api.model('MultimodalSuccess', {
        'status': fields.String(example='success'),
        'data': fields.Raw(
            description=(
                'Fused prediction with per-modality breakdown. '
                'All labels normalized to: positive | neutral | negative.'
            ),
            example={
                'fused_label': 'negative',
                'fused_confidence': 0.6823,
                'modality_scores': {
                    'text': {
                        'label': 'negative',
                        'confidence': 0.87,
                        'weight': 0.45
                    },
                    'facial': {
                        'label': 'neutral',
                        'confidence': 0.61,
                        'weight': 0.35
                    },
                    'prosody': {
                        'label': 'negative',
                        'confidence': 0.42,
                        'weight': 0.20
                    }
                }
            }
        )
    })

    multimodal_bad_request_model = api.model('MultimodalBadRequest', {
        'status': fields.String(example='error'),
        'error':  fields.String(
            example='At least one modality must be provided.'
        ),
        'data':   fields.Raw(example=None)
    })

    multimodal_server_error_model = api.model('MultimodalServerError', {
        'status': fields.String(example='error'),
        'error':  fields.String(
            example='An unexpected error occurred while processing the request.'
        ),
        'data':   fields.Raw(example=None)
    })

    @api.route('/analyze')
    class MultimodalAnalyze(Resource):

        @api.doc(
            description=(
                'Fuse facial expression, transcribed text sentiment, and '
                'voice prosody into a single sentiment prediction. '
                'At least one modality must be provided. Missing modalities '
                'are skipped and their weight redistributed to available ones. '
                'All output labels are normalized to: positive | neutral | negative.'
            )
        )
        @api.expect(multimodal_request_model)
        @api.response(200, 'Success', multimodal_success_model)
        @api.response(400, 'Bad Request', multimodal_bad_request_model)
        @api.response(500, 'Internal Server Error', multimodal_server_error_model)
        def post(self):
            """
            Multimodal sentiment fusion from facial, text, and prosody signals.
            """
            try:
                data = request.json

                if not data:
                    return {
                        'status': 'error',
                        'error':  'Request body must be valid JSON.',
                        'data':   None
                    }, 400

                text            = data.get('text')
                facial_emotions = data.get('facial_emotions')
                audio_path      = data.get('audio_path')

                has_text    = bool(text and text.strip())
                has_facial  = bool(facial_emotions)
                has_prosody = bool(
                    audio_path and data.get('end_ms') is not None
                )

                if not any([has_text, has_facial, has_prosody]):
                    return {
                        'status': 'error',
                        'error': (
                            'At least one modality must be provided: '
                            'text, facial_emotions, or audio_path + end_ms.'
                        ),
                        'data': None
                    }, 400

                # Build per-request weights without mutating the singleton.
                # Each request gets its own copy of the defaults, then
                # applies its own overrides on top — thread safe.
                request_weights = dict(engine.weights)
                if data.get('weights'):
                    override = data['weights']
                    if not isinstance(override, dict):
                        return {
                            'status': 'error',
                            'error':  'weights must be a JSON object.',
                            'data':   None
                        }, 400
                    request_weights.update(override)

                result = engine.analyze(
                    text=text if has_text else None,
                    facial_emotions=facial_emotions if has_facial else None,
                    audio_path=audio_path if has_prosody else None,
                    start_ms=data.get('start_ms', 0),
                    end_ms=data.get('end_ms'),
                    weights=request_weights
                )

                if 'error' in result:
                    return {
                        'status': 'error',
                        'error':  result['error'],
                        'data':   None
                    }, 500

                return {
                    'status': 'success',
                    'data':   result
                }, 200

            except Exception as e:
                logger.error(
                    "[MultimodalAnalyze] Unexpected error: %s", str(e)
                )
                return {
                    'status': 'error',
                    'error':  'An unexpected error occurred while processing the request.',
                    'data':   None
                }, 500


api = Namespace('Multimodal', description='Multimodal Sentiment Operations')
register_routes(api)
