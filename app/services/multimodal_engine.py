"""
MultimodalSentimentEngine — fuses facial, text, and voice prosody signals
into a single confidence-weighted sentiment prediction.

Design decisions:
- All three modalities map to a common 3-class label space
  ('positive', 'neutral', 'negative') before fusion.
- Missing modalities are skipped and their weight is redistributed
  proportionally to the available modalities.
- Per-request weight overrides are passed as arguments, not mutations
  of the shared singleton, to avoid race conditions under concurrent load.
"""

from app.services.sentiment_service import SentimentService
from app.services.prosody_service import ProsodyService
from app.utils.logger import logger


DEFAULT_WEIGHTS = {
    'text':    0.45,
    'facial':  0.35,
    'prosody': 0.20
}

LABEL_MAP = {
    'POS': 'positive',
    'NEU': 'neutral',
    'NEG': 'negative',
    'Happy':     'positive',
    'Surprised': 'positive',
    'Neutral':   'neutral',
    'Sad':       'negative',
    'Angry':     'negative',
    'Fearful':   'negative',
    'Disgusted': 'negative',
    'positive': 'positive',
    'neutral':  'neutral',
    'negative': 'negative',
}


class MultimodalSentimentEngine:
    """
    Fuses facial expression, transcribed text sentiment, and voice prosody
    into a single sentiment prediction per audio-visual chunk.

    The engine is designed as a module-level singleton in the route layer.
    Per-request weight overrides are passed as method arguments — not
    mutations of self.weights — so that concurrent requests cannot
    overwrite each other's weight configurations.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.sentiment_service = SentimentService()
        self.prosody_service = ProsodyService()

    def analyze(
        self,
        text: str = None,
        facial_emotions: dict = None,
        audio_path: str = None,
        start_ms: int = 0,
        end_ms: int = None,
        weights: dict = None,
    ) -> dict:
        """
        Fuse available modalities into a single sentiment prediction.

        At least one of text, facial_emotions, or audio_path must be
        provided. Missing modalities are skipped and their weight is
        redistributed proportionally to the available ones.

        Args:
            text           : transcribed text for the segment
            facial_emotions: dict of {emotion_label: percentage} from
                             the facial-sentiment-analysis-api
            audio_path     : path to the audio file for prosody extraction
            start_ms       : segment start time in milliseconds
            end_ms         : segment end time in milliseconds
            weights        : per-request weight override dict
                             (does NOT mutate the engine's default weights)

        Returns:
            dict with fused_label, fused_confidence, modality_scores
            or {'error': str} if no modality produced a valid result.
        """
        active_weights = weights if weights is not None else self.weights
        scores = {}

        # --- Text modality ---
        if text and text.strip():
            try:
                result = self.sentiment_service.analyze(text)
                if 'error' not in result:
                    normalized_label = LABEL_MAP.get(
                        result['label'], 'neutral'
                    )
                    scores['text'] = {
                        'label':      normalized_label,
                        'confidence': float(result['confidence']),
                        'weight':     active_weights.get('text', 0.45)
                    }
                    logger.debug(
                        "[MultimodalEngine] Text: %s (%.3f)",
                        normalized_label, result['confidence']
                    )
            except Exception as e:
                logger.error(
                    "[MultimodalEngine] Text modality failed: %s", str(e)
                )

        # --- Facial modality ---
        # Integration boundary: the facial API processes video files and
        # returns frame-level emotion percentages via POST /process_video.
        # The engine receives these as a pre-aggregated dict, keeping the
        # fusion layer decoupled from the streaming architecture of the
        # facial API. Temporal alignment (Deliverable G9) will handle the
        # case where frame-level outputs need to be matched to audio chunk
        # timestamps before aggregation.
        if facial_emotions:
            try:
                if not isinstance(facial_emotions, dict):
                    raise ValueError(
                        "facial_emotions must be a dict of {label: percentage}"
                    )
                dominant = max(facial_emotions, key=facial_emotions.get)
                confidence = facial_emotions[dominant] / 100.0
                normalized_label = LABEL_MAP.get(dominant, 'neutral')
                scores['facial'] = {
                    'label':      normalized_label,
                    'confidence': round(float(confidence), 4),
                    'weight':     active_weights.get('facial', 0.35)
                }
                logger.debug(
                    "[MultimodalEngine] Facial: %s (%.3f)",
                    normalized_label, confidence
                )
            except Exception as e:
                logger.error(
                    "[MultimodalEngine] Facial modality failed: %s", str(e)
                )

        # --- Prosody modality ---
        if audio_path and end_ms is not None:
            try:
                prosody = self.prosody_service.extract(
                    audio_path, start_ms, end_ms
                )
                if prosody:
                    scores['prosody'] = {
                        'label':      prosody['valence_estimate'],
                        'confidence': prosody['valence_score'],
                        'weight':     active_weights.get('prosody', 0.20)
                    }
                    logger.debug(
                        "[MultimodalEngine] Prosody: %s (%.3f)",
                        prosody['valence_estimate'],
                        prosody['valence_score']
                    )
            except Exception as e:
                logger.error(
                    "[MultimodalEngine] Prosody modality failed: %s", str(e)
                )

        if not scores:
            logger.error(
                "[MultimodalEngine] No modality produced a valid result."
            )
            return {'error': 'No modality produced a valid result.'}

        return self._fuse(scores)

    def _fuse(self, scores: dict) -> dict:
        """
        Confidence-weighted vote across available modalities.

        Missing modalities are skipped and their weight is redistributed
        proportionally so that total weight always sums to 1.0.
        """
        total_raw_weight = sum(v['weight'] for v in scores.values())

        label_buckets = {
            'positive': 0.0,
            'neutral':  0.0,
            'negative': 0.0
        }

        for modality, data in scores.items():
            normalized_w = data['weight'] / total_raw_weight
            label = data['label']
            label_buckets[label] += normalized_w * data['confidence']

        fused_label = max(label_buckets, key=label_buckets.get)
        fused_confidence = round(label_buckets[fused_label], 4)

        logger.debug(
            "[MultimodalEngine] Fused: %s (%.4f) from %d modalities",
            fused_label, fused_confidence, len(scores)
        )

        return {
            'fused_label':      fused_label,
            'fused_confidence': fused_confidence,
            'modality_scores':  scores
        }
