"""
Normalization utilities for converting raw model outputs into the
standardized sentiment/emotion schema.

Each normalize_* function takes the native output format of a specific
model or pipeline and returns a StandardizedOutput instance.
"""

from datetime import datetime, timezone
from typing import Optional

from normalization.schema import (
    StandardizedOutput,
    ResultEntry,
    Segment,
)

# --------------------------------------------------------------------------- #
# Label mappings
# --------------------------------------------------------------------------- #

# BERTweet sentiment labels -> standardized lowercase labels
BERTWEET_LABEL_MAP = {
    "POS": "positive",
    "NEG": "negative",
    "NEU": "neutral",
}

# Facial emotion labels (as returned by the facial-sentiment-analysis-api)
FACIAL_EMOTION_LABEL_MAP = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Fearful": "fearful",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprised",
}


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Text sentiment (single prediction)
# --------------------------------------------------------------------------- #

def normalize_text_sentiment(
    label: str,
    confidence: float,
    text: str,
    source_model: str = "bertweet-base-sentiment-analysis",
    task_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> StandardizedOutput:
    """Convert the output of the /sentiment/analyze endpoint.

    Parameters
    ----------
    label : str
        Raw label from the model (e.g. "POS", "NEG", "NEU").
    confidence : float
        Confidence score returned by the model.
    text : str
        The input text that was analyzed.
    source_model : str
        Identifier for the model that produced the result.
    task_id : str or None
        Optional task or session identifier.
    timestamp : str or None
        ISO 8601 timestamp. If not provided the current UTC time is used.
    """
    normalized_label = BERTWEET_LABEL_MAP.get(label, label.lower())

    return StandardizedOutput(
        analysis_type="sentiment",
        modality="text",
        source_model=source_model,
        timestamp=timestamp or _now_iso(),
        task_id=task_id,
        input_summary=text if len(text) <= 200 else text[:197] + "...",
        results=[
            ResultEntry(
                label=normalized_label,
                score=round(confidence, 4),
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Audio transcript + sentiment pipeline
# --------------------------------------------------------------------------- #

def normalize_pipeline_sentiment(
    utterances: list,
    audio_path: str,
    start_time_ms: int,
    end_time_ms: int,
    source_model: str = "bertweet-base-sentiment-analysis",
    task_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> StandardizedOutput:
    """Convert the output of the /audio-transcript-sentiment/process endpoint.

    Parameters
    ----------
    utterances : list[dict]
        List of utterance dicts, each with keys: timestamp (list of two
        floats), text (str), label (str), confidence (float). Utterances
        that have an "error" key instead of label/confidence are skipped.
    audio_path : str
        Path to the audio file that was analyzed.
    start_time_ms : int
        Start time of the extracted segment in milliseconds.
    end_time_ms : int
        End time of the extracted segment in milliseconds.
    source_model : str
        Identifier for the model that produced the result.
    task_id : str or None
        Optional task or session identifier.
    timestamp : str or None
        ISO 8601 timestamp.
    """
    results = []
    for u in utterances:
        if "error" in u:
            continue
        raw_label = u.get("label", "")
        normalized_label = BERTWEET_LABEL_MAP.get(raw_label, raw_label.lower())
        ts = u.get("timestamp", [0.0, 0.0])
        results.append(
            ResultEntry(
                label=normalized_label,
                score=round(u.get("confidence", 0.0), 4),
                segment=Segment(
                    start=float(ts[0]),
                    end=float(ts[1]),
                    text=u.get("text"),
                ),
            )
        )

    summary = f"{audio_path} ({start_time_ms}ms to {end_time_ms}ms)"

    return StandardizedOutput(
        analysis_type="sentiment",
        modality="audio",
        source_model=source_model,
        timestamp=timestamp or _now_iso(),
        task_id=task_id,
        input_summary=summary,
        results=results,
    )


# --------------------------------------------------------------------------- #
# Facial emotion percentages
# --------------------------------------------------------------------------- #

def normalize_facial_emotions(
    emotion_percentages: dict,
    video_path: str,
    source_model: str = "emotiondetector-model1",
    task_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> StandardizedOutput:
    """Convert the output of the facial-sentiment-analysis-api.

    Parameters
    ----------
    emotion_percentages : dict
        Dictionary mapping emotion names to float percentages (0-100 or 0-1).
        Keys should be title-case names like "Happy", "Angry", etc.
    video_path : str
        Path or identifier of the video that was analyzed.
    source_model : str
        Identifier for the model that produced the result.
    task_id : str or None
        Optional task or session identifier.
    timestamp : str or None
        ISO 8601 timestamp.
    """
    results = []
    for raw_label, score in emotion_percentages.items():
        normalized_label = FACIAL_EMOTION_LABEL_MAP.get(raw_label, raw_label.lower())
        # If scores are in 0-100 range, convert to 0-1
        if score > 1.0:
            score = score / 100.0
        results.append(
            ResultEntry(
                label=normalized_label,
                score=round(score, 4),
            )
        )

    # Sort by score descending so the dominant emotion comes first
    results.sort(key=lambda r: r.score, reverse=True)

    return StandardizedOutput(
        analysis_type="emotion",
        modality="facial",
        source_model=source_model,
        timestamp=timestamp or _now_iso(),
        task_id=task_id,
        input_summary=video_path,
        results=results,
    )
