"""
Unit tests for the sentiment normalization module.
"""

import json
import pytest

from sentiment_normalization.schema import StandardizedOutput, ResultEntry, Segment
from sentiment_normalization.normalizer import (
    normalize_text_sentiment,
    normalize_pipeline_sentiment,
    normalize_facial_emotions,
    BERTWEET_LABEL_MAP,
    FACIAL_EMOTION_LABEL_MAP,
)


class TestSchema:
    """Tests for the Pydantic schema models."""

    def test_result_entry_minimal(self):
        entry = ResultEntry(label="positive", score=0.95)
        assert entry.label == "positive"
        assert entry.score == 0.95
        assert entry.intensity is None
        assert entry.segment is None

    def test_result_entry_with_segment(self):
        seg = Segment(start=0.0, end=3.5, text="hello")
        entry = ResultEntry(label="negative", score=0.8, segment=seg)
        assert entry.segment.start == 0.0
        assert entry.segment.text == "hello"

    def test_result_entry_with_intensity(self):
        entry = ResultEntry(label="angry", score=0.6, intensity="high")
        assert entry.intensity == "high"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            ResultEntry(label="positive", score=1.5)
        with pytest.raises(Exception):
            ResultEntry(label="positive", score=-0.1)

    def test_standardized_output_serialization(self):
        output = StandardizedOutput(
            analysis_type="sentiment",
            modality="text",
            source_model="test-model",
            timestamp="2026-03-09T12:00:00Z",
            input_summary="test input",
            results=[ResultEntry(label="positive", score=0.9)],
        )
        data = output.model_dump()
        assert data["schema_version"] == "1.0"
        assert data["analysis_type"] == "sentiment"
        assert len(data["results"]) == 1

    def test_standardized_output_to_json(self):
        output = StandardizedOutput(
            analysis_type="emotion",
            modality="facial",
            source_model="test-model",
            timestamp="2026-03-09T12:00:00Z",
            input_summary="video.mp4",
            results=[
                ResultEntry(label="happy", score=0.5),
                ResultEntry(label="neutral", score=0.3),
            ],
        )
        json_str = output.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["modality"] == "facial"
        assert len(parsed["results"]) == 2


class TestNormalizeTextSentiment:
    """Tests for normalize_text_sentiment."""

    def test_positive_label(self):
        result = normalize_text_sentiment(
            label="POS",
            confidence=0.95,
            text="I love this product!",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.analysis_type == "sentiment"
        assert result.modality == "text"
        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.95

    def test_negative_label(self):
        result = normalize_text_sentiment(
            label="NEG",
            confidence=0.87,
            text="This is terrible.",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.results[0].label == "negative"

    def test_neutral_label(self):
        result = normalize_text_sentiment(
            label="NEU",
            confidence=0.62,
            text="It is okay.",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.results[0].label == "neutral"

    def test_unknown_label_lowercased(self):
        result = normalize_text_sentiment(
            label="MIXED",
            confidence=0.5,
            text="Not sure about this.",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.results[0].label == "mixed"

    def test_long_text_truncated(self):
        long_text = "a" * 300
        result = normalize_text_sentiment(
            label="POS",
            confidence=0.8,
            text=long_text,
            timestamp="2026-03-09T12:00:00Z",
        )
        assert len(result.input_summary) == 200

    def test_task_id_included(self):
        result = normalize_text_sentiment(
            label="POS",
            confidence=0.9,
            text="Great!",
            task_id="task-7",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.task_id == "task-7"

    def test_auto_timestamp(self):
        result = normalize_text_sentiment(
            label="POS", confidence=0.9, text="Nice"
        )
        assert result.timestamp is not None
        assert "T" in result.timestamp


class TestNormalizePipelineSentiment:
    """Tests for normalize_pipeline_sentiment."""

    def setup_method(self):
        self.utterances = [
            {
                "timestamp": [0.0, 3.5],
                "text": "I liked the interface",
                "label": "POS",
                "confidence": 0.92,
            },
            {
                "timestamp": [3.5, 7.0],
                "text": "but search was confusing",
                "label": "NEG",
                "confidence": 0.78,
            },
        ]

    def test_basic_conversion(self):
        result = normalize_pipeline_sentiment(
            utterances=self.utterances,
            audio_path="audio.mp3",
            start_time_ms=0,
            end_time_ms=10000,
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.modality == "audio"
        assert len(result.results) == 2
        assert result.results[0].label == "positive"
        assert result.results[0].segment.start == 0.0
        assert result.results[0].segment.text == "I liked the interface"
        assert result.results[1].label == "negative"

    def test_skips_error_utterances(self):
        utterances = self.utterances + [{"error": "something failed"}]
        result = normalize_pipeline_sentiment(
            utterances=utterances,
            audio_path="audio.mp3",
            start_time_ms=0,
            end_time_ms=10000,
            timestamp="2026-03-09T12:00:00Z",
        )
        assert len(result.results) == 2

    def test_input_summary_format(self):
        result = normalize_pipeline_sentiment(
            utterances=self.utterances,
            audio_path="segment.mp3",
            start_time_ms=5000,
            end_time_ms=15000,
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.input_summary == "segment.mp3 (5000ms to 15000ms)"

    def test_empty_utterances(self):
        result = normalize_pipeline_sentiment(
            utterances=[],
            audio_path="empty.mp3",
            start_time_ms=0,
            end_time_ms=0,
            timestamp="2026-03-09T12:00:00Z",
        )
        assert len(result.results) == 0


class TestNormalizeFacialEmotions:
    """Tests for normalize_facial_emotions."""

    def setup_method(self):
        self.emotions = {
            "Angry": 4.0,
            "Disgusted": 1.0,
            "Fearful": 2.0,
            "Happy": 45.0,
            "Neutral": 30.0,
            "Sad": 6.0,
            "Surprised": 12.0,
        }

    def test_basic_conversion(self):
        result = normalize_facial_emotions(
            emotion_percentages=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.analysis_type == "emotion"
        assert result.modality == "facial"
        assert len(result.results) == 7

    def test_labels_lowercased(self):
        result = normalize_facial_emotions(
            emotion_percentages=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels = [r.label for r in result.results]
        for label in labels:
            assert label == label.lower()

    def test_scores_normalized_to_0_1(self):
        result = normalize_facial_emotions(
            emotion_percentages=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        for r in result.results:
            assert 0.0 <= r.score <= 1.0

    def test_sorted_by_score_descending(self):
        result = normalize_facial_emotions(
            emotion_percentages=self.emotions,
            video_path="recording.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_already_in_0_1_range(self):
        emotions_01 = {
            "Happy": 0.45,
            "Neutral": 0.30,
            "Surprised": 0.12,
            "Sad": 0.06,
            "Angry": 0.04,
            "Fearful": 0.02,
            "Disgusted": 0.01,
        }
        result = normalize_facial_emotions(
            emotion_percentages=emotions_01,
            video_path="video.mp4",
            timestamp="2026-03-09T12:00:00Z",
        )
        labels_scores = {r.label: r.score for r in result.results}
        assert labels_scores["happy"] == 0.45
        assert labels_scores["disgusted"] == 0.01

    def test_task_id_passed(self):
        result = normalize_facial_emotions(
            emotion_percentages=self.emotions,
            video_path="recording.mp4",
            task_id="session-99",
            timestamp="2026-03-09T12:00:00Z",
        )
        assert result.task_id == "session-99"


class TestLabelMaps:
    """Verify the label mapping dictionaries are complete."""

    def test_bertweet_map_has_all_labels(self):
        expected = {"POS", "NEG", "NEU"}
        assert set(BERTWEET_LABEL_MAP.keys()) == expected

    def test_facial_map_has_all_labels(self):
        expected = {"Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"}
        assert set(FACIAL_EMOTION_LABEL_MAP.keys()) == expected
