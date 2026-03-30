"""
Standardized output schema for sentiment and emotion analysis results.

Defines Pydantic models that represent the v1 schema documented in
docs/emotion_output_schema_v1.md. These models can be used to validate,
serialize, and deserialize analysis results across different modalities
(text, audio, facial) and analysis types (sentiment, emotion).
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Segment(BaseModel):
    """Temporal and textual segment associated with a result entry."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: Optional[str] = Field(None, description="Transcript text for this segment")


class ResultEntry(BaseModel):
    """A single analysis result (one label with its score)."""
    label: str = Field(..., description="Predicted label in lowercase")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence or proportion score")
    intensity: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Optional intensity qualifier"
    )
    segment: Optional[Segment] = Field(
        None, description="Temporal/textual segment info, null if result covers the full input"
    )


class StandardizedOutput(BaseModel):
    """Top-level standardized output for any sentiment or emotion analysis."""
    schema_version: str = Field("1.0", description="Schema format version")
    analysis_type: Literal["sentiment", "emotion"] = Field(
        ..., description="Type of analysis performed"
    )
    modality: Literal["text", "facial", "audio"] = Field(
        ..., description="Input modality that was analyzed"
    )
    source_model: str = Field(
        ..., description="Identifier of the model or service that produced the result"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 datetime of when the analysis was performed"
    )
    task_id: Optional[str] = Field(
        None, description="Optional identifier for a usability test task or session"
    )
    input_summary: str = Field(
        ..., description="Short description of the analyzed input"
    )
    results: List[ResultEntry] = Field(
        ..., description="List of individual result entries"
    )
