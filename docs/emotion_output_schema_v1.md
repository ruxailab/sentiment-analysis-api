# Standardized Emotion and Sentiment Output Schema (v1)

## Problem

The RUXAILAB platform currently consumes sentiment and emotion analysis results from multiple services, each returning data in a different format. The text-based sentiment API returns results with labels like `POS`, `NEG`, `NEU` alongside a confidence score, while the facial emotion API returns a flat object with emotion names mapped to percentage values (`Angry`, `Happy`, `Neutral`, etc.). There is no shared structure between these outputs.

This makes it harder to build generalized components on the frontend, to store results consistently in Firestore, and to compare or combine results across modalities. Any new analysis service (e.g. speech prosody, physiological signals) would add yet another format to handle.

## Motivation

Standardizing the output format across all sentiment and emotion analysis services would:

- Allow the frontend to render results from any modality using a shared set of components
- Simplify data storage by using a consistent document structure in Firestore
- Make it easier to aggregate or compare sentiment across modalities for the same usability test session
- Reduce integration effort when adding new analysis backends

This schema is intentionally minimal. It captures the fields that are common to all current analysis types while leaving room for future extension.

## Proposed Schema

```json
{
  "schema_version": "1.0",
  "analysis_type": "sentiment | emotion",
  "modality": "text | facial | audio",
  "source_model": "string",
  "timestamp": "ISO 8601 datetime string",
  "task_id": "string | null",
  "input_summary": "string",
  "results": [
    {
      "label": "string (lowercase)",
      "score": "float (0.0 to 1.0)",
      "intensity": "string | null",
      "segment": {
        "start": "float (seconds)",
        "end": "float (seconds)",
        "text": "string | null"
      }
    }
  ]
}
```

## Field Descriptions

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Version of this schema format. Currently `"1.0"`. |
| `analysis_type` | string | yes | Either `"sentiment"` or `"emotion"`. Sentiment covers polarity (positive/negative/neutral). Emotion covers categorical states (happy, sad, angry, etc.). |
| `modality` | string | yes | The input modality that was analyzed. One of `"text"`, `"facial"`, or `"audio"`. |
| `source_model` | string | yes | Identifier for the model or service that produced the result. For example `"bertweet-base-sentiment-analysis"` or `"emotiondetector-model1"`. |
| `timestamp` | string | yes | ISO 8601 formatted datetime of when the analysis was performed. |
| `task_id` | string or null | no | Optional identifier linking this result to a specific usability test task or session. |
| `input_summary` | string | yes | A short description of what was analyzed. For text input this would be the text itself (possibly truncated). For audio or video it could be the filename or a description. |
| `results` | array | yes | List of individual result entries. For simple text sentiment this will have one entry. For timestamped audio or multi-emotion facial analysis this will have multiple entries. |

### Result entry fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | string | yes | The predicted label in lowercase. For sentiment: `"positive"`, `"negative"`, `"neutral"`. For emotion: `"happy"`, `"sad"`, `"angry"`, `"disgusted"`, `"fearful"`, `"surprised"`, `"neutral"`. |
| `score` | float | yes | Confidence or proportion score between 0.0 and 1.0. |
| `intensity` | string or null | no | Optional intensity qualifier. Could be `"low"`, `"medium"`, `"high"` or null if not applicable. Reserved for future use. |
| `segment` | object or null | no | Temporal or textual segment information. Null for results that apply to the entire input. |

### Segment fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start` | float | yes | Start time in seconds from the beginning of the input. |
| `end` | float | yes | End time in seconds. |
| `text` | string or null | no | Transcript text for this segment, if available. |

## Label Normalization

A key part of standardization is normalizing labels from different models to a consistent vocabulary. The current mappings are:

**Sentiment labels (BERTweet)**
- `POS` -> `positive`
- `NEG` -> `negative`
- `NEU` -> `neutral`

**Emotion labels (facial analysis)**
- `Angry` -> `angry`
- `Disgusted` -> `disgusted`
- `Fearful` -> `fearful`
- `Happy` -> `happy`
- `Neutral` -> `neutral`
- `Sad` -> `sad`
- `Surprised` -> `surprised`

New models that use different label sets should add their own mapping in the normalizer module.

## Example Outputs

### Text sentiment analysis

```json
{
  "schema_version": "1.0",
  "analysis_type": "sentiment",
  "modality": "text",
  "source_model": "bertweet-base-sentiment-analysis",
  "timestamp": "2026-03-09T14:22:00Z",
  "task_id": null,
  "input_summary": "I love this product!",
  "results": [
    {
      "label": "positive",
      "score": 0.95,
      "intensity": null,
      "segment": null
    }
  ]
}
```

### Audio transcript sentiment (pipeline output)

```json
{
  "schema_version": "1.0",
  "analysis_type": "sentiment",
  "modality": "audio",
  "source_model": "bertweet-base-sentiment-analysis",
  "timestamp": "2026-03-09T14:25:00Z",
  "task_id": "usability-session-42",
  "input_summary": "sample_audio.mp3 (0ms to 10000ms)",
  "results": [
    {
      "label": "positive",
      "score": 0.92,
      "intensity": null,
      "segment": {
        "start": 0.0,
        "end": 3.5,
        "text": "I really liked the navigation"
      }
    },
    {
      "label": "negative",
      "score": 0.78,
      "intensity": null,
      "segment": {
        "start": 3.5,
        "end": 7.2,
        "text": "but the search was confusing"
      }
    }
  ]
}
```

### Facial emotion analysis

```json
{
  "schema_version": "1.0",
  "analysis_type": "emotion",
  "modality": "facial",
  "source_model": "emotiondetector-model1",
  "timestamp": "2026-03-09T14:30:00Z",
  "task_id": "usability-session-42",
  "input_summary": "recording_task3.mp4",
  "results": [
    {"label": "happy", "score": 0.45, "intensity": null, "segment": null},
    {"label": "neutral", "score": 0.30, "intensity": null, "segment": null},
    {"label": "surprised", "score": 0.12, "intensity": null, "segment": null},
    {"label": "sad", "score": 0.06, "intensity": null, "segment": null},
    {"label": "angry", "score": 0.04, "intensity": null, "segment": null},
    {"label": "fearful", "score": 0.02, "intensity": null, "segment": null},
    {"label": "disgusted", "score": 0.01, "intensity": null, "segment": null}
  ]
}
```

## Compatibility Considerations

### Existing API responses

This schema is designed to coexist with the current API response format. The existing `/sentiment/analyze` and `/audio-transcript-sentiment/process` endpoints will continue to return their current format. The standardized format can be introduced as an optional wrapper or as an additional field in future versions of the API response.

### RUXAILAB frontend

The frontend currently expects sentiment results with `label` (POS/NEG/NEU) and `confidence` fields for audio sentiment, and emotion percentage objects for facial analysis. Adopting this schema would require updating the `AudioSentimentController` and `FacialSentimentPanel` components to read from the new structure. This can be done incrementally since the new format is a superset of the information in the current format.

### Firestore storage

The standardized schema maps directly to a Firestore document structure. The `results` array is compatible with Firestore's array type. The `task_id` field enables querying results by task or session.

### Facial sentiment API

The facial-sentiment-analysis-api currently returns emotion percentages as a flat dictionary. The normalizer module includes a converter that transforms this format into the standardized schema. No changes to the facial API itself are required.

## Future Extensions

- Additional modalities (e.g. `physiological`, `speech_prosody`)
- Per-segment emotion-sentiment fusion results
- Severity or intensity classification once models support it
- Batch analysis results with multiple inputs
