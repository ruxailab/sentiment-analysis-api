# API Documentation

## Overview

The Sentiment Analysis API provides endpoints for:
- **Audio Extraction** - Extract audio segments from video/audio files
- **Transcription** - Convert audio to text using Whisper
- **Sentiment Analysis** - Analyze sentiment of text using BERTweet
- **Complete Pipeline** - Process audio/video in one request and get transcription + sentiment

## Base URL

```
http://localhost:8001
```

## Response Format

All responses follow a consistent JSON structure:

### Success Response
```json
{
  "status": "success",
  "data": {
    // Endpoint-specific data
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Error message describing what went wrong",
  "data": null
}
```

## HTTP Status Codes

- **200 OK** - Request successful
- **400 Bad Request** - Missing or invalid request parameters
- **500 Internal Server Error** - Server-side error during processing

---

## Endpoints

### 1. Health Check - Ping Server

**Endpoint:** `GET /ping/`

**Description:** Check if the server is running and responding.

**Parameters:** None

**cURL Example:**
```bash
curl -X GET http://localhost:8001/ping/
```

**Python Example:**
```python
import requests

response = requests.get('http://localhost:8001/ping/')
print(response.json())
```

**JavaScript Example:**
```javascript
fetch('http://localhost:8001/ping/')
  .then(response => response.json())
  .then(data => console.log(data));
```

**Success Response (200):**
```json
{
  "status": "success",
  "data": {
    "message": "Pong"
  }
}
```

---

### 2. Extract Audio Segment

**Endpoint:** `POST /audio/extract`

**Description:** Extract an audio segment from a video or audio file by start and end time.

**Request Body:**
```json
{
  "url": "path/to/file.mp4",
  "start_time_ms": 0,
  "end_time_ms": 5000,
  "user_id": "user123"
}
```

**Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| url | string | Yes | Path or URL of audio/video file | `/samples/video.mp4` |
| start_time_ms | number | Yes | Start time in milliseconds | 0 |
| end_time_ms | number | Yes | End time in milliseconds | 5000 |
| user_id | string | No | User ID for organizing files | `user123` |

**cURL Example:**
```bash
curl -X POST http://localhost:8001/audio/extract \
  -H "Content-Type: application/json" \
  -d '{
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 5000,
    "user_id": "user123"
  }'
```

**Python Example:**
```python
import requests
import json

payload = {
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 5000,
    "user_id": "user123"
}

response = requests.post('http://localhost:8001/audio/extract', json=payload)
print(response.json())
```

**JavaScript Example:**
```javascript
const payload = {
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 5000,
    "user_id": "user123"
};

fetch('http://localhost:8001/audio/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(data));
```

**Success Response (200):**
```json
{
  "status": "success",
  "data": {
    "audio_path": "path/to/extracted_audio.wav",
    "start_time_ms": 0,
    "end_time_ms": 5000
  }
}
```

**Error Response (400):**
```json
{
  "status": "error",
  "error": "url is required",
  "data": null
}
```

---

### 3. Transcribe Audio

**Endpoint:** `POST /transcription/transcribe`

**Description:** Convert audio file to text using Whisper model.

**Request Body:**
```json
{
  "file_path": "path/to/audio.mp3"
}
```

**Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| file_path | string | Yes | Path to audio file | `static/audio/user123/audio.mp3` |

**Supported Audio Formats:**
- MP3
- WAV
- M4A
- FLAC
- OGG

**cURL Example:**
```bash
curl -X POST http://localhost:8001/transcription/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_path": "static/audio/user123/audio.mp3"}'
```

**Python Example:**
```python
import requests

payload = {"file_path": "static/audio/user123/audio.mp3"}
response = requests.post('http://localhost:8001/transcription/transcribe', json=payload)
print(response.json())
```

**JavaScript Example:**
```javascript
fetch('http://localhost:8001/transcription/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({"file_path": "static/audio/user123/audio.mp3"})
})
.then(response => response.json())
.then(data => console.log(data));
```

**Success Response (200):**
```json
{
  "status": "success",
  "data": {
    "transcription": "Hello, this is a sample audio transcription."
  }
}
```

**Error Response (400):**
```json
{
  "status": "error",
  "error": "file_path is required",
  "data": null
}
```

---

### 4. Analyze Sentiment

**Endpoint:** `POST /sentiment/analyze`

**Description:** Analyze sentiment of a given text using BERTweet model.

**Request Body:**
```json
{
  "text": "I love this product!"
}
```

**Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| text | string | Yes | Text to analyze | `I love this product!` |

**Sentiment Labels:**
- `POS` - Positive sentiment
- `NEG` - Negative sentiment
- `NEU` - Neutral sentiment

**cURL Example:**
```bash
curl -X POST http://localhost:8001/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Python Example:**
```python
import requests

payload = {"text": "I love this product!"}
response = requests.post('http://localhost:8001/sentiment/analyze', json=payload)
print(response.json())
```

**JavaScript Example:**
```javascript
fetch('http://localhost:8001/sentiment/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({"text": "I love this product!"})
})
.then(response => response.json())
.then(data => console.log(data));
```

**Success Response (200):**
```json
{
  "status": "success",
  "data": {
    "label": "POS",
    "confidence": 0.98
  }
}
```

**Error Response (400):**
```json
{
  "status": "error",
  "error": "text is required",
  "data": null
}
```

---

### 5. Complete Pipeline - Audio Transcription & Sentiment Analysis

**Endpoint:** `POST /audio_transcript_sentiment/process`

**Description:** Process an audio/video file end-to-end: extract audio segment, transcribe it, and perform sentiment analysis on transcribed utterances.

**Request Body:**
```json
{
  "url": "path/to/file.mp4",
  "start_time_ms": 0,
  "end_time_ms": 10000
}
```

**Parameters:**
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| url | string | Yes | Path or URL of audio/video file | `/samples/video.mp4` |
| start_time_ms | number | Yes | Start time in milliseconds | 0 |
| end_time_ms | number | Yes | End time in milliseconds | 10000 |

**Processing Steps:**
1. Extract audio segment from video/audio file
2. Transcribe extracted audio to text
3. Analyze sentiment for each utterance/segment
4. Return combined results with timestamps and sentiment labels

**cURL Example:**
```bash
curl -X POST http://localhost:8001/audio_transcript_sentiment/process \
  -H "Content-Type: application/json" \
  -d '{
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 10000
  }'
```

**Python Example:**
```python
import requests
import json

payload = {
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 10000
}

response = requests.post(
    'http://localhost:8001/audio_transcript_sentiment/process',
    json=payload
)
print(json.dumps(response.json(), indent=2))
```

**JavaScript Example:**
```javascript
const payload = {
    "url": "/samples/video.mp4",
    "start_time_ms": 0,
    "end_time_ms": 10000
};

fetch('http://localhost:8001/audio_transcript_sentiment/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(JSON.stringify(data, null, 2)));
```

**Success Response (200):**
```json
{
  "status": "success",
  "data": {
    "audio_path": "static/audio/extracted_audio.wav",
    "start_time_ms": 0,
    "end_time_ms": 10000,
    "transcription": "Hello, I love this product. It is amazing!",
    "utterances_sentiment": [
      {
        "timestamp": [0, 3000],
        "text": "Hello, I love this product.",
        "label": "POS",
        "confidence": 0.97
      },
      {
        "timestamp": [3000, 6000],
        "text": "It is amazing!",
        "label": "POS",
        "confidence": 0.99
      }
    ]
  }
}
```

**Error Response (400):**
```json
{
  "status": "error",
  "error": "url is required",
  "data": null
}
```

---

## Error Handling

### Common Errors

| HTTP Code | Error Message | Cause | Solution |
|-----------|---------------|-------|----------|
| 400 | `url is required` | Missing required parameter | Add url parameter |
| 400 | `text is required` | Missing required parameter | Add text parameter |
| 400 | `file_path is required` | Missing required parameter | Add file_path parameter |
| 400 | `start_time_ms < 0` | Invalid time range | Ensure time is >= 0 |
| 400 | `start_time_ms > end_time_ms` | Invalid time range | Ensure start < end |
| 500 | `An unexpected error occurred during sentiment analysis` | Server error | Check server logs |
| 500 | `An unexpected error occurred during transcription` | Server error | Check file exists and format is supported |
| 500 | `An unexpected error occurred while processing the audio` | Server error | Check audio file is valid |

---

## Rate Limiting & Best Practices

1. **Batch Processing:** For multiple files, process sequentially to avoid memory issues
2. **File Size:** Keep audio files under 2GB for optimal performance
3. **Timeouts:** Set request timeout to at least 60 seconds for large files
4. **Error Handling:** Always check response status before processing data

### Recommended Request Structure

```python
import requests
import time

def process_audio_safely(url, start_time, end_time, max_retries=3):
    """Process audio with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:8001/audio_transcript_sentiment/process',
                json={
                    "url": url,
                    "start_time_ms": start_time,
                    "end_time_ms": end_time
                },
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code >= 500:
                # Retry on server error
                time.sleep(2 ** attempt)
                continue
            else:
                # Bad request - don't retry
                raise Exception(response.json()['error'])
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    raise Exception("Max retries exceeded")
```

---

## Integration Examples

### Example 1: Single File Processing

```python
import requests

def analyze_video(video_path):
    """Analyze entire video file"""
    response = requests.post(
        'http://localhost:8001/audio_transcript_sentiment/process',
        json={
            "url": video_path,
            "start_time_ms": 0,
            "end_time_ms": float('inf')  # Process entire file
        }
    )
    
    if response.status_code == 200:
        result = response.json()['data']
        print(f"Full transcription: {result['transcription']}")
        print(f"Sentiment segments: {result['utterances_sentiment']}")
    else:
        print(f"Error: {response.json()['error']}")

# Usage
analyze_video('/samples/interview.mp4')
```

### Example 2: Segment Processing

```python
import requests

def analyze_video_segments(video_path, segment_duration_ms=5000):
    """Analyze video in fixed segments"""
    total_duration = 60000  # 60 seconds
    
    for start_ms in range(0, total_duration, segment_duration_ms):
        end_ms = min(start_ms + segment_duration_ms, total_duration)
        
        response = requests.post(
            'http://localhost:8001/audio_transcript_sentiment/process',
            json={
                "url": video_path,
                "start_time_ms": start_ms,
                "end_time_ms": end_ms
            }
        )
        
        if response.status_code == 200:
            data = response.json()['data']
            print(f"Segment {start_ms}-{end_ms}ms: {data['utterances_sentiment']}")

# Usage
analyze_video_segments('/samples/meeting.mp4')
```

### Example 3: Batch Processing Multiple Files

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def process_files(file_list):
    """Process multiple files concurrently"""
    
    def process_file(file_path):
        try:
            response = requests.post(
                'http://localhost:8001/audio_transcript_sentiment/process',
                json={
                    "url": file_path,
                    "start_time_ms": 0,
                    "end_time_ms": 30000  # First 30 seconds
                },
                timeout=120
            )
            return {
                'file': file_path,
                'status': 'success' if response.status_code == 200 else 'failed',
                'data': response.json()
            }
        except Exception as e:
            return {'file': file_path, 'status': 'error', 'error': str(e)}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_file, file_list))
    
    return results

# Usage
files = ['/samples/video1.mp4', '/samples/video2.mp4', '/samples/audio.mp3']
results = process_files(files)
for result in results:
    print(f"{result['file']}: {result['status']}")
```

---

## Support

For issues or questions:
1. Check the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
2. Review [Configuration Guide](../docs/CONFIGURATION.md)
3. Check server logs: `logs/app.log`
4. Open an issue on GitHub
