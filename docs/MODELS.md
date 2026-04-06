# Models Documentation

This document provides detailed information about the machine learning models used in the Sentiment Analysis API.

## Table of Contents
- [Whisper (Transcription)](#whisper-transcription)
- [BERTweet (Sentiment Analysis)](#bertweet-sentiment-analysis)
- [Model Comparison](#model-comparison)
- [Performance Metrics](#performance-metrics)
- [Switching Models](#switching-models)
- [Model Requirements](#model-requirements)

---

## Whisper (Transcription)

### Overview

**Whisper** is OpenAI's open-source speech recognition model trained on 680,000 hours of multilingual audio data from the web.

**Model Name:** `openai/whisper`

### Model Sizes

Whisper comes in 5 different sizes with varying accuracy and speed trade-offs:

| Size | Parameters | English-only | Multilingual | Relative Speed | Relative VRAM |
|------|-----------|-------------|--------------|----------------|---------------|
| tiny | 39M | ✓ | ✓ | 1x | 1x (~1GB) |
| base | 74M | ✓ | ✓ | 2x | 1x (~1.5GB) |
| small | 244M | ✓ | ✓ | 3x | 2x (~3GB) |
| medium | 769M | ✓ | ✓ | 6x | 5x (~5GB) |
| large | 1.5B | ✓ | ✓ | 12x | 10x (~10GB) |

### Features

✓ **Multilingual Support:** 99 languages supported
✓ **Robustness:** Handles various audio conditions (background noise, accents)
✓ **Open Source:** Publicly available and free to use
✓ **Real-time Capable:** Can process audio in chunks
✓ **Automatic Language Detection:** Identifies language from audio

### Supported Languages

**Full Language List:**
Afrikaans, Arabic, Armenian, Assamese, Azerbaijani, Bashkir, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Catalan, Cebuano, Chinese (Mandarin), Chinese (Cantonese), Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Javanese, Kannada, Karachay-Balkar, Kazakh, Khmer, Korean, Lao, Latin, Latvian, Lingala, Lithuanian, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Marathi, Minangkabau, Moldavian, Mongolian, Myanmar, Nepali, Norwegian, Occitan, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Samoan, Sanskrit, Serbian, Shona, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish, Turkmen, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, Xhosa, Yiddish, Yoruba, Yue Chinese, Zhuang, Zulu

### Audio Format Support

**Supported Formats:**
- MP3
- WAV
- M4A
- FLAC
- OGG
- WEBM

**Audio Requirements:**
- Sampling Rate: 16kHz or higher (automatically resampled if needed)
- Channels: Mono or Stereo
- Duration: 5 seconds to 30+ minutes

### Configuration

```yaml
transcription:
  default_model: "whisper"
  whisper:
    model_size: "base"          # tiny, base, small, medium, large
    device: 'cpu'               # 'cpu' or 'cuda'
    chunk_length_s: 30          # Process audio in 30-second chunks
```

### Performance Metrics

**Word Error Rate (WER) - English:**
| Model | WER |
|-------|-----|
| tiny | 12.10% |
| base | 8.40% |
| small | 6.10% |
| medium | 5.40% |
| large | 4.95% |

Lower WER is better (more accurate).

**Processing Speed (CPU - Intel i5):**
| Model | ~1 min audio |
|-------|-------------|
| tiny | 10 seconds |
| base | 20 seconds |
| small | 60 seconds |
| medium | 120+ seconds |
| large | 240+ seconds |

**Processing Speed (GPU - NVIDIA RTX 3080):**
| Model | ~1 min audio |
|-------|-------------|
| tiny | 2 seconds |
| base | 3 seconds |
| small | 5 seconds |
| medium | 10 seconds |
| large | 20 seconds |

### Recommendation

- **Fast Processing:** Use `tiny` or `base` (good for real-time applications)
- **Balanced:** Use `small` (recommended for most use cases)
- **High Accuracy:** Use `medium` or `large` (for critical transcriptions)

---

## BERTweet (Sentiment Analysis)

### Overview

**BERTweet** is a RoBERTa-based transformer model trained on Twitter data for sentiment analysis. It's a specialized version of BERT optimized for social media text.

**Model Name:** `finiteautomata/bertweet-base-sentiment-analysis`

### Architecture

- **Base Model:** RoBERTa
- **Training Data:** 2.7 million tweets from Twitter
- **Vocabulary:** Optimized for social media language and abbreviations
- **Input:** Text strings up to 512 tokens
- **Output:** Sentiment probabilities for 3 classes

### Sentiment Labels

BERTweet classifies text into three sentiment categories:

| Label | Description | Example |
|-------|-------------|---------|
| `POS` | Positive | "I love this!" |
| `NEU` | Neutral | "The weather is cloudy." |
| `NEG` | Negative | "This is terrible." |

### Features

✓ **Fast Classification:** Real-time sentiment analysis
✓ **Robust:** Handles emoji, slang, abbreviations
✓ **Lightweight:** ~500MB model size
✓ **Universal:** Works with diverse text types
✓ **Confidence Scores:** Returns probability for each prediction

### Configuration

```yaml
sentiment_analysis:
  default_model: "bertweet"
  bertweet:
    model_name: "finiteautomata/bertweet-base-sentiment-analysis"
    device: 'cpu'               # 'cpu' or 'cuda'
```

### Performance Metrics

**Accuracy on Benchmark Datasets:**
| Dataset | Accuracy |
|---------|----------|
| Twitter (SST) | 90.1% |
| SemEval 2017 Task 4A | 89.5% |
| Stanford Sentiment Treebank | 88.3% |

**Inference Speed (CPU):**
- Single text: ~10-50ms
- Batch of 100: ~500-1000ms

**Inference Speed (GPU):**
- Single text: ~2-5ms
- Batch of 100: ~50-100ms

### Limitations

⚠️ **English Only:** Currently supports English text only
⚠️ **Twitter-Optimized:** Best performance on informal, social media text
⚠️ **Context Limited:** 512 token limit (text length)
⚠️ **Emoji Dependent:** Emoji handling is optimized for specific sets

### Input Constraints

- **Maximum Length:** 512 tokens (~2000 characters for English)
- **Minimum Length:** 1 token (even single words work)
- **Language:** English only
- **Special Characters:** Supports emoji and punctuation

### Text Preprocessing

BERTweet expects:
- Original text (no need for cleaning in most cases)
- Handles: URL, @mentions, hashtags naturally
- Preserves emoji as features

**Example Preprocessing:**
```python
text = "I love this product! 🎉 #amazing"
# BERTweet works with text as-is - no cleaning needed
```

---

## Model Comparison

### Feature Comparison

| Feature | Whisper | BERTweet |
|---------|---------|----------|
| Task | Speech-to-Text | Text Classification |
| Input | Audio | Text |
| Output | Text Transcription | Sentiment (POS/NEG/NEU) |
| Languages | 99 | English only |
| Model Size | 39M - 1.5B parameters | ~110M parameters |
| Speed | Fast to Very Slow | Very Fast |
| Accuracy | ~95% (WER depends on model) | ~90% accuracy |
| GPU Requirements | Recommended | Optional |
| Memory | 1GB - 10GB | ~500MB |

### Processing Pipeline

```
Video/Audio File
        ↓
   [Whisper]
        ↓
   Transcription
        ↓
  [BERTweet]
        ↓
 Sentiment + Confidence
        ↓
   Final Output
```

---

## Performance Metrics

### Accuracy Metrics

**Whisper WER (Word Error Rate):**
- Lower is better
- Varies by language and audio quality
- English: 4-12% depending on model size

**BERTweet Accuracy:**
- Top 1 accuracy: ~90%
- Macro-averaged F1: ~89%
- Per-class performance:
  - POS: 91%
  - NEG: 89%
  - NEU: 87%

### Latency Analysis

**Single Request Latency:**
```
Whisper (base, 1 min audio, CPU): ~20 seconds
BERTweet (sentiment analysis): ~5ms per utterance
Total: ~20 seconds (dominated by Whisper)

Whisper (base, 1 min audio, GPU): ~3 seconds
BERTweet (sentiment analysis): ~1ms per utterance
Total: ~3 seconds
```

### Throughput

**CPU Processing:**
- 1 file (5 min audio): ~5-10 minutes
- Sequential: ~30 files/hour

**GPU Processing:**
- 1 file (5 min audio): ~1-2 minutes
- Parallel: 100+ files/hour

---

## Switching Models

### Changing Whisper Size

1. **Edit config.yaml:**
```yaml
transcription:
  whisper:
    model_size: "small"  # Change from 'base'
```

2. **Restart the application**

3. **First run will download the model** (~1.4GB for 'small')

```bash
python run.py
# First run downloads model from: https://openaipublic.blob.core.windows.net/main/whisper/models/
```

### Adding Alternative Models

**Example: Adding Faster Whisper (optimized):**

1. Install faster-whisper:
```bash
pip install faster-whisper
```

2. Create new model file (app/models/faster_whisper_model.py)

3. Update config.yaml:
```yaml
transcription:
  default_model: "faster_whisper"
  faster_whisper:
    model_size: "base"
    device: 'cpu'
```

4. Update service to use new model

### Adding Alternative Sentiment Models

**Example: Adding VADER sentiment analyzer:**

1. Install VADER:
```bash
pip install vaderSentiment
```

2. Create new model file (app/models/vader_model.py)

3. Update config.yaml:
```yaml
sentiment_analysis:
  default_model: "vader"
  vader:
    # VADER-specific configuration
```

---

## Model Requirements

### System Requirements

**Minimum (CPU-only):**
- RAM: 4GB
- Storage: 20GB
- CPU: 2 cores minimum, 4+ recommended
- OS: Linux, Windows, macOS

**Recommended (With GPU):**
- RAM: 8GB+
- VRAM: 4GB+ (NVIDIA GPU)
- Storage: 30GB
- CPU: 4+ cores
- GPU: NVIDIA RTX 3080 or equivalent

### Storage Requirements

**Model Downloads:**
- Whisper tiny: 39MB
- Whisper base: 74MB
- Whisper small: 244MB
- Whisper medium: 769MB
- Whisper large: 1.5GB
- BERTweet base: ~500MB

**Total with all models:** ~4GB

**Generated Audio Files:**
- Depends on audio duration
- 1 hour audio: ~100-300MB (depending on compression)

### Installation Requirements

```bash
# Minimum Python version
python >= 3.11

# Core dependencies
torch >= 2.6.0
transformers >= 4.48.2
openai-whisper >= 20230314
```

---

## Troubleshooting Models

### Model Not Downloading

```bash
# Check internet connection first
ping hf-mirror.com

# Manually download Whisper model
python -c "import whisper; whisper.load_model('base')"

# Manually download BERTweet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Poor Transcription Quality

- Use larger Whisper model (small, medium, large)
- Ensure audio quality (reduce background noise)
- Use correct language if non-English
- Check audio format is supported

### Incorrect Sentiment Predictions

- Verify input text is English
- Check for very short inputs (< 3 words)
- Test with benchmark examples
- Review confidence scores (low confidence = uncertain)

### Memory Issues

- Reduce Whisper model size
- Use GPU acceleration
- Process files in smaller segments
- Monitor system resources

---

## Further Reading

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [BERTweet GitHub](https://github.com/VinAIResearch/BERTweet)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

## References

- OpenAI Whisper: https://github.com/openai/whisper
- BERTweet: https://github.com/VinAIResearch/BERTweet
- Hugging Face Models: https://huggingface.co/
