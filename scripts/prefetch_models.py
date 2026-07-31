import yaml
from faster_whisper import WhisperModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

with open("config.yaml") as f:
    config = yaml.safe_load(f)

whisper = config["transcription"]["whisper"]
WhisperModel(
    whisper.get("model_size", "base"),
    device=whisper.get("device", "cpu"),
    compute_type=whisper.get("compute_type", "int8"),
)

model_name = config["sentiment_analysis"]["bertweet"]["model_name"]
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)