import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

with open("config.yaml") as f:
    config = yaml.safe_load(f)

whisper_size = config["transcription"]["whisper"]["model_size"]
pipeline(
    "automatic-speech-recognition",
    model=f"openai/whisper-{whisper_size}",
)

model_name = config["sentiment_analysis"]["bertweet"]["model_name"]
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)