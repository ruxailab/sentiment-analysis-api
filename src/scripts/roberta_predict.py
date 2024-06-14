import json 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

sentiment_labels = ["negative", "neutral", "positive"]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")


def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
        
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted sentiment
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[0][predicted_class].item()

with open("./data/demos/sportify/sportify.mp4_utterances_timestamps_transcript.json", "r") as file:
    utterances = json.load(file)

    transcript_sentiment=[] 

    for utterance in utterances:
        print(utterance)

        text = utterance["text"]

        sentiment, confidence = predict_sentiment(text)

        transcript_sentiment.append({
            'start_time': utterance["start_time"],
            'end_time': utterance["end_time"],
            'speaker': utterance["speaker"],
            'text': text,
            'sentiment': sentiment_labels[sentiment],
            'confidence': confidence
        })


        save_path='./sentiment.json'
        with open(save_path, 'w') as f:
            json.dump(transcript_sentiment, f, indent=4)
            print(f"Sentiment saved to {save_path}")


# PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict


# Examples
# text = "I love using BERT models!"
# text = "I like spotify! I can't wait to listen to my favorite song"
# text = "I am not able to use this product. It is useless"
# text = "I think it should be free"
# sentiment, confidence = predict_sentiment(text)


# print(f"Sentiment: {sentiment_labels[sentiment]}")
# print(f"Confidence: {confidence:.2f}")