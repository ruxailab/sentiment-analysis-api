from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

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



# # Read Transcript from file
# # with open("./transcript.txt", "r") as file:
# with open("./transcript.txt", "r") as file:
#     transcript = file.read()
# # Split the text into sentences by ., !, or ?
# sentences = re.split(r'(?<=[.!?]) +', transcript)
# # Print each sentence
# for sentence in sentences:

#     sentiment, confidence = predict_sentiment(sentence)
#     print(sentence + "---->" + f"Sentiment: {sentiment_labels[sentiment]}")
# # text = "I love using BERT models!"


# Read Transcript from file
# with open("./transcript.txt", "r") as file:
with open("./utterances_timestamps_hr_transcript.txt", "r") as file:
    transcript = file.read()
    # Split by new line
    utterances = transcript.split("\n")

# Print each sentence
for utterance in utterances:

    # Regular expression to match the timestamp and speaker label pattern
    match = re.match(r'\[.*\] .*: (.+)', utterance)
    if match:
        text = match.group(1)
    else:
        print("No match", utterance)
        continue
        
    sentiment, confidence = predict_sentiment(text)
    print(utterance + "---->" + f"Sentiment: {sentiment_labels[sentiment]}")


# text = "I love using BERT models!"



# text = "I like spotify! I can't wait to listen to my favorite song"
# text = "I am not able to use this product. It is useless"
# text = "I think it should be free"
# sentiment, confidence = predict_sentiment(text)


# print(f"Sentiment: {sentiment_labels[sentiment]}")
# print(f"Confidence: {confidence:.2f}")