import json 
import argparse
import time
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

DEVICE = None

def predict_sentiment(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
        
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted sentiment
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[0][predicted_class].item()


def roberta_sentiment_analysis(utterances:list, save_path:str=None, per_sentence:bool=False):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis").to(DEVICE)

    sentiment_labels = ["negative", "neutral", "positive"]

    transcript_sentiment=[]

    start = time.time()
    for utterance in utterances:
        # print(utterance)

        text = utterance["text"]
        # print(text)

        if per_sentence:
            sentences = re.split(r'(?<=[.!?]) +', text)

            sentences_sentiment = []
            for sentence in sentences:
                sentiment, confidence = predict_sentiment(sentence, tokenizer, model)

                sentences_sentiment.append({
                    'sentence': sentence,
                    'sentiment': sentiment_labels[sentiment],
                    'confidence': confidence,
                })

            transcript_sentiment.append({
                **utterance,
                'sentences': sentences_sentiment,
            })

        else:
            sentiment, confidence = predict_sentiment(text, tokenizer, model)

            transcript_sentiment.append({
                **utterance,
                'sentiment': sentiment_labels[sentiment],
                'confidence': confidence,
            })


    end=time.time()
    print(f"Time taken to analyze the sentiment: {end-start} seconds")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(transcript_sentiment, f, indent=4)
            print(f"Sentiment saved to {save_path}")


if __name__ == "__main__":
    # Take Args from the command line
    parser = argparse.ArgumentParser(description="Sentiment analysis on audio transcripts.")

    parser.add_argument('file_path', type=str, help='Path to the JSON file to be analyzed.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the analyzed transcript (JSON format).')
    parser.add_argument('--device', choices=["cpu", "gpu"], default="cpu", help='Device to run the model on.')
    parser.add_argument('--per_sentence', action='store_true', help='Analyze sentiment per sentence.')

    # Optional arguments for key for the JSON file
    parser.add_argument('--key', type=str, default=None, help='Key for the JSON file.')  # chucks for the JSON file from whisper-tiny

    args = parser.parse_args()


    # set the device
    if args.device =="gpu" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using GPU")

    else:
        DEVICE = torch.device("cpu")
        if args.device == "gpu":
            print("GPU not available. Using CPU")
        else:
            print("Using CPU")



    # Load the transcript
    with open(args.file_path, "r") as file:
        utterances = json.load(file)

        if args.key:
            utterances = utterances[args.key]

        roberta_sentiment_analysis(utterances, args.save_path,args.per_sentence)



# PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript.json
# PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript.json --save_path ./test.json
# PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript.json --save_path ./test.json --device gpu
# PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript.json --save_path ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript_per_sentence.json --device gpu --per_sentence
# PS D:\sentiment-analysis-api>  python -m src.scripts.roberta_predict ./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript_whipher_tiny.json --save_path ./res.json --device gpu --key chunks

# import json 

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Check if CUDA is available and set the device accordingly
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# sentiment_labels = ["negative", "neutral", "positive"]

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis").to(device)


# def predict_sentiment(text):
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(**inputs)
    
        
#     # Convert logits to probabilities
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
#     # Get the predicted sentiment
#     predicted_class = torch.argmax(probabilities, dim=1).item()
    
#     return predicted_class, probabilities[0][predicted_class].item()

# with open("./data/demos/sportify/sportify_full.mp3_utterances_timestamps_transcript.json", "r") as file:
#     utterances = json.load(file)

#     transcript_sentiment=[] 

#     for utterance in utterances:
#         print(utterance)

#         text = utterance["text"]

#         sentiment, confidence = predict_sentiment(text)

#         transcript_sentiment.append({
#             'start_time': utterance["start_time"],
#             'end_time': utterance["end_time"],
#             'speaker': utterance["speaker"],
#             'text': text,
#             'sentiment': sentiment_labels[sentiment],
#             'confidence': confidence
#         })


#         save_path='./sentiment.json'
#         with open(save_path, 'w') as f:
#             json.dump(transcript_sentiment, f, indent=4)
#             print(f"Sentiment saved to {save_path}")


# # PS D:\sentiment-analysis-api> python -m src.scripts.roberta_predict


# # Examples
# # text = "I love using BERT models!"
# # text = "I like spotify! I can't wait to listen to my favorite song"
# # text = "I am not able to use this product. It is useless"
# # text = "I think it should be free"
# # sentiment, confidence = predict_sentiment(text)


# # print(f"Sentiment: {sentiment_labels[sentiment]}")
# # print(f"Confidence: {confidence:.2f}")