import sys
import os
# Add the path to the sentiment_analysis module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
from src.sentiment_analysis.inference.inference import Inference

# Define the app
from flask import Flask, jsonify, request


app = Flask(__name__)
# 
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Hello, World from Sentiment Analysis App!'})


@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():

    # Get the video file [Path]
    video_file = request.json['video_file']
    print(video_file)


    if not video_file:
        return jsonify({'error': 'No video file path provided'}), 400

    # Make Inference Object
    inference=Inference()
    # print("Inference Object Created")

    try:
        # Perform inference to get the transcript sentiment
        transcript_sentiment = inference.infer(video_file)
        
        # Format the response
        response = {
            'data': transcript_sentiment
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
#  python -m flask run
#  python -m venv env  
#  .\env\Scripts\activate
#  python -m pip install flask