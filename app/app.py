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

    # Make Inference Object
    inference=Inference()
    print("Inference Object Created")
    # Get 
    return jsonify({'message': 'Hello, World from Sentiment Analysis App!'})

#  python -m flask run
#  python -m venv env  
#  .\env\Scripts\activate
#  python -m pip install flask