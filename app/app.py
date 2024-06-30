import sys
import os
# from dotenv import load_dotenv

# Add the path to the sentiment_analysis module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
from src.sentiment_analysis.inference.inference import Inference

# Define the app
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load environment variables from .env file
# load_dotenv()

# 
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'Hello, World from Sentiment Analysis App!'})



@app.route('/sentiment-analysis/whisper', methods=['POST'])
def sentiment_analysis():


    # Get the video file [Path]
    video_file = request.json['video_file']
    # print(video_file)
    if not video_file:
        return jsonify({'error': 'No video file path provided'}), 400
    
    # Get the whisper model size
    whisper_model_size = request.json['whisper_model_size']
    # print(whisper_model_size)
    if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
        return jsonify({'error': 'Invalid whisper model size'}), 400
    

    # Get the device from environment variables
    device = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if DEVICE is not specified

    # Make Inference Object
    inference=Inference(whisper_model_size=whisper_model_size, device=device)
    # print("Inference Object Created")

    try:
        # Perform inference to get the transcript sentiment
        transcript_sentiment = inference.infer(video_file)
        
        # Format the response
        response = {
            'video_file': video_file,
            'whisper_model_size': whisper_model_size,
            'sentiment': transcript_sentiment
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
#  .\env\Scripts\activate
#  python -m flask run
#  python -m venv env  
#  python -m pip install flask