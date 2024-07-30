import sys
import os
import time
# from dotenv import load_dotenv

# Add the path to the sentiment_analysis module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
from src.sentiment_analysis.inference.inference import Inference
from src.video_processing.video_2_audio import video_2_audio

from pydub import AudioSegment, silence



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


@app.route('/async-sentiment-analysis/whisper', methods=['POST'])
def async_sentiment_analysis():

    # Get the video file [Path]
    video_file = request.json['video_file']


    # Get the whisper model size
    whisper_model_size = request.json['whisper_model_size']
    # print(whisper_model_size)
    if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
        return jsonify({'error': 'Invalid whisper model size'}), 400
    

    # Get the device from environment variables
    device = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if DEVICE is not specified


    # Initialize timing dictionary
    timing = {
        'video_to_audio_time': None,
        'split_audio_time': None,
        'inference_times':{
            'total_time': 0,
            'chunks': []
        }
    }

    # Make Inference Object
    inference=Inference(whisper_model_size=whisper_model_size, device=device)


    # Extract Audio from Video
    start_time = time.time()
    audio=video_2_audio(video_file,True,"temp.mp3")
    end_time = time.time()
    timing['video_to_audio_time'] = end_time - start_time

    # Split the audio by silence
    start_time = time.time()
    audio = AudioSegment.from_file('temp.mp3')
    chunks = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-50)
    end_time = time.time()
    timing['split_audio_time'] = end_time - start_time


    try:
        sentiment = []

        for i, chunk in enumerate(chunks):
            start_time = time.time()
            # Save in .mp3 temp
            chunk.export(f"temp.mp3", format="mp3")

            # Perform inference to get the transcript sentiment
            transcript_sentiment = inference.infer_2('temp.mp3')
            sentiment.append(transcript_sentiment)
            end_time = time.time()

            timing['inference_times']['chunks'].append(end_time - start_time)
            timing['inference_times']['total_time'] += end_time - start_time
        
        # Format the response
        response = {
            # 'video_file': video_file,
            # 'whisper_model_size': whisper_model_size,
            'sentiment': sentiment,
            'timing': timing
        }

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
#  .\env\Scripts\activate
#  python -m flask run
#  python -m venv env  
#  python -m pip install flask