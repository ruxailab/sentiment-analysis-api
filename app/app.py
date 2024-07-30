import sys
import os
import time
import json
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
    

@app.route('/split-audio', methods=['POST'])
def split_audio():

    # Get the video file [Path]
    video_file = request.json['video_file']
    # Check if video_file is provided
    if not video_file:
        return jsonify({'error': 'No video file path provided'}), 400

    timing = {
        'video_to_audio_time': None,
        'split_audio_time': None
    }

    # Extract Audio from Video
    start_time = time.time()
    audio = video_2_audio(video_file,True,"temp/temp.mp3")
    end_time = time.time()
    timing['video_to_audio_time'] = end_time - start_time

    # Split the audio by silence
    start_time_split = time.time()
    audio = AudioSegment.from_file('temp/temp.mp3')
    chunks = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-50)

    # Initialize variables
    start_time = 0
    chunk_info = []

    # Process each chunk and keep track of timestamps
    for i,chunk in enumerate(chunks):
        end_time = start_time + len(chunk)  # Calculate end time
        chunk_info.append({
            'chunk': i,
            'start_time': start_time,
            'end_time': end_time
        })

        # Save chunk in .mp3 format
        chunk.export(f"temp/chunck_{i}_{start_time}_{end_time}.mp3", format="mp3")

        start_time = end_time  # Update start time for the next chunk
    end_time_split = time.time()
    timing['split_audio_time'] = end_time_split - start_time_split

    # Save the chunk_info to a JSON file
    import json
    with open('temp/chunk_info.json', 'w') as file:
        json.dump(chunk_info, file)



    response = {
        video_file: video_file,
        # 'whisper_model_size': whisper_model_size,
        'chunks_len': len(chunks),
        'timing': timing,
    }

    return jsonify(response), 200
    




@app.route('/sentiment-analysis-timestamp', methods=['POST'])
def sentiment_analysis_timestamp():    
    # Get start and end time
    start_time = request.json['start_time']
    end_time = request.json['end_time']

    # Check if start_time and end_time are provided
    if not start_time or not end_time:
        return jsonify({'error': 'Start time or end time not provided'}), 400
    
    # Convert seconds to milliseconds
    start_time = start_time * 1000
    end_time = end_time * 1000

    # Get the whisper model size
    whisper_model_size = request.json['whisper_model_size']
    
    # Check if whisper_model_size is provided
    if not whisper_model_size:
        return jsonify({'error': 'Whisper model size not provided'}), 400
    
    # Check if whisper_model_size is valid
    if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
        return jsonify({'error': 'Invalid whisper model size'}), 400

    
    # Get chuck file
    # TODO get that from id of the answer
    # Read and parse the JSON file
    with open('temp/chunk_info.json', 'r') as file:
        chunks = json.load(file)

    # # Check if the timestamp is valid
    min_timestamp = 0
    max_timestamp = chunks[-1]['end_time']

    if start_time < min_timestamp or end_time > max_timestamp:
        return jsonify({'error': f'Invalid timestamp range. Timestamp should be between {min_timestamp} and {max_timestamp}'}), 400
    
    # Get the nearest chunk to the start time
    for chunk in chunks:
        if start_time >= chunk['start_time'] and start_time <= chunk['end_time']:
            start_chunk = chunk
            break

    # Get the nearest chunk to the end time
    for chunk in chunks:
        if end_time >= chunk['start_time'] and end_time <= chunk['end_time']:
            end_chunk = chunk
            break

    # Process the chunks
    # Get the device from environment variables
    device = os.getenv('DEVICE', 'cpu')

    # Make Inference Object
    inference=Inference(whisper_model_size=whisper_model_size, device=device)


    # Loop through the chunks
    sentiment = []

    for i in range(start_chunk['chunk'], end_chunk['chunk']+1):

        # Check if this chuck has previously been processed
        if False:
            # Just read previous result
            pass
        else:
            # Perform inference to get the transcript sentiment
            transcript_sentiment = inference.infer_2(f'temp/chunck_{i}_{chunks[i]["start_time"]}_{chunks[i]["end_time"]}.mp3')
            sentiment.append(transcript_sentiment)

    # Format the response
    response = {
        'start_time': start_time,
        'end_time': end_time,
        'start_chunk': start_chunk,
        'end_chunk': end_chunk,
        'sentiment': sentiment
    }

    return jsonify(response), 200
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

# # Create a virtual environment named 'env'
# python3 -m venv env
# # Activate the virtual environment
# source env/bin/activate


#  ./env/Scripts/activate
#  python -m flask run
#  python -m venv env  
#  python -m pip install flask