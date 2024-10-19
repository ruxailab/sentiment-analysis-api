import sys
import os


import requests
from io import BytesIO

# Add the path to the sentiment_analysis module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))

# Import the required modules
from src.sentiment_analysis.inference.inference import Inference
from pydub import AudioSegment, silence


# Define the FLask app
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes and restrict to specific origins
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# Load environment variables from .env file
# load_dotenv()

# API Documentation
from flasgger import Swagger
from flask_restful import Api, Resource

swagger = Swagger(app,template={
    "info": {
        "title": "Sentiment Analysis API",
        "description": "API for performing sentiment analysis.",
        "version": "1.0.0"
    }
})
# Initialize Flask-RESTful API
api = Api(app)

class Hello(Resource):
    def get(self):
        """
        This is an example endpoint that returns 'Hello from Sentiment Analysis App!!'
        ---
        responses:
            200:
                description: A successful response
                examples:
                    application/json: "Hello from Sentiment Analysis App!!"
        """
        return {'message': 'Hello from Sentiment Analysis App!!'} , 200



class WhispherSentimentAnalysis(Resource):
    def post(self):
        """
        Perform sentiment analysis on a video file using a Whisper model.
        ---
        tags:
          - Sentiment Analysis
        parameters:
            - name: url
              in: body
              type: string
              required: true
              description: The URL of the video file to analyze.
              example: "https://example.com/path/to/video.mp4"

            - name: whisper_model_size
              in: body
              type: string
              required: true
              enum: [tiny, base, small, medium, large, large-v2, large-v3]
              description: |
                The size of the Whisper model to use for analysis. Available sizes are:
                - `tiny`: Smallest model, fastest but less accurate.
                - `base`: Base model, moderate speed and accuracy.
                - `small`: Small model, balanced between speed and accuracy.
                - `medium`: Medium model, higher accuracy but slower.
                - `large`: Large model, high accuracy with slower performance.
                - `large-v2`: Improved large model, better accuracy.
                - `large-v3`: Latest large model, highest accuracy.
              example: "small"
        reponses:
            200:
                description: Sentiment analysis results.
                content:
                    application/json:
                        schema:
                            type: object
                            properties:
                                url:
                                    type: string
                                    description: The URL of the analyzed video file.
                                whisper_model_size:
                                    type: string
                                    description: The Whisper model size used for analysis.
                                sentiment:
                                    type: array
                                    items:
                                        type: object
                                        properties:
                                            text:
                                                type: string
            400:
                description: Bad Request. Returned when required data is missing or invalid.

            500:
                description: Internal Server Error. Returned when there is an issue with performing the analysis.
        """
        json_data = request.json

        # Get the video file [Path]
        video_file = json_data.get('video_file')
        # print(video_file)
        if not video_file:
            return {'error': 'No video file path provided'}, 400
        
        # Get the whisper model size
        whisper_model_size = json_data.get('whisper_model_size')

        # print(whisper_model_size)
        if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
            return {'error': 'Invalid whisper model size'}, 400
        

        # Get the device from environment variables
        device = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if DEVICE is not specified

        # Make Inference Object
        inference=Inference(whisper_model_size=whisper_model_size, device=device)
        # print("Inference Object Created")

  

        try:
            # Perform inference to get the transcript sentiment
            transcript_sentiment = inference.infer(video_file)
            # print("transcript_sentiment",transcript_sentiment)

            # Format the response
            response = {
                'video_file': video_file,
                'whisper_model_size': whisper_model_size,
                'sentiment': transcript_sentiment
            }


            return response , 200
        except Exception as e:
            return {'error': str(e)}, 500

class WhispherTimeStampedSentimentAnalysis(Resource):
    def post(self):
        """
        Perform sentiment analysis on a specified timestamped region of an audio file using a Whisper model.
        ---
        tags:
          - Time Stamped Sentiment Analysis
        parameters:
          - name: url
            in: body
            type: string
            required: true
            description: The URL of the audio file to analyze.
            example: "https://example.com/audiofile.mp3"
          - name: start_time
            in: body
            type: number
            required: true
            description: The start time (in seconds) of the region to analyze.
            example: 10
          - name: end_time
            in: body
            type: number
            required: true
            description: The end time (in seconds) of the region to analyze.
            example: 20.0
          - name: whisper_model_size
            in: body
            type: string
            required: true
            enum: [tiny, base, small, medium, large, large-v2, large-v3]
            description: |
              The size of the Whisper model to use for analysis. Available sizes are:
              - `tiny`: Smallest model, fastest but less accurate.
              - `base`: Base model, moderate speed and accuracy.
              - `small`: Small model, balanced between speed and accuracy.
              - `medium`: Medium model, higher accuracy but slower.
              - `large`: Large model, high accuracy with slower performance.
              - `large-v2`: Improved large model, better accuracy.
              - `large-v3`: Latest large model, highest accuracy.
            example: "base"
        responses:
          200:
            description: Sentiment analysis results.
            schema:
              type: object
              properties:
                url:
                  type: string
                  description: The URL of the analyzed audio file.
                start_time:
                  type: number
                  description: The start time (in seconds) of the analyzed region.
                end_time:
                  type: number
                  description: The end time (in seconds) of the analyzed region.
                whisper_model_size:
                  type: string
                  description: The Whisper model size used for analysis.
                utterances_sentiment:
                  type: array
                  items:
                    type: object
                    properties:
                      timestamp:
                        type: array
                        items:
                          type: number
                        description: Start and end timestamps of the transcribed text.
                      text:
                        type: string
                        description: The transcribed text.
                      sentiment:
                        type: string
                        description: The sentiment of the text (e.g., positive, negative, neutral).
                      confidence:
                        type: number
                        description: The confidence level of the sentiment analysis.
          400:
            description: Bad Request. Returned when required data is missing or invalid.
          500:
            description: Internal Server Error. Returned when there is an issue with downloading or processing the audio file.
        """
        try:
          data = request.json

          url = data.get('url')
          start_time = data.get('start_time')
          end_time = data.get('end_time')
          whisper_model_size = data.get('whisper_model_size')


          # print(url)
          # print(start_time)
          # print(end_time)
          # print(whisper_model_size)


          if not url:
              return {'error': 'No URL provided'}, 400
          
          if start_time!=0 and not start_time:
              return {'error': 'No start time provided'}, 400
          
          if end_time!=0 and not end_time:
              return {'error': 'No end time provided'}, 400

          # Check if whisper_model_size is provided
          if not whisper_model_size:
              return {'error': 'Whisper model size not provided'}, 400
          
          # Check if whisper_model_size is valid
          if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
              return {'error': 'Invalid whisper model size'}, 400



          # Convert start_time and end_time to milliseconds
          start_time_ms = start_time * 1000
          end_time_ms = end_time * 1000


          # Download the audio file from the Storage
          url_response = requests.get(url)
          if url_response.status_code != 200:
              return {'error': 'Failed to download audio file'}, 500
          

          # Load audio file into pydub
          audio = AudioSegment.from_file(BytesIO(url_response.content))

          # Extract the specified region
          segment = audio[start_time_ms:end_time_ms]


          # Save the segment to .mp3
          segment.export('temp.mp3', format='mp3')   


          # Get the device from environment variables
          device = os.getenv('DEVICE', 'cpu')
    

          # Make Inference Object
          inference=Inference(whisper_model_size=whisper_model_size, device=device)


          transcript_sentiment = inference.infer_audio_file('temp.mp3')
          # print(transcript_sentiment)

          # Add start time stamp to each utterance
          for utterance in transcript_sentiment:
              utterance['timestamp'] = (start_time + utterance['timestamp'][0], start_time + utterance['timestamp'][1])

          response = {
              'url': url,
              'start_time': start_time,
              'end_time': end_time,
              'whisper_model_size': whisper_model_size,   
              'utterances_sentiment': transcript_sentiment # Array of dictionaries {timestamp, text, sentiment, confidence}
          }

          return response, 200
          
        except Exception as e:
          return {'error': str(e)}, 500


# Add the resources to the API
api.add_resource(Hello, '/')
api.add_resource(WhispherSentimentAnalysis, '/sentiment-analysis/whisper')
api.add_resource(WhispherTimeStampedSentimentAnalysis, '/sentiment-analysis-timestamped/whisper')




# @app.route('/async-sentiment-analysis/whisper', methods=['POST'])
# def async_sentiment_analysis():

#     # Get the video file [Path]
#     video_file = request.json['video_file']


#     # Get the whisper model size
#     whisper_model_size = request.json['whisper_model_size']
#     # print(whisper_model_size)
#     if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
#         return jsonify({'error': 'Invalid whisper model size'}), 400
    

#     # Get the device from environment variables
#     device = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if DEVICE is not specified


#     # Initialize timing dictionary
#     timing = {
#         'video_to_audio_time': None,
#         'split_audio_time': None,
#         'inference_times':{
#             'total_time': 0,
#             'chunks': []
#         }
#     }

#     # Make Inference Object
#     inference=Inference(whisper_model_size=whisper_model_size, device=device)


#     # Extract Audio from Video
#     start_time = time.time()
#     audio=video_2_audio(video_file,True,"temp.mp3")
#     end_time = time.time()
#     timing['video_to_audio_time'] = end_time - start_time

#     # Split the audio by silence
#     start_time = time.time()
#     audio = AudioSegment.from_file('temp.mp3')
#     chunks = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-50)
#     end_time = time.time()
#     timing['split_audio_time'] = end_time - start_time


#     try:
#         sentiment = []

#         for i, chunk in enumerate(chunks):
#             start_time = time.time()
#             # Save in .mp3 temp
#             chunk.export(f"temp.mp3", format="mp3")

#             # Perform inference to get the transcript sentiment
#             transcript_sentiment = inference.infer_2('temp.mp3')
#             sentiment.append(transcript_sentiment)
#             end_time = time.time()

#             timing['inference_times']['chunks'].append(end_time - start_time)
#             timing['inference_times']['total_time'] += end_time - start_time
        
#         # Format the response
#         response = {
#             # 'video_file': video_file,
#             # 'whisper_model_size': whisper_model_size,
#             'sentiment': sentiment,
#             'timing': timing
#         }

#         return jsonify(response), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    

# @app.route('/sentiment-analysis-timestamp', methods=['POST'])
# def sentiment_analysis_timestamp():

#     # Get the video file [Path]
#     video_file = request.json['video_file']

#         # Check if video_file is provided
#     if not video_file:
#         return jsonify({'error': 'No video file path provided'}), 400

#     # Get start and end time
#     start_time = request.json['start_time']
#     end_time = request.json['end_time']

#     # Check if start_time and end_time are provided
#     if not start_time or not end_time:
#         return jsonify({'error': 'Start time or end time not provided'}), 400
    
#     # Get the whisper model size
#     whisper_model_size = request.json['whisper_model_size']
    
#     # Check if whisper_model_size is provided
#     if not whisper_model_size:
#         return jsonify({'error': 'Whisper model size not provided'}), 400
    
#     # Check if whisper_model_size is valid
#     if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
#         return jsonify({'error': 'Invalid whisper model size'}), 400
    


#     # Extract Audio from Video
#     audio = video_2_audio(video_file,True,"temp.mp3")
#     if not audio:
#         return jsonify({'error': 'Error extracting audio'}), 500
    
#     audio = AudioSegment.from_file('temp.mp3')
#     # print(audio.duration_seconds)
    
#     if start_time < 0 or end_time > audio.duration_seconds:
#         return jsonify({'error': 'Invalid timestamp range'}), 400
    


#     # Convert start_time and end_time to milliseconds
#     start_time_ms = start_time * 1000
#     end_time_ms = end_time * 1000

#     # Slice the audio file
#     audio_segment = audio[start_time_ms:end_time_ms]

#     # Save the sliced audio file
#     audio_segment.export('temp.mp3', format='mp3')


#     # Get the device from environment variables
#     device = os.getenv('DEVICE', 'cpu')
    
#     # Make Inference Object
#     inference=Inference(whisper_model_size=whisper_model_size, device=device)

#     # Perform inference to get the transcript sentiment
#     transcript_sentiment = inference.infer_2(f'temp.mp3')



#     response = {
#         "transcript_sentiment": transcript_sentiment
#     }


#     return jsonify(response), 200
    

    

# @app.route('/split-audio', methods=['POST'])
# def split_audio():

#     # Get the video file [Path]
#     video_file = request.json['video_file']
#     # Check if video_file is provided
#     if not video_file:
#         return jsonify({'error': 'No video file path provided'}), 400

#     timing = {
#         'video_to_audio_time': None,
#         'split_audio_time': None
#     }

#     # Extract Audio from Video
#     start_time = time.time()
#     audio = video_2_audio(video_file,True,"temp/temp.mp3")
#     end_time = time.time()
#     timing['video_to_audio_time'] = end_time - start_time

#     # Split the audio by silence
#     start_time_split = time.time()
#     audio = AudioSegment.from_file('temp/temp.mp3')
#     chunks = silence.split_on_silence(audio, min_silence_len=1000, silence_thresh=-50)

#     # Initialize variables
#     start_time = 0
#     chunk_info = []

#     # Process each chunk and keep track of timestamps
#     for i,chunk in enumerate(chunks):
#         end_time = start_time + len(chunk)  # Calculate end time
#         chunk_info.append({
#             'chunk': i,
#             'start_time': start_time,
#             'end_time': end_time
#         })

#         # Save chunk in .mp3 format
#         chunk.export(f"temp/chunck_{i}_{start_time}_{end_time}.mp3", format="mp3")

#         start_time = end_time  # Update start time for the next chunk
#     end_time_split = time.time()
#     timing['split_audio_time'] = end_time_split - start_time_split

#     # Save the chunk_info to a JSON file
#     import json
#     with open('temp/chunk_info.json', 'w') as file:
#         json.dump(chunk_info, file)



#     response = {
#         video_file: video_file,
#         # 'whisper_model_size': whisper_model_size,
#         'chunks_len': len(chunks),
#         'timing': timing,
#     }

#     return jsonify(response), 200
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


# Create a virtual environment named 'env'
# python3 -m venv env

# Activate the virtual environment
# source env/bin/activate

# Run Flask App
# python app.py
