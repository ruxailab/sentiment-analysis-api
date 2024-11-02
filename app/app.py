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



# class WhispherSentimentAnalysis(Resource):
#     def post(self):
#         """
#         Perform sentiment analysis on a video file using a Whisper model.
#         ---
#         tags:
#           - Sentiment Analysis
#         parameters:
#             - name: url
#               in: body
#               type: string
#               required: true
#               description: The URL of the video file to analyze.
#               example: "https://example.com/path/to/video.mp4"

#             - name: whisper_model_size
#               in: body
#               type: string
#               required: true
#               enum: [tiny, base, small, medium, large, large-v2, large-v3]
#               description: |
#                 The size of the Whisper model to use for analysis. Available sizes are:
#                 - `tiny`: Smallest model, fastest but less accurate.
#                 - `base`: Base model, moderate speed and accuracy.
#                 - `small`: Small model, balanced between speed and accuracy.
#                 - `medium`: Medium model, higher accuracy but slower.
#                 - `large`: Large model, high accuracy with slower performance.
#                 - `large-v2`: Improved large model, better accuracy.
#                 - `large-v3`: Latest large model, highest accuracy.
#               example: "small"
#         reponses:
#             200:
#                 description: Sentiment analysis results.
#                 content:
#                     application/json:
#                         schema:
#                             type: object
#                             properties:
#                                 url:
#                                     type: string
#                                     description: The URL of the analyzed video file.
#                                 whisper_model_size:
#                                     type: string
#                                     description: The Whisper model size used for analysis.
#                                 sentiment:
#                                     type: array
#                                     items:
#                                         type: object
#                                         properties:
#                                             text:
#                                                 type: string
#             400:
#                 description: Bad Request. Returned when required data is missing or invalid.

#             500:
#                 description: Internal Server Error. Returned when there is an issue with performing the analysis.
#         """
#         json_data = request.json

#         # Get the video file [Path]
#         video_file = json_data.get('video_file')
#         # print(video_file)
#         if not video_file:
#             return {'error': 'No video file path provided'}, 400
        
#         # Get the whisper model size
#         whisper_model_size = json_data.get('whisper_model_size')

#         # print(whisper_model_size)
#         if whisper_model_size not in ['tiny','base','small','medium','large','large-v2','large-v3']:
#             return {'error': 'Invalid whisper model size'}, 400
        

#         # Get the device from environment variables
#         device = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if DEVICE is not specified

#         # Make Inference Object
#         inference=Inference(whisper_model_size=whisper_model_size, device=device)
#         # print("Inference Object Created")

  

#         try:
#             # Perform inference to get the transcript sentiment
#             transcript_sentiment = inference.infer(video_file)
#             # print("transcript_sentiment",transcript_sentiment)

#             # Format the response
#             response = {
#                 'video_file': video_file,
#                 'whisper_model_size': whisper_model_size,
#                 'sentiment': transcript_sentiment
#             }


#             return response , 200
#         except Exception as e:
#             return {'error': str(e)}, 500

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
                  description: The URL or path of the analyzed audio file
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

          print("Path:", url)
          print("Does file exist:", os.path.isfile(url))


          # Download the audio file from the Storage
          # Determine if url is a URL or a local file
          if os.path.isfile(url):
              # Load audio from the local file
              audio = AudioSegment.from_file(url)
          else:
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
          print("Running on ", device)
    

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
# api.add_resource(WhispherSentimentAnalysis, '/sentiment-analysis/whisper')
api.add_resource(WhispherTimeStampedSentimentAnalysis, '/sentiment-analysis-timestamped/whisper')    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


# Create a virtual environment named 'env'
# python3 -m venv env

# Activate the virtual environment
# source env/bin/activate

# Run Flask App
# python app.py
