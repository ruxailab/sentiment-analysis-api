import os
from urllib.parse import urlparse
import requests
from io import BytesIO
from pydub import AudioSegment

class AudioDataLayer:
    def __init__(self,config):
        # self.config = config
        self.debug = config.get('debug')
       
        
    def fetch_audio(self, url: str):
        """
        Fetch the audio file from a local path or URL.
        :param url: Local file path or URL to the audio file.
        :return: AudioSegment or error message.
        """
        try:
            # Check if the provided URL is a valid URL
            parsed_url = urlparse(url)

            if bool(parsed_url.scheme) and bool(parsed_url.netloc):  # This checks if the URL has a scheme and netloc
                # It's a URL
                if self.debug:
                    print(f"[debug] [Data Layer] [AudioDataLayer] [fetch_audio] Downloading audio file from URL: {url}")
                try:
                    url_response = requests.get(url)
                    if url_response.status_code != 200:
                        # Capture and format the error message for the upper layers
                        error_message = f'An error occurred during the HTTP request: HTTP status: {url_response.status_code}'
                        print(f"[error] [Data Layer] [AudioDataLayer] [fetch_audio] {error_message}")
                        return {'error': error_message} # Return error in structured format
                    
                    # Load audio file into pydub from the response content
                    return AudioSegment.from_file(BytesIO(url_response.content))
                
                except requests.exceptions.RequestException as req_err:
                    # Handle any specific errors related to HTTP requests
                    print(f"[error] [Data Layer] [AudioDataLayer] [fetch_audio] HTTP request error: {str(req_err)}")
                    return {'error': f'An error occurred during the HTTP request: {str(req_err)}'}
                
            elif os.path.exists(url) and os.path.isfile(url):
                # It's a local file path
                if self.debug:
                    print(f"[debug] [Data Layer] [AudioDataLayer] [fetch_audio] Downloading audio file from local path: {url}")
                return AudioSegment.from_file(url)
            
            else:
                # If it's neither a valid URL nor an existing file path, raise an error
                error_message = 'Provided url is neither a valid URL nor a valid file path.'
                print(f"[error] [Data Layer] [AudioDataLayer] [fetch_audio] {error_message}")
                return {'error': error_message}
                

        except Exception as e:
            # Catch any other exceptions
            print(f"[error] [Data Layer] [AudioDataLayer] [fetch_audio] An unexpected error occurred: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}  # Generic error message


        
        

# if __name__ == "__main__":
#     config = {
#         'debug': True
#     }
    # audio_data_layer = AudioDataLayer(config)

    # audio = audio_data_layer.fetch_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v")
    # print("audio",audio)

    # audio = audio_data_layer.fetch_audio("https://invalid-url.com/audio.mp3")
    # print("audio",audio)

    # audio = audio_data_layer.fetch_audio("./samples/sample_0.mp4")
    # print("audio",audio)

    # audio = audio_data_layer.fetch_audio("./non-exist.mp4")
    # print("audio",audio)

# #  Run:
# # python -m app.data.audio_data
    