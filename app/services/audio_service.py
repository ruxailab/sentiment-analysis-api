import uuid
import os
from pydub import AudioSegment

from app.config import Config

# Data layer for fetching audio files
from app.data.audio_data import AudioDataLayer

config = Config().config # Load the configuration

class AudioService:
    def __init__(self,static_folder="static/audio"):
        self.debug = config.get('debug')

        self.audio_data_layer = AudioDataLayer(config)
        self.static_folder = static_folder

    def extract_audio(self, url: str, start_time_ms: int, end_time_ms: int = None, user_id: str = None):
        """
        Extract a segment from the audio file.
        :param url: URL or local file path to the audio file.
        :param start_time_ms: Start time of the segment to extract (in milliseconds).
        :param end_time_ms: End time of the segment to extract (in milliseconds).
        :param user_id: (Optional) User ID for creating user-specific subdirectories.
        :return: Path to the saved audio file or error message
        """
        try:
            # Validate start_time_ms (must be a non-negative integer)
            if not isinstance(start_time_ms, int) or start_time_ms < 0:
                return {
                    'error': 'Start time must be a non-negative integer.'
                }
        
            # Fetch the audio file using the AudioDataLayer [Data Layer]
            audio = self.audio_data_layer.fetch_audio(url)

            if isinstance(audio, dict) and 'error' in audio:
                # If there was an error fetching the audio, return it
                return {
                    'error': audio['error'] # Return the error message``
                }
            
            # If end_time_ms is None, set it to the length of the audio file
            if end_time_ms is None or end_time_ms > len(audio):
                end_time_ms = len(audio)

            # Validate that end_time_ms is not less than start_time_ms
            if end_time_ms < start_time_ms:
                return {
                    'error': 'End time must not be less than start time.'
                }

            # Extract the segment from the audio
            extracted_audio = audio[start_time_ms:end_time_ms]

            # Save the extracted audio (with a unique filename)
            file_path = self._save_audio(extracted_audio, user_id)

            # Return the file path or URL to access the audio
            return {
                "audio_path": file_path,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms
            }
            
        except Exception as e:
            # Catch any other exceptions
            print(f"[error] [Service Layer] [AudioService] [extract_audio] An error occurred during the audio extraction: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}  # Generic error message


    def _save_audio(self, audio: AudioSegment, user_id: str = None):
        """
        Save the audio segment with a unique filename.
        :param audio: The audio segment to save.
        :param user_id: (Optional) User ID for creating user-specific subdirectories.
        :return: The path to the saved audio file.
        """
        # Generate a unique filename using UUID
        unique_filename = f"{str(uuid.uuid4())}_audio.mp3"

        # Optionally, create a user-specific subdirectory if user_id is provided
        if user_id:
            user_folder = os.path.join(self.static_folder, user_id).replace("\\", "/")
            os.makedirs(user_folder, exist_ok=True)
            file_path = f"{user_folder}/{unique_filename}"
        else:
            os.makedirs(self.static_folder, exist_ok=True)
            file_path = f"{self.static_folder}/{unique_filename}"


        # Export the audio to the file path
        audio.export(file_path, format="mp3")

        return file_path


if __name__ == "__main__":
    service = AudioService()
   
    # # start_time_ms < 0
    # audio = service.extract_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", -100,100)
    # print("audio",audio)
   
    # # Invalid URL
    # audio = service.extract_audio("https://invalid-url.com/audio.mp3", 0, 5000)
    # print("audio",audio)

    # # Invalid local file
    # audio = service.extract_audio("./samples/non-exist.mp4",0,5000)
    # print("audio",audio)

    # # end_time_ms is None
    # audio = service.extract_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", 0)
    # print("audio",audio)

    # # end_time_ms > len(audio)
    # audio = service.extract_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", 0, 500000000)
    # print("audio",audio)

    # # start_time_ms > end_time_ms
    # audio = service.extract_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", 500,100)
    # print("audio",audio)

    # # From URL (Normal Case)
    # result = service.extract_audio("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", 0, 5000)
    # print(result)

    # # From local file (Normal Case)
    # audio = service.extract_audio("./samples/sample_0.mp4",0,5000)
    # print("audio",audio)



# #  Run:
# #  python -m app.services.audio_service