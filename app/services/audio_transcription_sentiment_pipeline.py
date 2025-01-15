import os

from app.config import Config

# Services
from app.services.audio_service import AudioService
from app.services.transcript_service import TranscriptService
from app.services.sentiment_service import SentimentService

config = Config().config # Load the configuration


from pydantic import BaseModel
from typing import List, Union

class TranscriptionChunk(BaseModel):
    timestamp: List[int]  # [start_time_ms, end_time_ms]
    text: str  # Text from the chunk
    label: Union[str, None] = None  # Sentiment label (optional)
    confidence: Union[float, None] = None  # Sentiment confidence score (optional)

class AudioTranscriptionSentimentResult(BaseModel):
    audio_path: str  # Path to the extracted audio segment
    start_time_ms: int  # Start time of the segment (in milliseconds)
    end_time_ms: int  # End time of the segment (in milliseconds)
    transcription: str  # Full transcription of the audio segment
    utterances_sentiment: List[TranscriptionChunk]  # Sentiment analysis for each chunk

class ErrorResponse(BaseModel):
    error: str  # Error message describing what went wrong

# Union type to handle both successful and error responses
ProcessResponse = Union[AudioTranscriptionSentimentResult, ErrorResponse]


class AudioTranscriptionSentimentPipeline:
    def __init__(self):
        self.debug = config.get('debug')

        self.config = config.get('audio_transcription_sentiment_pipeline')
        self.remove_audio = self.config.get('remove_audio')

        self.audio_service = AudioService()
        self.transcript_service = TranscriptService()
        self.sentiment_service = SentimentService()

    def process(self, url: str, start_time_ms: int, end_time_ms: int = None, user_id: str = None)-> ProcessResponse:
        """
        Process the Video/Audio file by extracting a segment, transcribing it, and performing sentiment analysis.
        :param url: URL or local file path to the audio file.
        :param start_time_ms: Start time of the segment to extract (in milliseconds).
        :param end_time_ms: End time of the segment to extract (in milliseconds).
        :param user_id: (Optional) User ID for creating user-specific subdirectories
        :return: Transcription, sentiment analysis, and audio segment details
        """
        try:
            # Step(1) Extract the audio segment
            audio_result = self.audio_service.extract_audio(url, start_time_ms, end_time_ms, user_id)

            if isinstance(audio_result, dict) and 'error' in audio_result:
                return {
                    'error': f'An error occurred while extracting the audio: {audio_result["error"]}'
                }
            
            if self.debug:
                print("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [audio_result]", audio_result)
            
            # Parse the audio segment details
            audio_path = audio_result['audio_path']
            start_time_ms = audio_result['start_time_ms']
            end_time_ms = audio_result['end_time_ms']

            # Step(2) Transcribe the audio segment
            transcription_result = self.transcript_service.transcribe(audio_path)

            if isinstance(transcription_result, dict) and 'error' in transcription_result:
                return {
                    'error': f'An error occurred while transcribing the audio: {transcription_result["error"]}'
                }
            
            if self.debug:
                print("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [transcription_result]", transcription_result)

            # Parse the transcription details
            transcription = transcription_result['transcription'] # Full transcription text
            chunks = transcription_result['chunks'] # Transcription chunks [{'timestamp': (,), 'text':""}]


            # Remove the audio file after processing
            if self.remove_audio:
                print(f"[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] Removing audio file: {audio_path}")
                os.remove(audio_path)


            # Step(3) Perform sentiment [Per chunk :D]
            for chunk in chunks:
                timestamp = chunk['timestamp']
                text = chunk['text']

                sentiment_result = self.sentiment_service.analyze(text)
                if isinstance(sentiment_result, dict) and 'error' in sentiment_result:
                    print("[error] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [sentiment_result]", sentiment_result)
                    continue # Skip this chunk if there was an error :D

                if self.debug:
                    print("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [sentiment_result]", sentiment_result)

                # Add the sentiment result to the chunk
                chunk['label'] = sentiment_result['label']
                chunk['confidence'] = sentiment_result['confidence']

            # Return the transcription, sentiment analysis, and audio segment details
            return {
                'audio_path': audio_path,
                'start_time_ms': start_time_ms,
                'end_time_ms': end_time_ms,
                'transcription': transcription,
                'utterances_sentiment': chunks,
            }
        except Exception as e:
            print(f"[error] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] An error occurred during processing: {str(e)}")
            return {'error': f'An error occurred during processing: {str(e)}'}
        


# if __name__ == "__main__":
#     pipeline = AudioTranscriptionSentimentPipeline()
#     print("pipeline",pipeline)

    # # URL to Video File
    # result = pipeline.process("https://drive.usercontent.google.com/u/2/uc?id=1BJ-0fvbc0mlDWaBGci0Ma-f1k6iElh6v", 0, 10000)
    # print("result",result)

    # # Invalid URL Video
    # result = pipeline.process("https://invalid-url.com/video.mp4", 0, 10000)
    # print("result",result)

    # # Local Video File Path    
    # result = pipeline.process("./samples/sample_0.mp4", 0, 10000)
    # print("result",result)

    # # Invalid Video File Path
    # result = pipeline.process("./samples/non-exist.mp4", 0, 10000)
    # print("result",result)

    # # Local Audio File Path
    # result = pipeline.process("./samples/sample_1.mp3", 0, 10000)
    # print("result",result)


# #  Run:
# #  python -m app.services.audio_transcription_sentiment_pipeline