import os

from app.config import Config

from app.utils.logger import logger

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
                # If there was an error extracting the audio, return it
                return {
                    'error': audio_result["error"] # Return the error message
                }
            
            if self.debug:
                logger.debug(f"[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [audio_result]", audio_result)
                # print("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [audio_result]", audio_result)
            
            # Parse the audio segment details
            audio_path = audio_result['audio_path']
            start_time_ms = audio_result['start_time_ms']
            end_time_ms = audio_result['end_time_ms']

            # Step(2) Transcribe the audio segment
            transcription_result = self.transcript_service.transcribe(audio_path)

            if isinstance(transcription_result, dict) and 'error' in transcription_result:
                return {
                    'error': transcription_result['error'] # Return the error message
                }
            
            if self.debug:
                logger.debug("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [transcription_result]", transcription_result)
                # print("[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] [transcription_result]", transcription_result)

            # Parse the transcription details
            transcription = transcription_result['transcription'] # Full transcription text
            chunks = transcription_result['chunks'] # Transcription chunks [{'timestamp': (,), 'text':""}]


            # Remove the audio file after processing
            if self.remove_audio:
                logger.debug(f"[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] Removing audio file: {audio_path}")
                # print(f"[debug] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] Removing audio file: {audio_path}")
                os.remove(audio_path)


            # Step(3) Perform sentiment analysis [Batch Processing]
            try:
                # Extract all text chunks into a list for batch processing
                texts_to_analyze = [chunk['text'] for chunk in chunks]

                if texts_to_analyze:
                    # Perform batch inference
                    batch_results = self.sentiment_service.analyze(texts_to_analyze)

                    if isinstance(batch_results, dict) and 'error' in batch_results:
                        logger.error(f"[error] [Service Layer] [Pipeline] Batch sentiment analysis failed: {batch_results['error']}")
                        # Handle the error by marking chunks if necessary
                    else:
                        # Map the results back to each chunk
                        for i, result in enumerate(batch_results):
                            chunks[i]['label'] = result.get('label')
                            chunks[i]['confidence'] = result.get('confidence')
                
                if self.debug:
                    logger.debug(f"[debug] [Service Layer] [Pipeline] Processed {len(chunks)} chunks using batching.")

            except Exception as e:
                logger.error(f"[error] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] Sentiment batching failed: {str(e)}")
                # Optional: fallback to the original chunks without sentiment if critical

            # Return the full result
            return {
                'audio_path': audio_path,
                'start_time_ms': start_time_ms,
                'end_time_ms': end_time_ms,
                'transcription': transcription,
                'utterances_sentiment': chunks,
            }
        except Exception as e:
            logger.error(f"[error] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] An error occurred during processing: {str(e)}")
            # print(f"[error] [Service Layer] [AudioTranscriptionSentimentPipeline] [process] An error occurred during processing: {str(e)}")
            return {'error': 'An unexpected error occurred while processing the request.'}  # Generic error message
        


# if __name__ == "__main__":
#     # Initialize the pipeline
#     pipeline = AudioTranscriptionSentimentPipeline()
#     print(f"Pipeline initialized: {pipeline}")

#     # Test with a local video or audio file (Make sure the path exists)
#     sample_path = "./samples/sample_0.mp4" 
    
#     if os.path.exists(sample_path):
#         print(f"\n--- Processing Sample: {sample_path} ---")
#         # Extract and analyze the first 30 seconds
#         result = pipeline.process(sample_path, start_time_ms=0, end_time_ms=30000)

#         if 'error' in result:
#             print(f"Error occurred: {result['error']}")
#         else:
#             print(f"Successfully processed audio: {result['audio_path']}")
#             print(f"Full Transcription: {result['transcription'][:100]}...") # Print first 100 chars
            
#             # Check the batch results
#             utterances = result.get('utterances_sentiment', [])
#             print(f"Number of chunks analyzed: {len(utterances)}")
            
#             # Print the first few results to verify
#             for i, chunk in enumerate(utterances[:3]):
#                 print(f"Chunk {i}: Text: '{chunk['text'][:30]}...' -> Label: {chunk.get('label')}, Confidence: {chunk.get('confidence')}")
#     else:
#         print(f"\n[Note] Sample file not found at {sample_path}. Skip local test.")

# #  Run:
# #  python -m app.services.audio_transcription_sentiment_pipeline