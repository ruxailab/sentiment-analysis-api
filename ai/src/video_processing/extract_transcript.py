from dotenv import load_dotenv

import assemblyai as aai  # Import your assembly_ai module
import os
import time
import json

# Utils
from src.utils import format_time


# TODO: Fix to take audio itself as input not the audio file path
def extract_transcript(audio_file:str,save:bool=False,transcript_file:str=None) -> str:
        try:

            # Load the .env file
            load_dotenv()

            # Set the API key
            assembly_ai_api_key = os.getenv('ASSEMBLY_AI_API_KEY')
            aai.settings.api_key = assembly_ai_api_key

            # Transcriber object
            transcriber = aai.Transcriber()
            # configure the transcription to include speaker labels :D
            config = aai.TranscriptionConfig(speaker_labels=True)

            start = time.time()
            # Transcribe the audio file
            transcript=transcriber.transcribe(audio_file,config=config)
            # transcript = transcriber.transcribe(file_path)
            end = time.time()


            detailed_transcript=[] 
            for utterance in transcript.utterances:
                    start_time = format_time(utterance.start)
                    end_time = format_time(utterance.end)
                    speaker = utterance.speaker
                    text = utterance.text
                    
                    detailed_transcript.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'speaker': speaker,
                        'text': text,
                    })

            print("Transcript extracted successfully")
            print(f"Time taken to create the transcript: {end-start} seconds")
            
            if save:
                    # Ensure the save_path ends with .json
                    if (transcript_file is None):
                            transcript_file = audio_file.rsplit('.', 1)[0] + ".json"
                    
                    with open(transcript_file, 'w') as f:
                            json.dump(detailed_transcript, f, indent=4)
                            print(f"Transcript saved to {transcript_file}")

            return detailed_transcript
        except Exception as e:
            print(f"Error extracting transcript: {e}")

        return None

if __name__ == "__main__":
    # Replace with your input and output file paths
    input_audio = "./temp.mp3"
    output_transcript = "./temp.json"  # Output transcript file format can be .json, .txt, etc.

    transcript=transcript=extract_transcript(input_audio,True,output_transcript)
    print(transcript)
# python -m src.video_processing.extract_transcript