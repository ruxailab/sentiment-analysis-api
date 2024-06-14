import os
import sys
import re
import argparse
import json
import time

from dotenv import load_dotenv


# Utils
from src.utils import format_time

def assembly_ai_speech2text(file_path: str,save_path:str=None) -> str:

    # Transcriber object
    transcriber = aai.Transcriber()

    # Transcribe the audio file
    # Cal time for creating the transcript
    start = time.time()
    # transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
    # transcript = transcriber.transcribe("./my-local-audio-file.wav")
    transcript = transcriber.transcribe(file_path)
    end = time.time()
    print(f"Time taken to create the transcript: {end-start} seconds")


    detailed_transcript=[] 
    # Split by . , ! or ?
    sentences = re.split(r'(?<=[.!?]) +', transcript.text)

    # Write each sentence to the file
    for sentence in sentences:
        # Write the sentence to the file json key is index
        detailed_transcript.append({'text': sentence})

    # Optionally save the detailed transcript to a JSON file
    if save_path:
        # Ensure the save_path ends with .json
        if not save_path.endswith(".json"):
            save_path = save_path.rsplit('.', 1)[0] + ".json"
        
        with open(save_path, 'w') as f:
            json.dump(detailed_transcript, f, indent=4)
        


    return detailed_transcript




def assembly_ai_speech2text_with_speaker_labels(file_path: str,save_path:str=None) -> str:
        # Transcriber object
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speaker_labels=True)

        # Transcribe the audio file
        # Cal time for creating the transcript
        start = time.time()
        # transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
        # transcript = transcriber.transcribe("./my-local-audio-file.wav")
        transcript = transcriber.transcribe(file_path,config=config)
        end = time.time()
        print(f"Time taken to create the transcript: {end-start} seconds")


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
        

         # Optionally save the detailed transcript to a JSON file
        if save_path:
             # Ensure the save_path ends with .json
            if not save_path.endswith(".json"):
                save_path = save_path.rsplit('.', 1)[0] + ".json"
        
            with open(save_path, 'w') as f:
                json.dump(detailed_transcript, f, indent=4)


        return detailed_transcript




if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    
    # Take Args from the command line
    parser = argparse.ArgumentParser(description="Transcribe audio files to text with optional speaker labels.")

    parser.add_argument('file_path', type=str, help='Path to the audio file to be transcribed.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the detailed transcript (JSON format).')
    parser.add_argument('--module', type=str, choices=['assembly_ai', 'google_cloud'], required=True, help='Transcription module to be used (assembly_ai or google_cloud).')
    parser.add_argument('--speaker_labels', action='store_true', help='Whether to use speaker labels in the transcription.')
    
    args = parser.parse_args()



    # Here you would instantiate the correct transcriber based on the module argument
    if args.module == 'assembly_ai':
        import assemblyai as aai  # Import your assembly_ai module

        # Set the API key
        assembly_ai_api_key = os.getenv('ASSEMBLY_AI_API_KEY')
        aai.settings.api_key = assembly_ai_api_key


        if args.speaker_labels:
            assembly_ai_speech2text_with_speaker_labels(args.file_path,args.save_path)
        else:
            assembly_ai_speech2text(args.file_path,args.save_path)  # Adjust this line as necessary


    elif args.module == 'google_cloud':
        # import google_cloud as gc  # Import your google_cloud module
        # transcriber = gc.Transcriber()  # Adjust this line as necessary
        pass



# PS D:\sentiment-analysis-api>  python -m src.scripts.audio_2_transcript ./data/demos/sportify/sportify_3s.mp4 --module assembly_ai
# PS D:\sentiment-analysis-api>  python -m src.scripts.audio_2_transcript ./data/demos/sportify/sportify_3s.mp4 --save_path ./data/demos/sportify/sportify_3s.json --module assembly_ai
# PS D:\sentiment-analysis-api>  python -m src.scripts.audio_2_transcript ./data/demos/sportify/sportify_3s.mp4 --module assembly_ai --speaker_labels