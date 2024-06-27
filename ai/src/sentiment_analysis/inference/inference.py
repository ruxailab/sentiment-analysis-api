

# Modules
from src.video_processing.video_2_audio import video_2_audio
from src.video_processing.extract_transcript import extract_transcript
from src.video_processing.models.whisper_transcript import WhisperTranscript
from src.sentiment_analysis.models.roberta_sentiment import RobertaSentiment

import time as time
class Inference():
    def __init__(self,whisper_model_size:str="tiny",device:str="cpu"):
        # Set the device
        self.device=device

        # Load Transcription model
        self.transcript_model=WhisperTranscript(model_size=whisper_model_size,device=self.device)

        # Load Sentiment Model
        self.sentiment_model=RobertaSentiment(device=self.device)

        # Set the model to evaluation mode
        self.sentiment_model.eval()

    def infer(self,video_file:str):
        # Video2Audio
        audio=video_2_audio(video_file,True,"./temp.mp3")
        # Get the audio data as a NumPy array
        # audio = audio.to_soundarray()

        # Audio2Transcript
        # transcript=extract_transcript("./temp.mp3",True,"./temp.json")
        # transcript=extract_transcript("./temp.mp3",False)
        # # Read the transcript
        # import json
        # with open("./temp.json", "r") as file:
        #     transcript = json.load(file)
        # # transcript=extract_transcript(video_file,False)
        _,transcript=self.transcript_model('./temp.mp3')

        # Transcript2Sentiment
        for utterance in transcript:
            text = utterance["text"]
            _,_,label,prob=self.sentiment_model(text)

            # Add the sentiment to the transcript
            utterance["sentiment"]=label
            utterance["confidence"]=prob

        return transcript
    
if __name__ == "__main__":
    inference=Inference(whisper_model_size='tiny',device='cpu')
    # inference=Inference(whisper_model_size='tiny',device='cuda:0')

    # Time to test the pipeline
    start = time.time()
    transcript_sentiment=inference.infer("./data/demos/sportify/sportify_3s.mp4")
    end = time.time()
    print("Time taken to infer: ",end-start," seconds")
    print(transcript_sentiment)
    
    # Save the output to a JSON file
    import json
    with open("./temp.json", 'w') as f:
        json.dump(transcript_sentiment, f, indent=4)
        print(f"Transcription Sentiment saved to ./temp.json")

# PS D:\sentiment-analysis-api\ai> python -m src.sentiment_analysis.inference.inference