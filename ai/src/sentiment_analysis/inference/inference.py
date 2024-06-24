

# Modules
from src.video_processing.video_2_audio import video_2_audio
from src.video_processing.extract_transcript import extract_transcript
from src.sentiment_analysis.models.roberta_sentiment import RobertaSentiment

class Inference():
    def __init__(self):

        # Load Model
        self.model=RobertaSentiment()

    def infer(self,video_file:str):
        # # Video2Audio
        # audio=video_2_audio(video_file,True,"./temp.mp3")

        # # Audio2Transcript
        # transcript=extract_transcript("./temp.mp3",True,"./temp.json")
        # Read the transcript
        import json
        with open("./temp.json", "r") as file:
            transcript = json.load(file)

        # Transcript2Sentiment
        for utterance in transcript:
            text = utterance["text"]
            _,_,label,prob=self.model(text)

            # Add the sentiment to the transcript
            utterance["sentiment"]=label
            utterance["confidence"]=prob

        return transcript
    
if __name__ == "__main__":
    inference=Inference()

    transcript_sentiment=inference.infer("./data/demos/sportify/sportify_3s.mp4")
    print(transcript_sentiment)

# PS D:\sentiment-analysis-api\ai> python -m src.sentiment_analysis.inference.inference