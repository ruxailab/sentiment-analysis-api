# import assemblyai as aai


# aai.settings.api_key = "ec2319d292614886909ceac0036f7f30"

# transcriber = aai.Transcriber()

# # transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
# # transcript = transcriber.transcribe("./my-local-audio-file.wav")
# transcript = transcriber.transcribe("D:\sentiment-analysis-api\sportify.mp4")

# print(transcript.text)



# -_------------------------------------------------------------
# With speakers

# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai

# Replace with your API key
aai.settings.api_key = "ec2319d292614886909ceac0036f7f30"

# URL of the file to transcribe
# FILE_URL = "D:\sentiment-analysis-api\sportify.mp4"
FILE_URL = "D:\sentiment-analysis-api\sportify_full.mp4"

# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(
  FILE_URL,
  config=config
)

print(transcript)

def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{hours:02}:{minutes % 60:02}:{seconds % 60:02}"




# Write the transcript to a file with timestamps
with open("utterances_timestamps_transcript(sportify_full).txt", "w") as file:
    for utterance in transcript.utterances:
        start_time = utterance.start
        end_time = utterance.end
        speaker = utterance.speaker
        text = utterance.text
        line = f"[{start_time} - {end_time}] Speaker {speaker}: {text}\n"
        file.write(line)



# # Write the transcript to a file with timestamps
# with open("utterances_timestamps_hr_transcript(sportify_full).txt", "w") as file:
#   for utterance in transcript.utterances:
#         start_time = format_time(utterance.start)
#         end_time = format_time(utterance.end)
#         speaker = utterance.speaker
#         text = utterance.text
#         line = f"[{start_time} - {end_time}] Speaker {speaker}: {text}\n"
#         file.write(line)