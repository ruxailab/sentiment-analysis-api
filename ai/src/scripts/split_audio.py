
from pydub import AudioSegment, silence

def split_audio_by_silence( audio_file: str, min_silence_len=1000, silence_thresh=-50) -> list:
    audio = AudioSegment.from_file(audio_file)
    print(audio.duration_seconds)
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return chunks



audio_file="./data/demos/sportify_basma/Basma_sportify_1_Side.mp4"
audio_chunks = split_audio_by_silence(audio_file)

print(len(audio_chunks))
print(audio_chunks)

for i, chunk in enumerate(audio_chunks):
    chunk.export(f"./data/demos/sportify_basma/chunks/chunk_{i}.wav", format="wav")