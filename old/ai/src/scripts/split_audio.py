
# # from pydub import AudioSegment, silence

# # def split_audio_by_silence( audio_file: str, min_silence_len=1000, silence_thresh=-50) -> list:
# #     audio = AudioSegment.from_file(audio_file)
# #     print(audio.duration_seconds)
# #     chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
# #     return chunks



# # audio_file="./data/demos/sportify_basma/Basma_sportify_1_Side.mp4"
# # audio_chunks = split_audio_by_silence(audio_file)

# # print(len(audio_chunks))
# # print(audio_chunks)

# # for i, chunk in enumerate(audio_chunks):
# #     chunk.export(f"./data/demos/sportify_basma/chunks/chunk_{i}.wav", format="wav")


# from pydub import AudioSegment, silence
# import numpy as np

# def get_audio_statistics(audio, chunk_size=1000):
#     # Break audio into chunks and compute their loudness
#     chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
#     loudness = [chunk.dBFS for chunk in chunks]
    
#     # Compute the mean and standard deviation of loudness
#     loudness_mean = np.mean(loudness)
#     loudness_std = np.std(loudness)
    
#     # Estimate silence threshold
#     silence_thresh = loudness_mean - 2 * loudness_std
    
#     return silence_thresh, loudness_mean, loudness_std

# def split_audio_on_silence(audio, silence_thresh, min_silence_len_factor=1.5):
#     # Estimate min silence length as a multiple of average chunk duration
#     chunk_durations = [chunk.duration_seconds for chunk in silence.detect_silence(audio, silence_thresh=silence_thresh)]
#     if chunk_durations:
#         min_silence_len = np.mean(chunk_durations) * min_silence_len_factor * 1000
#     else:
#         min_silence_len = 1000  # Fallback to 1 second if no silence detected

#     print(min_silence_len)
#     print(silence_thresh)
    
#     chunks = silence.split_on_silence(audio, 
#                                       min_silence_len=int(min_silence_len), 
#                                       silence_thresh=silence_thresh)
#     return chunks


# audio_file="./data/demos/sportify_basma/Basma_sportify_1_Side.mp4"

# # Load the audio file
# audio = AudioSegment.from_file(audio_file)

# # Get statistics from the audio
# silence_thresh, loudness_mean, loudness_std = get_audio_statistics(audio)

# # Split the audio based on dynamically determined parameters
# chunks = split_audio_on_silence(audio, silence_thresh)

# print(len(chunks))
# # # Process each chunk as needed
# # for i, chunk in enumerate(chunks):
# #     chunk.export(f"chunk{i}.mp3", format="mp3")