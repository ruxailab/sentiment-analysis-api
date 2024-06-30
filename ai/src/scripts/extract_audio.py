from moviepy.editor import VideoFileClip

def extract_audio(video_file, audio_file,save_wav=False):
    try:
        # Load video file
        video = VideoFileClip(video_file)
        
        # Extract audio
        audio = video.audio
        
        # Save audio file
        if save_wav:
            audio_file = audio_file.replace(".mp3", ".wav")

        audio.write_audiofile(audio_file)
        
        print(f"Audio extracted successfully and saved as {audio_file}")
        
    except Exception as e:
        print(f"Error extracting audio: {e}")

if __name__ == "__main__":
    # Replace with your input and output file paths
    input_video = "./data/demos/sportify/sportify_3s.mp4"
    # output_audio = "./data/demos/sportify/sportify_full.mp3"  # Output audio file format can be .mp3, .wav, etc.
    # input_video = "./data/demos/sportify/sportify_full.mp4"
    output_audio = "./test.mp3"  # Output audio file format can be .mp3, .wav, etc.
    
    extract_audio(input_video, output_audio,save_wav=True)
