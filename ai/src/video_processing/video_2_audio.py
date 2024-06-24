from moviepy.editor import VideoFileClip

def video_2_audio(video_file:str,save:bool=False,audio_file:str=None) -> VideoFileClip:
    try:
        # Load video file
        video = VideoFileClip(video_file)
        
        # Extract audio
        audio = video.audio
        print(f"Audio extracted successfully")

        if save:
            # Ensure the audio file ends with .mp3
            if (audio_file is None):
                audio_file = video_file.rsplit('.', 1)[0] + ".mp3"
            
            # Save audio file
            audio.write_audiofile(audio_file)
            
            print(f"Audio Saved as {audio_file}")

        return audio
           
    except Exception as e:
        print(f"Error extracting audio: {e}")

    return None

if __name__ == "__main__":
    # Replace with your input and output file paths
    input_video = "./data/demos/sportify/sportify_3s.mp4"
    output_audio = "./test.mp3"  # Output audio file format can be .mp3, .wav, etc.
    
    audio=video_2_audio(input_video,True,output_audio)
    print(audio)

# python -m src.video_processing.video_2_audio