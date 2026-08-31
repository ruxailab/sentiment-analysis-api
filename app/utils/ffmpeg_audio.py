"""
FFmpeg helpers for probing duration and extracting audio segments
without loading the full media into memory.
"""
import subprocess


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe command fails."""


def get_duration_ms(input_path: str) -> int:
    """
    Return media duration in milliseconds using ffprobe.
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise FFmpegError(f"ffprobe failed: {stderr or exc}") from exc

    duration_raw = (result.stdout or "").strip()
    if not duration_raw:
        raise FFmpegError("ffprobe returned empty duration")

    return int(float(duration_raw) * 1000)


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_time_ms: float,
    end_time_ms: float,
    sample_rate: int = 16000,
) -> None:
    """
    Extract [start_time_ms, end_time_ms) as mono PCM WAV via ffmpeg.

    Uses input seeking (-ss before -i) to avoid decoding the whole file.
    Output is 16 kHz mono WAV, which is efficient for Whisper.
    """
    start_s = max(start_time_ms, 0) / 1000.0
    duration_s = max(end_time_ms - start_time_ms, 0) / 1000.0

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration_s:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise FFmpegError(f"ffmpeg failed: {stderr or exc}") from exc
