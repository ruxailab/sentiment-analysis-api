"""
FFmpeg helpers for probing duration and extracting audio segments
without loading the full media into memory.
"""
import subprocess
from typing import Optional


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe command fails."""


_INVALID_DURATION_TOKENS = {"N/A", "NA", "INF", "+INF", "-INF"}


def _parse_duration_seconds(raw: str) -> Optional[float]:
    """
    Parse ffprobe duration output. Returns None when duration is unknown.
    """
    for line in (raw or "").splitlines():
        value = line.strip()
        if not value or value.upper() in _INVALID_DURATION_TOKENS:
            continue
        try:
            seconds = float(value)
        except ValueError:
            continue
        if seconds < 0 or seconds == float("inf"):
            continue
        return seconds
    return None


def _run_ffprobe(command: list[str]) -> str:
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

    return result.stdout or ""


def get_duration_ms(input_path: str) -> Optional[int]:
    """
    Return media duration in milliseconds using ffprobe.

    Browser-recorded WebM files often omit format.duration (ffprobe prints N/A).
    Falls back to the first audio stream duration. Returns None when unknown.
    """
    probes = [
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
    ]

    for command in probes:
        seconds = _parse_duration_seconds(_run_ffprobe(command))
        if seconds is not None:
            return int(seconds * 1000)

    return None


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_time_ms: float,
    end_time_ms: Optional[float] = None,
    sample_rate: int = 16000,
) -> None:
    """
    Extract [start_time_ms, end_time_ms) as mono PCM WAV via ffmpeg.

    Uses input seeking (-ss before -i) to avoid decoding the whole file.
    Output is 16 kHz mono WAV, which is efficient for Whisper.
    When end_time_ms is None, extracts until EOF.
    """
    start_s = max(start_time_ms, 0) / 1000.0

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
    ]

    if end_time_ms is not None:
        duration_s = max(end_time_ms - start_time_ms, 0) / 1000.0
        command.extend(["-t", f"{duration_s:.3f}"])

    command.extend([
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        output_path,
    ])
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
