"""
Unit tests for ffmpeg audio helpers.
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.utils.ffmpeg_audio import FFmpegError, extract_audio_segment, get_duration_ms


def _completed(stdout: str) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    return result


class TestGetDurationMs:
    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_uses_format_duration(self, mock_run):
        mock_run.return_value = _completed("12.345\n")

        assert get_duration_ms("/tmp/audio.webm") == 12345
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0][4] == "format=duration"

    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_falls_back_to_stream_duration_when_format_is_na(self, mock_run):
        mock_run.side_effect = [
            _completed("N/A\n"),
            _completed("3.5\n"),
        ]

        assert get_duration_ms("/tmp/audio.webm") == 3500
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1].args[0][4] == "a:0"

    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_returns_none_when_duration_is_unknown(self, mock_run):
        mock_run.side_effect = [
            _completed("N/A"),
            _completed("N/A"),
        ]

        assert get_duration_ms("/tmp/audio.webm") is None

    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_raises_when_ffprobe_fails(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe"],
            stderr="invalid data",
        )

        with pytest.raises(FFmpegError, match="ffprobe failed"):
            get_duration_ms("/tmp/audio.webm")


class TestExtractAudioSegment:
    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_includes_duration_when_end_time_is_set(self, mock_run):
        mock_run.return_value = _completed("")

        extract_audio_segment(
            input_path="/tmp/in.webm",
            output_path="/tmp/out.wav",
            start_time_ms=1000,
            end_time_ms=4000,
        )

        command = mock_run.call_args.args[0]
        assert "-t" in command
        assert "3.000" in command

    @patch("app.utils.ffmpeg_audio.subprocess.run")
    def test_omits_duration_when_end_time_is_none(self, mock_run):
        mock_run.return_value = _completed("")

        extract_audio_segment(
            input_path="/tmp/in.webm",
            output_path="/tmp/out.wav",
            start_time_ms=0,
            end_time_ms=None,
        )

        command = mock_run.call_args.args[0]
        assert "-t" not in command
