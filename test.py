from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local audio transcript sentiment analysis with speaker labels."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Local audio/video file path. Defaults to samples/sample_1.mp3.",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=0,
        help="Start time in milliseconds. Default: 0",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=None,
        help="End time in milliseconds. Default: end of file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a readable summary.",
    )
    return parser.parse_args()


def resolve_input_file(user_path: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent
    default_file = repo_root / "samples" / "sample_1.mp3"

    if user_path:
        candidate = Path(user_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()

    return default_file.resolve()


def format_chunk_line(index: int, chunk: dict) -> str:
    timestamp = chunk.get("timestamp", [])
    if len(timestamp) == 2:
        start_s = f"{float(timestamp[0]):.2f}s"
        end_s = f"{float(timestamp[1]):.2f}s"
    else:
        start_s = "?"
        end_s = "?"

    speaker = chunk.get("speaker", "UNKNOWN")

    if "error" in chunk:
        sentiment = f"ERROR: {chunk['error']}"
    else:
        label = chunk.get("label", "UNKNOWN")
        confidence = chunk.get("confidence")
        if confidence is None:
            sentiment = label
        else:
            sentiment = f"{label} ({float(confidence):.4f})"

    text = chunk.get("text", "").strip()
    return f"[{index}] {start_s} -> {end_s} | {speaker} | {sentiment}\n    {text}"


def print_summary(audio_file: Path, result: dict) -> None:
    print(f"Input file: {audio_file}")
    print(f"Audio path: {result.get('audio_path')}")
    print(f"Range: {result.get('start_time_ms')}ms -> {result.get('end_time_ms')}ms")
    print(f"Transcription: {result.get('transcription', '').strip()}")
    print("")
    print("Chunks:")

    chunks = result.get("utterances_sentiment", [])
    for index, chunk in enumerate(chunks, start=1):
        print(format_chunk_line(index, chunk))


def configure_quiet_runtime() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*std\(\): degrees of freedom is <= 0.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The input name `inputs` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*transcription using a multilingual Whisper will default to language detection.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Passing a tuple of `past_key_values` is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The attention mask is not set and cannot be inferred.*",
    )

    for logger_name in ("transformers", "pyannote", "speechbrain"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def run_pipeline_quietly(pipeline, **kwargs) -> dict:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return pipeline.process(**kwargs)


def main() -> int:
    configure_quiet_runtime()

    from app.services.local_audio_transcription_sentiment_pipeline import (
        LocalAudioTranscriptionSentimentPipeline,
    )

    args = parse_args()
    audio_file = resolve_input_file(args.file)

    if not audio_file.exists():
        print(f"Input file not found: {audio_file}")
        return 1

    if args.start_ms < 0:
        print("--start-ms cannot be negative.")
        return 1

    if args.end_ms is not None and args.end_ms < 0:
        print("--end-ms cannot be negative.")
        return 1

    if args.end_ms is not None and args.end_ms <= args.start_ms:
        print("--end-ms must be greater than --start-ms.")
        return 1

    pipeline = LocalAudioTranscriptionSentimentPipeline()
    result = run_pipeline_quietly(
        pipeline,
        url=str(audio_file),
        start_time_ms=args.start_ms,
        end_time_ms=args.end_ms,
    )

    if "error" in result:
        print(f"Processing failed: {result['error']}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_summary(audio_file, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
