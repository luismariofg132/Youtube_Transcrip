"""Transcribe YouTube videos and local media with NVIDIA CUDA or an AMD/Intel CPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from runtime import ROOT, configure_cuda_libraries, configure_dependencies, resolve_model

configure_dependencies()


@dataclass(frozen=True)
class MediaSource:
    path: Path
    title: str
    source: str
    channel: str = ""


@dataclass(frozen=True)
class TranscriptionEngine:
    model: Any
    device: str
    compute_type: str
    gpu: str | None


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    return f"{total_seconds // 3600:02}:{total_seconds // 60 % 60:02}:{total_seconds % 60:02}"


def safe_filename(title: str) -> str:
    """Produce a portable filename, including Windows reserved-name handling."""
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")[:110].rstrip(" .")
    reserved_names = {"CON", "PRN", "AUX", "NUL"}
    reserved_names.update(f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10))
    if not filename or filename.upper().split(".")[0] in reserved_names:
        filename = "transcript_" + filename
    return filename


def normalize_youtube_url(source: str) -> str | None:
    """Validate video URLs and discard playlist/start-time parameters."""
    parsed = urlparse(source)
    if parsed.scheme not in ("http", "https"):
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        raise ValueError("Provide a YouTube video URL or a local media file.")
    path_parts = parsed.path.strip("/").split("/")
    video_id = ""
    if hostname == "youtu.be":
        video_id = path_parts[0]
    elif parsed.path.rstrip("/") == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif len(path_parts) == 2 and path_parts[0] in {"shorts", "live", "embed"}:
        video_id = path_parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("The URL must identify one video, not a playlist or channel.")
    return "https://www.youtube.com/watch?v=" + video_id


def download_youtube_audio(url: str) -> MediaSource:
    """Download one media stream; PyAV handles decoding without external FFmpeg."""
    from yt_dlp import YoutubeDL

    download_directory = ROOT / "downloads"
    download_directory.mkdir(exist_ok=True)
    bundled_node = ROOT / "bin" / "node.exe"
    node_path = str(bundled_node) if bundled_node.is_file() else shutil.which("node")
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(download_directory / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "overwrites": False,
        "continuedl": True,
        "windowsfilenames": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    if node_path:
        options["js_runtimes"] = {"node": {"path": node_path}}
    with YoutubeDL(options) as downloader:
        metadata = downloader.extract_info(url, download=True)
        if not metadata:
            raise RuntimeError("YouTube did not return video metadata.")
        media_path = Path(downloader.prepare_filename(metadata))
    if not media_path.is_file():
        raise FileNotFoundError(f"Downloaded media was not found: {media_path}")
    return MediaSource(
        path=media_path,
        title=metadata.get("title") or media_path.stem,
        source=url,
        channel=metadata.get("channel") or metadata.get("uploader") or "",
    )


def resolve_source(source: str) -> MediaSource:
    source = source.strip().strip('"')
    url = normalize_youtube_url(source)
    if url:
        return download_youtube_audio(url)
    media_path = Path(source).expanduser().resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file was not found: {media_path}")
    return MediaSource(path=media_path, title=media_path.stem, source=str(media_path))


def query_nvidia_gpu() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def load_engine(options: argparse.Namespace) -> TranscriptionEngine:
    import ctranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device = options.device
    cuda_available = device != "cpu" and ctranslate2.get_cuda_device_count() > 0
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
        print(f"Automatic device selection: {device}", flush=True)
    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            "No compatible NVIDIA GPU was detected. Use --device cpu on AMD/Intel CPUs."
        )
    if device == "cuda":
        configure_cuda_libraries()
    compute_type = "int8_float16" if device == "cuda" else "int8"
    gpu = query_nvidia_gpu() if device == "cuda" else None
    print(f"Device: {device} | compute type: {compute_type} | model: {options.model}", flush=True)
    if gpu:
        print(f"GPU: {gpu}", flush=True)
    started = time.perf_counter()
    model = WhisperModel(
        resolve_model(options.model),
        device=device,
        compute_type=compute_type,
        download_root=str(ROOT / "models"),
        cpu_threads=options.cpu_threads,
    )
    engine = BatchedInferencePipeline(model=model) if options.batch_size > 1 else model
    print(f"Model loaded in {time.perf_counter() - started:.1f} seconds.", flush=True)
    return TranscriptionEngine(model=engine, device=device, compute_type=compute_type, gpu=gpu)


def run_diagnostic(engine: TranscriptionEngine, batch_size: int) -> None:
    """Perform inference, so a visible GPU alone is not considered a passing test."""
    import numpy as np

    options = {"language": "es", "vad_filter": False}
    if batch_size > 1:
        options.update(batch_size=batch_size, clip_timestamps=[{"start": 0, "end": 1}])
    segments, _ = engine.model.transcribe(np.zeros(16000, dtype=np.float32), **options)
    list(segments)
    print(f"DIAGNOSTIC PASSED: inference completed on {engine.device}.", flush=True)


def transcribe_source(
    source: str, engine: TranscriptionEngine, options: argparse.Namespace
) -> dict:
    total_started = time.perf_counter()
    media = resolve_source(source)
    model_options = {
        "language": None if options.language == "auto" else options.language,
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": True,
    }
    if options.batch_size > 1:
        model_options["batch_size"] = options.batch_size
    print(f"\nTranscribing: {media.title}", flush=True)
    started = time.perf_counter()
    segments, audio_info = engine.model.transcribe(str(media.path), **model_options)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = options.output_dir / f"{safe_filename(media.title)}_{unique_id}.txt"
    partial_path = output_path.with_suffix(".txt.partial")
    segment_count, last_segment_end, next_progress = 0, 0.0, 0.0
    try:
        with partial_path.open("x", encoding="utf-8") as transcript:
            transcript.write(f"Title: {media.title}\n")
            if media.channel:
                transcript.write(f"Channel: {media.channel}\n")
            transcript.write(f"Source: {media.source}\nLanguage: {audio_info.language}\n")
            transcript.write(
                f"Audio duration: {format_timestamp(audio_info.duration)}\nModel: {options.model}\n"
            )
            transcript.write(f"Device: {engine.device} | Compute type: {engine.compute_type}\n")
            transcript.write("Automatic transcription; speech recognition errors may occur.\n\n")
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                timestamps = (
                    f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] "
                )
                transcript.write(("" if options.no_timestamps else timestamps) + text + "\n")
                transcript.flush()
                segment_count += 1
                last_segment_end = segment.end
                if segment.end >= next_progress:
                    progress = min(100.0, 100 * segment.end / max(audio_info.duration, 1))
                    print(
                        f"{format_timestamp(segment.end)} / "
                        f"{format_timestamp(audio_info.duration)} ({progress:.0f}%)",
                        flush=True,
                    )
                    next_progress = segment.end + 60
            if not segment_count:
                transcript.write("[No transcribable speech was detected.]\n")
        partial_path.rename(output_path)
    except BaseException:
        print(f"Incomplete transcription saved to: {partial_path}", file=sys.stderr)
        raise
    elapsed = time.perf_counter() - started
    report = {
        "title": media.title,
        "channel": media.channel,
        "source": media.source,
        "transcript_path": str(output_path.resolve()),
        "media_path": str(media.path),
        "device": engine.device,
        "gpu": engine.gpu,
        "model": options.model,
        "compute_type": engine.compute_type,
        "language": audio_info.language,
        "audio_duration_seconds": audio_info.duration,
        "last_segment_end_seconds": last_segment_end,
        "segment_count": segment_count,
        "transcription_seconds": round(elapsed, 2),
        "total_seconds": round(time.perf_counter() - total_started, 2),
        "speed_relative_to_audio": round(audio_info.duration / max(elapsed, 0.001), 2),
        "batch_size": options.batch_size,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"DONE: {output_path}\nTranscription time: {elapsed:.1f} seconds.", flush=True)
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    config_path = ROOT / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources", nargs="*", help="YouTube URLs or local audio/video paths; quote each source."
    )
    parser.add_argument(
        "--list", type=Path, help="UTF-8 file containing one source per line (# marks comments)."
    )
    parser.add_argument("--model", default=config.get("model", "small"))
    parser.add_argument(
        "--language", default=config.get("language", "es"), help="es, en, etc., or auto."
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default=config.get("device", "auto")
    )
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 1))
    parser.add_argument(
        "--cpu-threads", type=int, default=config.get("cpu_threads", min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "transcripts")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument(
        "--diagnose", action="store_true", help="Check libraries and run an inference test."
    )
    options = parser.parse_args(argv)
    if options.device not in {"auto", "cuda", "cpu"}:
        parser.error("device must be auto, cuda, or cpu")
    if not isinstance(options.batch_size, int) or options.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    if not isinstance(options.cpu_threads, int) or options.cpu_threads < 1:
        parser.error("--cpu-threads must be a positive integer")
    return options


def collect_sources(options: argparse.Namespace) -> list[str]:
    sources = list(options.sources)
    if options.list:
        sources.extend(
            line.strip()
            for line in options.list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not sources and not options.diagnose:
        while True:
            try:
                entry = input("\nPaste a YouTube URL or media path (blank line to start): ").strip()
            except EOFError:
                break
            if not entry:
                break
            sources.append(entry)
    return sources


def main(argv: list[str] | None = None) -> int:
    options = parse_arguments(argv)
    sources = collect_sources(options)
    if not sources and not options.diagnose:
        print("No sources were provided.")
        return 0
    engine = load_engine(options)
    if options.diagnose:
        run_diagnostic(engine, options.batch_size)
    failures = 0
    for source in sources:
        try:
            transcribe_source(source, engine, options)
        except Exception as error:
            failures += 1
            print(f"ERROR for {source}: {error}", file=sys.stderr, flush=True)
            print(
                "For YouTube download problems, update yt-dlp or provide a local file.\n"
                "For CUDA problems, follow README.md; "
                "try --model small --batch-size 1 for low VRAM.",
                file=sys.stderr,
            )
    return 1 if failures else 0


def cli() -> None:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled. Completed transcripts are preserved.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}\nSee README.md for setup and troubleshooting.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
