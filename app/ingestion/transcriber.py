
import argparse
import json
import logging
import os
import subprocess
import tempfile
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

METADATA_DIR = BASE_DIR / "data" / "metadata"
AUDIO_DIR = BASE_DIR / "data" / "audio"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"

TRANSCRIPTION_MANIFEST_DIR = METADATA_DIR

MODEL_NAME = "whisper-large-v3-turbo"

MAX_UPLOAD_BYTES = 24 * 1024 * 1024

CHUNK_SECONDS = 300
OVERLAP_SECONDS = 5

SAMPLE_RATE = 16000
CHANNELS = 1

MAX_RETRIES = 3

load_dotenv(BASE_DIR / ".env")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class TranscriptionError(Exception):
    pass


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class AudioPart:
    path: Path
    offset: float
    core_start: float
    core_end: float


# ============================================================
# METADATA FUNCTIONS
# ============================================================

def load_playlist_metadata(playlist_id: str) -> dict:

    metadata_path = METADATA_DIR / f"{playlist_id}.json"

    if not metadata_path.exists():
        raise TranscriptionError(
            f"Playlist metadata not found: {metadata_path}"
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        raise TranscriptionError(
            f"Unable to read playlist metadata: {error}"
        ) from error


def get_video_metadata(
    playlist_id: str,
    video_id: str
) -> dict:

    metadata = load_playlist_metadata(playlist_id)

    for video in metadata.get("videos", []):

        if video.get("video_id") == video_id:
            return video

    raise TranscriptionError(
        f"Video {video_id} not found in playlist metadata."
    )


# ============================================================
# AUDIO UTILITIES
# ============================================================

def check_ffmpeg() -> None:

    for executable in ("ffmpeg", "ffprobe"):

        try:
            subprocess.run(
                [executable, "-version"],
                capture_output=True,
                text=True,
                check=True
            )

        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise TranscriptionError(
                f"{executable} is not installed or not working."
            ) from error


def get_audio_duration(audio_path: Path) -> float:

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    try:
        duration = float(result.stdout.strip())

    except ValueError as error:
        raise TranscriptionError(
            f"Could not determine audio duration: {audio_path}"
        ) from error

    if duration <= 0:
        raise TranscriptionError(
            f"Invalid audio duration: {duration}"
        )

    return duration


def convert_to_flac(
    input_path: Path,
    output_path: Path
) -> Path:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "flac",
        str(output_path)
    ]

    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as error:
        raise TranscriptionError(
            f"FLAC conversion failed: {error.stderr}"
        ) from error

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TranscriptionError(
            "FLAC conversion produced an empty file."
        )

    return output_path


# ============================================================
# AUDIO SPLITTING WITH OVERLAP
# ============================================================

def split_flac(
    audio_path: Path,
    output_dir: Path,
    total_duration: float,
    chunk_seconds: int = CHUNK_SECONDS,
    overlap_seconds: int = OVERLAP_SECONDS
) -> list[AudioPart]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")

    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError(
            "overlap_seconds must be non-negative and smaller than chunk_seconds."
        )

    parts = []

    core_start = 0.0
    part_number = 1

    while core_start < total_duration:

        core_end = min(
            core_start + chunk_seconds,
            total_duration
        )

        extraction_start = max(
            0.0,
            core_start - overlap_seconds
        )

        extraction_end = min(
            total_duration,
            core_end + overlap_seconds
        )

        extraction_duration = (
            extraction_end - extraction_start
        )

        output_path = (
            output_dir / f"part_{part_number:03d}.flac"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(extraction_start),
            "-i", str(audio_path),
            "-t", str(extraction_duration),
            "-vn",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-c:a", "flac",
            str(output_path)
        ]

        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

        except subprocess.CalledProcessError as error:
            raise TranscriptionError(
                f"Failed to create audio part {part_number}: "
                f"{error.stderr}"
            ) from error

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TranscriptionError(
                f"Audio part {part_number} is empty."
            )

        parts.append(
            AudioPart(
                path=output_path,
                offset=extraction_start,
                core_start=core_start,
                core_end=core_end
            )
        )

        logger.info(
            "Created part %s | Extraction: %.2f-%.2f | Core: %.2f-%.2f",
            part_number,
            extraction_start,
            extraction_end,
            core_start,
            core_end
        )

        core_start = core_end
        part_number += 1

    return parts


# ============================================================
# GROQ TRANSCRIPTION
# ============================================================

def get_field(
    obj: Any,
    field: str,
    default=None
):
    """Read a field from either a dictionary or SDK object."""

    if isinstance(obj, dict):
        return obj.get(field, default)

    return getattr(obj, field, default)


def transcribe_with_groq(audio_path: Path) -> dict:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise TranscriptionError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    file_size = audio_path.stat().st_size

    if file_size > MAX_UPLOAD_BYTES:
        raise TranscriptionError(
            f"Audio part exceeds configured upload limit: "
            f"{file_size / (1024 * 1024):.2f} MB"
        )

    client = Groq(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                "Sending %s to Groq | Attempt %s/%s",
                audio_path.name,
                attempt,
                MAX_RETRIES
            )

            with open(audio_path, "rb") as audio_file:

                response = client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file),
                    model=MODEL_NAME,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    temperature=0.0
                )

            segments = []

            for index, segment in enumerate(
                get_field(response, "segments", []) or []
            ):

                start = get_field(segment, "start")
                end = get_field(segment, "end")
                text = get_field(segment, "text", "")

                if start is None or end is None:
                    continue

                segments.append({
                    "id": index,
                    "start": float(start),
                    "end": float(end),
                    "text": str(text).strip()
                })

            return {
                "model": MODEL_NAME,
                "language": get_field(response, "language"),
                "duration": get_field(response, "duration"),
                "text": get_field(response, "text", ""),
                "segments": segments
            }

        except Exception as error:

            logger.warning(
                "Groq transcription failed: %s",
                error
            )

            if attempt == MAX_RETRIES:
                raise TranscriptionError(
                    f"Groq transcription failed after "
                    f"{MAX_RETRIES} attempts: {error}"
                ) from error

            time.sleep(2 * attempt)

    raise TranscriptionError(
        "Unexpected transcription failure."
    )


# ============================================================
# TRANSCRIBE ALL AUDIO PARTS
# ============================================================

def transcribe_audio_parts(
    parts: list[AudioPart]
) -> dict:

    all_segments = []

    detected_language = None

    for index, part in enumerate(parts, start=1):

        logger.info(
            "Transcribing part %s/%s",
            index,
            len(parts)
        )

        result = transcribe_with_groq(part.path)

        if detected_language is None:
            detected_language = result.get("language")

        for segment in result.get("segments", []):

            local_start = float(segment["start"])
            local_end = float(segment["end"])

            absolute_start = local_start + part.offset
            absolute_end = local_end + part.offset

            midpoint = (
                absolute_start + absolute_end
            ) / 2

            # Keep a segment only in the core window
            # assigned to this part.
            belongs_to_core = (
                part.core_start <= midpoint < part.core_end
            )

            if not belongs_to_core:
                continue

            all_segments.append({
                "start": round(absolute_start, 3),
                "end": round(absolute_end, 3),
                "text": segment["text"].strip()
            })

    # Maintain chronological order.
    all_segments.sort(
        key=lambda segment: (
            segment["start"],
            segment["end"]
        )
    )

    # Assign clean sequential IDs.
    for index, segment in enumerate(all_segments):
        segment["id"] = index

    full_text = " ".join(
        segment["text"]
        for segment in all_segments
        if segment["text"]
    )

    return {
        "model": MODEL_NAME,
        "language": detected_language,
        "segments": all_segments,
        "text": full_text
    }


# ============================================================
# SAVE TRANSCRIPT
# ============================================================

def save_transcript(
    playlist_id: str,
    video_id: str,
    video_metadata: dict,
    audio_path: Path,
    duration: float,
    transcript: dict
) -> Path:

    output_dir = TRANSCRIPT_DIR / playlist_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / f"{video_id}.json"

    video_url = video_metadata.get("url") or (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    data = {
        "playlist_id": playlist_id,
        "video_id": video_id,
        "video_title": video_metadata.get(
            "title",
            "Unknown title"
        ),
        "video_url": video_url,
        "audio_path": str(audio_path.relative_to(BASE_DIR)),
        "provider": "Groq",
        "model": transcript["model"],
        "language": transcript.get("language"),
        "duration": duration,
        "full_text": transcript["text"],
        "segments": transcript["segments"],
        "transcribed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    temporary_path = output_path.with_suffix(".tmp")

    try:

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        os.replace(
            temporary_path,
            output_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Transcript saved: %s",
        output_path
    )

    return output_path


# ============================================================
# TRANSCRIBE ONE VIDEO
# ============================================================

def transcribe_one_video(
    playlist_id: str,
    video_id: str,
    force: bool = False
) -> Path:

    video_metadata = get_video_metadata(
        playlist_id,
        video_id
    )

    audio_path = (
        AUDIO_DIR / playlist_id / f"{video_id}.wav"
    )

    transcript_path = (
        TRANSCRIPT_DIR / playlist_id / f"{video_id}.json"
    )

    if not audio_path.exists():
        raise TranscriptionError(
            f"Audio file not found: {audio_path}"
        )

    if (
        transcript_path.exists()
        and not force
        and is_valid_transcript(
            transcript_path,
            playlist_id,
            video_id
        )
    ):

        logger.info(
            "Transcript already exists. Skipping %s",
            video_id
        )

        return transcript_path

    check_ffmpeg()

    duration = get_audio_duration(audio_path)

    logger.info(
        "Video: %s",
        video_metadata.get("title", video_id)
    )

    logger.info(
        "Audio duration: %.2f seconds",
        duration
    )

    # Temporary files are automatically removed.
    with tempfile.TemporaryDirectory(
        prefix=f"transcription_{video_id}_"
    ) as temporary_directory:

        temporary_dir = Path(temporary_directory)

        flac_path = (
            temporary_dir / f"{video_id}.flac"
        )

        chunks_dir = (
            temporary_dir / "chunks"
        )

        # Convert WAV to FLAC.
        convert_to_flac(
            audio_path,
            flac_path
        )

        # Split into overlapping parts.
        parts = split_flac(
            audio_path=flac_path,
            output_dir=chunks_dir,
            total_duration=duration,
            chunk_seconds=CHUNK_SECONDS,
            overlap_seconds=OVERLAP_SECONDS
        )

        logger.info(
            "Total audio parts: %s",
            len(parts)
        )

        # Transcribe all parts and combine timestamps.
        transcript = transcribe_audio_parts(parts)

        if not transcript["segments"]:
            raise TranscriptionError(
                f"No transcript segments generated for {video_id}"
            )

        # Save the final transcript.
        output_path = save_transcript(
            playlist_id=playlist_id,
            video_id=video_id,
            video_metadata=video_metadata,
            audio_path=audio_path,
            duration=duration,
            transcript=transcript
        )

    return output_path


# ============================================================
# TRANSCRIPTION MANIFEST
# ============================================================

def is_valid_transcript(
    transcript_path: Path,
    playlist_id: str,
    video_id: str
) -> bool:

    if not transcript_path.exists():
        return False

    try:

        with open(
            transcript_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return (
            data.get("playlist_id") == playlist_id
            and data.get("video_id") == video_id
            and isinstance(data.get("segments"), list)
            and len(data["segments"]) > 0
        )

    except (json.JSONDecodeError, OSError):
        return False


def save_transcription_manifest(
    playlist_id: str,
    manifest: dict
) -> Path:

    manifest_path = (
        TRANSCRIPTION_MANIFEST_DIR
        / f"{playlist_id}_transcription_manifest.json"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    statuses = [
        item["status"]
        for item in manifest["videos"].values()
    ]

    manifest["completed_count"] = statuses.count("completed")
    manifest["skipped_count"] = statuses.count("skipped")
    manifest["failed_count"] = statuses.count("failed")

    temporary_path = manifest_path.with_suffix(".tmp")

    try:

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4,
                ensure_ascii=False
            )

        os.replace(
            temporary_path,
            manifest_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

    return manifest_path


# ============================================================
# TRANSCRIBE ENTIRE PLAYLIST
# ============================================================

def transcribe_playlist(
    playlist_id: str,
    force: bool = False
) -> dict:

    metadata = load_playlist_metadata(playlist_id)

    videos = metadata.get("videos", [])

    if not videos:
        raise TranscriptionError(
            "No videos found in playlist metadata."
        )

    manifest_path = (
        TRANSCRIPTION_MANIFEST_DIR
        / f"{playlist_id}_transcription_manifest.json"
    )

    # Load previous progress.
    if manifest_path.exists():

        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as file:

                manifest = json.load(file)

        except (json.JSONDecodeError, OSError):

            logger.warning(
                "Previous manifest is invalid. Starting fresh."
            )

            manifest = {}

    else:
        manifest = {}

    if manifest.get("playlist_id") != playlist_id:
        manifest = {}

    manifest["playlist_id"] = playlist_id
    manifest["total_videos"] = len(videos)

    if not isinstance(manifest.get("videos"), dict):
        manifest["videos"] = {}

    total_videos = len(videos)

    for index, video in enumerate(videos, start=1):

        video_id = video.get("video_id")

        if not video_id:

            logger.warning(
                "Skipping entry %s: missing video ID.",
                index
            )

            continue

        title = video.get(
            "title",
            "Unknown title"
        )

        transcript_path = (
            TRANSCRIPT_DIR / playlist_id / f"{video_id}.json"
        )

        audio_path = (
            AUDIO_DIR / playlist_id / f"{video_id}.wav"
        )

        logger.info(
            "\n========== VIDEO %s/%s ==========",
            index,
            total_videos
        )

        logger.info("Title: %s", title)
        logger.info("Video ID: %s", video_id)

        # Skip videos with valid existing transcripts.
        if (
            not force
            and is_valid_transcript(
                transcript_path,
                playlist_id,
                video_id
            )
        ):

            logger.info(
                "Valid transcript exists. Skipping."
            )

            manifest["videos"][video_id] = {
                "title": title,
                "status": "skipped",
                "transcript_path": str(
                    transcript_path.relative_to(BASE_DIR)
                )
            }

            save_transcription_manifest(
                playlist_id,
                manifest
            )

            continue

        # Check whether the WAV exists.
        if not audio_path.exists():

            error_message = (
                f"Audio file missing: {audio_path}"
            )

            logger.error(error_message)

            manifest["videos"][video_id] = {
                "title": title,
                "status": "failed",
                "error": error_message
            }

            save_transcription_manifest(
                playlist_id,
                manifest
            )

            continue

        try:

            output_path = transcribe_one_video(
                playlist_id=playlist_id,
                video_id=video_id,
                force=True
            )

            manifest["videos"][video_id] = {
                "title": title,
                "status": "completed",
                "transcript_path": str(
                    output_path.relative_to(BASE_DIR)
                )
            }

            logger.info(
                "Successfully transcribed: %s",
                title
            )

        except Exception as error:

            logger.exception(
                "Failed to transcribe: %s",
                title
            )

            manifest["videos"][video_id] = {
                "title": title,
                "status": "failed",
                "error": str(error)
            }

        # Save progress after every video.
        save_transcription_manifest(
            playlist_id,
            manifest
        )

    # Final manifest update.
    save_transcription_manifest(
        playlist_id,
        manifest
    )

    logger.info("\n========== PLAYLIST SUMMARY ==========")

    logger.info(
        "Total videos: %s",
        manifest["total_videos"]
    )

    logger.info(
        "Completed: %s",
        manifest["completed_count"]
    )

    logger.info(
        "Skipped: %s",
        manifest["skipped_count"]
    )

    logger.info(
        "Failed: %s",
        manifest["failed_count"]
    )

    logger.info(
        "Manifest: %s",
        manifest_path
    )

    return manifest


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe YouTube playlist videos "
            "using Groq Whisper."
        )
    )

    parser.add_argument(
        "playlist_id",
        help="YouTube playlist ID"
    )

    parser.add_argument(
        "video_id",
        nargs="?",
        help="Optional video ID for single-video transcription"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_videos",
        help="Transcribe all videos in the playlist"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe existing transcripts"
    )

    args = parser.parse_args()

    if args.all_videos and args.video_id:
        parser.error(
            "Do not provide a video ID when using --all."
        )

    if not args.all_videos and not args.video_id:
        parser.error(
            "Provide a video ID or use --all."
        )

    try:

        if args.all_videos:

            transcribe_playlist(
                playlist_id=args.playlist_id,
                force=args.force
            )

        else:

            output_path = transcribe_one_video(
                playlist_id=args.playlist_id,
                video_id=args.video_id,
                force=args.force
            )

            print(
                f"\nTranscript saved: {output_path}"
            )

    except Exception as error:

        logger.exception(
            "Transcription failed: %s",
            error
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
