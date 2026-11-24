
import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yt_dlp


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

METADATA_DIR = BASE_DIR / "data" / "metadata"

AUDIO_DIR = BASE_DIR / "data" / "audio"

SAMPLE_RATE = 16000

CHANNELS = 1

AUDIO_CODEC = "pcm_s16le"


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. CUSTOM EXCEPTION
# --------------------------------------------------

class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


# --------------------------------------------------
# 4. RESULT DATA MODEL
# --------------------------------------------------

@dataclass
class AudioExtractionResult:

    video_id: str

    video_title: str

    status: str

    audio_path: Optional[str] = None

    duration: Optional[float] = None

    sample_rate: Optional[int] = None

    channels: Optional[int] = None

    error: Optional[str] = None


# --------------------------------------------------
# 5. CHECK EXTERNAL DEPENDENCIES
# --------------------------------------------------

def check_dependencies():

    required_tools = ["ffmpeg", "ffprobe"]

    for tool in required_tools:

        if shutil.which(tool) is None:

            raise AudioExtractionError(
                f"{tool} is not installed or not in PATH. "
                "Install it using: brew install ffmpeg"
            )


# --------------------------------------------------
# 6. LOAD PLAYLIST METADATA
# --------------------------------------------------

def load_playlist_metadata(playlist_id: str) -> dict:

    metadata_file = (
        METADATA_DIR / f"{playlist_id}.json"
    )

    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Playlist metadata not found: {metadata_file}"
        )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if metadata.get("playlist_id") != playlist_id:

        raise AudioExtractionError(
            "Playlist ID does not match metadata file."
        )

    if not isinstance(metadata.get("videos"), list):

        raise AudioExtractionError(
            "Invalid playlist metadata: videos must be a list."
        )

    return metadata


# --------------------------------------------------
# 7. VALIDATE AUDIO USING FFPROBE
# --------------------------------------------------

def validate_audio(audio_path: Path) -> dict:

    if not audio_path.exists():

        raise AudioExtractionError(
            "Audio file does not exist."
        )

    if audio_path.stat().st_size == 0:

        raise AudioExtractionError(
            "Audio file is empty."
        )

    command = [

        "ffprobe",

        "-v", "error",

        "-show_entries",
        "format=duration",

        "-show_entries",
        "stream=codec_name,sample_rate,channels",

        "-of", "json",

        str(audio_path)
    ]

    process = subprocess.run(

        command,

        capture_output=True,

        text=True,

        check=True
    )

    probe_data = json.loads(process.stdout)

    streams = probe_data.get("streams", [])

    audio_streams = [

        stream for stream in streams

        if stream.get("codec_name") == AUDIO_CODEC

    ]

    if not audio_streams:

        raise AudioExtractionError(
            "Expected PCM signed 16-bit audio stream not found."
        )

    stream = audio_streams[0]

    duration = float(
        probe_data["format"]["duration"]
    )

    sample_rate = int(
        stream["sample_rate"]
    )

    channels = int(
        stream["channels"]
    )

    if duration <= 0:

        raise AudioExtractionError(
            "Invalid audio duration."
        )

    if sample_rate != SAMPLE_RATE:

        raise AudioExtractionError(
            f"Unexpected sample rate: {sample_rate}"
        )

    if channels != CHANNELS:

        raise AudioExtractionError(
            f"Unexpected channel count: {channels}"
        )

    return {

        "duration": duration,

        "sample_rate": sample_rate,

        "channels": channels

    }


# --------------------------------------------------
# 8. DOWNLOAD ORIGINAL AUDIO STREAM
# --------------------------------------------------

def download_audio_stream(
    video_url: str,
    temp_dir: Path,
    video_id: str
) -> Path:

    ydl_options = {

        "format": "bestaudio/best",

        "outtmpl": str(
            temp_dir / "%(id)s.%(ext)s"
        ),

        "noplaylist": True,

        "quiet": True,

        "no_warnings": True,

        "retries": 3,

        "fragment_retries": 3,

        "continuedl": True,

    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:

        info = ydl.extract_info(
            video_url,
            download=True
        )

        if not info:

            raise AudioExtractionError(
                "yt-dlp could not extract audio."
            )

        if info.get("id") != video_id:

            raise AudioExtractionError(
                "Downloaded video ID does not match expected ID."
            )

        requested_downloads = (
            info.get("requested_downloads") or []
        )

        raw_audio_path = None

        if requested_downloads:

            raw_audio_path = requested_downloads[0].get(
                "filepath"
            )

        if not raw_audio_path:

            raw_audio_path = ydl.prepare_filename(info)

        raw_audio_path = Path(raw_audio_path)

        if not raw_audio_path.exists():

            candidates = list(
                temp_dir.glob(f"{video_id}.*")
            )

            candidates = [

                path for path in candidates

                if not path.name.endswith(".part")

            ]

            if candidates:

                raw_audio_path = candidates[0]

            else:

                raise AudioExtractionError(
                    "Downloaded audio file not found."
                )

        return raw_audio_path


# --------------------------------------------------
# 9. CONVERT AUDIO USING FFMPEG
# --------------------------------------------------

def convert_audio(
    input_path: Path,
    output_path: Path
):

    command = [

        "ffmpeg",

        "-nostdin",

        "-y",

        "-i", str(input_path),

        "-vn",

        "-ac", str(CHANNELS),

        "-ar", str(SAMPLE_RATE),

        "-c:a", AUDIO_CODEC,

        str(output_path)

    ]

    process = subprocess.run(

        command,

        capture_output=True,

        text=True
    )

    if process.returncode != 0:

        raise AudioExtractionError(
            "FFmpeg conversion failed: "
            + process.stderr[-2000:]
        )

    if not output_path.exists():

        raise AudioExtractionError(
            "FFmpeg did not generate the output file."
        )


# --------------------------------------------------
# 10. PROCESS ONE VIDEO
# --------------------------------------------------

def extract_audio_for_video(
    video: dict,
    playlist_id: str,
    force: bool = False
) -> AudioExtractionResult:

    video_id = video["video_id"]

    video_title = video.get(
        "title",
        "Unknown Video"
    )

    video_url = video.get(
        "video_url"
    )

    if not video_url:

        raise AudioExtractionError(
            f"Missing video URL for {video_id}"
        )

    output_dir = AUDIO_DIR / playlist_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / f"{video_id}.wav"
    )

    # Skip previously processed audio.

    if output_path.exists() and not force:

        try:

            audio_info = validate_audio(output_path)

            logger.info(
                "Skipping already processed video: %s",
                video_title
            )

            return AudioExtractionResult(

                video_id=video_id,

                video_title=video_title,

                status="skipped",

                audio_path=str(output_path),

                **audio_info
            )

        except Exception:

            logger.warning(
                "Existing audio is invalid. Reprocessing %s",
                video_id
            )

    logger.info(
        "Starting audio extraction: %s",
        video_title
    )

    # Temporary directory is cleaned automatically.

    with tempfile.TemporaryDirectory(
        prefix=f"{video_id}_",
        dir=output_dir
    ) as temp_directory:

        temp_dir = Path(temp_directory)

        raw_audio_path = download_audio_stream(

            video_url=video_url,

            temp_dir=temp_dir,

            video_id=video_id
        )

        temporary_output = (
            temp_dir / f"{video_id}.wav"
        )

        convert_audio(

            input_path=raw_audio_path,

            output_path=temporary_output
        )

        audio_info = validate_audio(
            temporary_output
        )

        # Atomic move after successful validation.

        os.replace(
            temporary_output,
            output_path
        )

    logger.info(
        "Audio extraction completed: %s",
        video_title
    )

    return AudioExtractionResult(

        video_id=video_id,

        video_title=video_title,

        status="success",

        audio_path=str(output_path),

        **audio_info
    )


# --------------------------------------------------
# 11. PROCESS COMPLETE PLAYLIST SEQUENTIALLY
# --------------------------------------------------

def extract_playlist_audio(
    playlist_id: str,
    force: bool = False
) -> list[AudioExtractionResult]:

    check_dependencies()

    metadata = load_playlist_metadata(
        playlist_id
    )

    videos = metadata["videos"]

    results = []

    logger.info(
        "Processing playlist: %s",
        metadata["playlist_title"]
    )

    for index, video in enumerate(
        videos,
        start=1
    ):

        logger.info(
            "Processing video %s/%s",
            index,
            len(videos)
        )

        try:

            result = extract_audio_for_video(

                video=video,

                playlist_id=playlist_id,

                force=force
            )

        except Exception as error:

            logger.exception(
                "Audio extraction failed for %s",
                video.get("video_id")
            )

            result = AudioExtractionResult(

                video_id=video.get("video_id", "unknown"),

                video_title=video.get("title", "Unknown"),

                status="failed",

                error=str(error)
            )

        results.append(result)

    return results


# --------------------------------------------------
# 12. SAVE EXTRACTION MANIFEST
# --------------------------------------------------

def save_audio_manifest(
    playlist_id: str,
    results: list[AudioExtractionResult]
):

    manifest_path = (
        METADATA_DIR /
        f"{playlist_id}_audio_manifest.json"
    )

    manifest = {

        "playlist_id": playlist_id,

        "total_processed": len(results),

        "successful": sum(
            result.status == "success"
            for result in results
        ),

        "skipped": sum(
            result.status == "skipped"
            for result in results
        ),

        "failed": sum(
            result.status == "failed"
            for result in results
        ),

        "results": [
            asdict(result)
            for result in results
        ]

    }

    temporary_path = manifest_path.with_suffix(
        ".tmp"
    )

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

    logger.info(
        "Audio manifest saved: %s",
        manifest_path
    )


# --------------------------------------------------
# 13. MAIN EXECUTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(

        description="Extract playlist audio using yt-dlp and FFmpeg"
    )

    parser.add_argument(
        "playlist_id",
        help="Playlist ID from Module 1"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract existing audio files"
    )

    args = parser.parse_args()

    try:

        results = extract_playlist_audio(

            playlist_id=args.playlist_id,

            force=args.force
        )

        save_audio_manifest(
            args.playlist_id,
            results
        )

        print("\n========== AUDIO EXTRACTION SUMMARY ==========")

        print(
            f"Total processed: {len(results)}"
        )

        print(
            f"Successful: {sum(r.status == 'success' for r in results)}"
        )

        print(
            f"Skipped: {sum(r.status == 'skipped' for r in results)}"
        )

        print(
            f"Failed: {sum(r.status == 'failed' for r in results)}"
        )

        for result in results:

            print(
                f"\n{result.video_title}"
            )

            print(
                f"Status: {result.status}"
            )

            if result.audio_path:

                print(
                    f"Audio: {result.audio_path}"
                )

            if result.error:

                print(
                    f"Error: {result.error}"
                )

    except Exception as error:

        logger.exception(
            "Playlist audio processing failed: %s",
            error
        )

        raise SystemExit(1)


if __name__ == "__main__":

    main()
