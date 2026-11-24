
import argparse
import json
import logging
import os

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import yt_dlp


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

METADATA_DIR = BASE_DIR / "data" / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. CUSTOM EXCEPTIONS
# --------------------------------------------------

class PlaylistExtractionError(Exception):
    """Raised when playlist extraction fails."""


# --------------------------------------------------
# 4. DATA MODELS
# --------------------------------------------------

@dataclass
class PlaylistVideo:

    video_id: str
    title: str
    video_url: str
    playlist_position: int
    duration: Optional[float] = None
    availability: str = "available"


@dataclass
class PlaylistMetadata:

    playlist_id: str
    playlist_title: str
    playlist_url: str

    total_videos: int
    available_videos: int

    videos: list[PlaylistVideo]

    extraction_issues: list[dict]

    extracted_at: str


# --------------------------------------------------
# 5. EXTRACT PLAYLIST ID
# --------------------------------------------------

def extract_playlist_id(playlist_url: str) -> str:

    parsed_url = urlparse(playlist_url)

    query_params = parse_qs(parsed_url.query)

    playlist_ids = query_params.get("list")

    if not playlist_ids or not playlist_ids[0].strip():

        raise ValueError(
            "Invalid YouTube playlist URL. "
            "Could not find playlist ID."
        )

    return playlist_ids[0].strip()


# --------------------------------------------------
# 6. NORMALIZE VIDEO INFORMATION
# --------------------------------------------------

def normalize_video(
    entry: dict,
    playlist_position: int
) -> PlaylistVideo:

    video_id = entry.get("id")

    if not video_id:

        raise ValueError(
            "Video ID is missing."
        )

    title = entry.get("title")

    availability = entry.get("availability") or "available"

    # Some unavailable entries may not have titles.
    if not title:

        title = f"Unavailable video ({video_id})"

    video_url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    duration = entry.get("duration")

    return PlaylistVideo(

        video_id=video_id,

        title=title,

        video_url=video_url,

        playlist_position=playlist_position,

        duration=duration,

        availability=availability,
    )


# --------------------------------------------------
# 7. EXTRACT PLAYLIST INFORMATION
# --------------------------------------------------

def extract_playlist(
    playlist_url: str
) -> PlaylistMetadata:

    playlist_id = extract_playlist_id(playlist_url)

    logger.info(
        "Starting playlist extraction: %s",
        playlist_id
    )

    ydl_options = {

        "extract_flat": "in_playlist",

        "ignoreerrors": True,

        "skip_download": True,

        "quiet": True,

        "no_warnings": True,

        "noplaylist": False,
    }

    videos = []

    extraction_issues = []

    seen_video_ids = set()

    try:

        with yt_dlp.YoutubeDL(ydl_options) as ydl:

            info = ydl.extract_info(
                playlist_url,
                download=False
            )

            if not info:

                raise PlaylistExtractionError(
                    "No playlist information returned."
                )

            playlist_title = (
                info.get("title") or "Untitled Playlist"
            )

            entries = info.get("entries") or []

            for position, entry in enumerate(
                entries,
                start=1
            ):

                if entry is None:

                    extraction_issues.append({

                        "playlist_position": position,

                        "reason": "Unavailable or unextractable entry"

                    })

                    logger.warning(
                        "Skipping unavailable entry at position %s",
                        position
                    )

                    continue

                video_id = entry.get("id")

                if not video_id:

                    extraction_issues.append({

                        "playlist_position": position,

                        "reason": "Missing video ID"

                    })

                    continue

                if video_id in seen_video_ids:

                    extraction_issues.append({

                        "playlist_position": position,

                        "video_id": video_id,

                        "reason": "Duplicate video"

                    })

                    logger.warning(
                        "Duplicate video skipped: %s",
                        video_id
                    )

                    continue

                try:

                    video = normalize_video(
                        entry,
                        position
                    )

                    videos.append(video)

                    seen_video_ids.add(video_id)

                except ValueError as error:

                    extraction_issues.append({

                        "playlist_position": position,

                        "video_id": video_id,

                        "reason": str(error)

                    })

        if not videos:

            raise PlaylistExtractionError(
                "No usable videos found in playlist. "
                "Check whether the playlist is empty, "
                "private, deleted or inaccessible."
            )

        result = PlaylistMetadata(

            playlist_id=playlist_id,

            playlist_title=playlist_title,

            playlist_url=playlist_url,

            total_videos=len(entries),

            available_videos=len(videos),

            videos=videos,

            extraction_issues=extraction_issues,

            extracted_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        logger.info(
            "Extraction completed. Found %s usable videos.",
            len(videos)
        )

        return result

    except PlaylistExtractionError:

        raise

    except Exception as error:

        raise PlaylistExtractionError(
            f"Playlist extraction failed: {error}"
        ) from error


# --------------------------------------------------
# 8. SAVE PLAYLIST METADATA
# --------------------------------------------------

def save_playlist_metadata(
    metadata: PlaylistMetadata
) -> Path:

    output_file = (
        METADATA_DIR /
        f"{metadata.playlist_id}.json"
    )

    temporary_file = output_file.with_suffix(".tmp")

    try:

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                asdict(metadata),
                file,
                indent=4,
                ensure_ascii=False
            )

        os.replace(
            temporary_file,
            output_file
        )

        logger.info(
            "Metadata saved: %s",
            output_file
        )

        return output_file

    finally:

        if temporary_file.exists():

            temporary_file.unlink()


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Extract metadata from a YouTube playlist"
        )
    )

    parser.add_argument(

        "playlist_url",

        help="YouTube playlist URL"
    )

    args = parser.parse_args()

    try:

        metadata = extract_playlist(
            args.playlist_url
        )

        output_file = save_playlist_metadata(
            metadata
        )

        print("\n========== PLAYLIST SUMMARY ==========")

        print(
            f"Playlist: {metadata.playlist_title}"
        )

        print(
            f"Playlist ID: {metadata.playlist_id}"
        )

        print(
            f"Total entries: {metadata.total_videos}"
        )

        print(
            f"Usable videos: {metadata.available_videos}"
        )

        print(
            f"Extraction issues: {len(metadata.extraction_issues)}"
        )

        print(
            f"Saved to: {output_file}"
        )

        print("\n========== VIDEO LIST ==========")

        for video in metadata.videos:

            print(
                f"{video.playlist_position}. "
                f"{video.title}"
            )

    except (
        PlaylistExtractionError,
        ValueError
    ) as error:

        logger.error("%s", error)

        raise SystemExit(1)


if __name__ == "__main__":

    main()
