
import argparse
import json
import logging

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
CHUNK_DIR = BASE_DIR / "data" / "chunks"

TARGET_CHUNK_SECONDS = 75
OVERLAP_SECONDS = 10


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

class ChunkingError(Exception):
    pass


# ============================================================
# LOAD TRANSCRIPT
# ============================================================

def load_transcript(
    playlist_id: str,
    video_id: str
) -> dict:

    transcript_path = (
        TRANSCRIPT_DIR
        / playlist_id
        / f"{video_id}.json"
    )

    if not transcript_path.exists():
        raise ChunkingError(
            f"Transcript not found: {transcript_path}"
        )

    try:

        with open(
            transcript_path,
            "r",
            encoding="utf-8"
        ) as file:

            transcript = json.load(file)

    except (json.JSONDecodeError, OSError) as error:

        raise ChunkingError(
            f"Could not read transcript: {error}"
        ) from error

    if not isinstance(transcript.get("segments"), list):
        raise ChunkingError(
            "Transcript does not contain a valid segments list."
        )

    return transcript


# ============================================================
# VALIDATE SEGMENTS
# ============================================================

def validate_segments(
    segments: list[dict]
) -> list[dict]:

    valid_segments = []

    for segment in segments:

        try:

            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment.get("text", "")).strip()

        except (KeyError, TypeError, ValueError):
            continue

        if not text:
            continue

        if start < 0 or end <= start:
            continue

        valid_segments.append({
            "start": start,
            "end": end,
            "text": text
        })

    valid_segments.sort(
        key=lambda item: (item["start"], item["end"])
    )

    return valid_segments


# ============================================================
# SENTENCE BOUNDARY DETECTION
# ============================================================

def is_sentence_end(text: str) -> bool:

    text = text.strip()

    return text.endswith((".", "?", "!"))


# ============================================================
# CREATE CHUNKS
# ============================================================



def create_chunks(
    segments: list[dict],
    target_seconds: int = TARGET_CHUNK_SECONDS,
    overlap_seconds: int = OVERLAP_SECONDS
) -> list[dict]:

    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive.")

    if overlap_seconds < 0 or overlap_seconds >= target_seconds:
        raise ValueError(
            "overlap_seconds must be non-negative and smaller than target_seconds."
        )

    segments = validate_segments(segments)

    if not segments:
        return []

    chunks = []
    start_index = 0
    chunk_number = 1
    total_segments = len(segments)

    while start_index < total_segments:

        chunk_start = segments[start_index]["start"]
        target_end = chunk_start + target_seconds

        # Find the first segment that reaches the target duration.
        end_index = start_index

        while (
            end_index < total_segments
            and segments[end_index]["end"] < target_end
        ):
            end_index += 1

        # Include all remaining segments if target exceeds transcript.
        if end_index >= total_segments:
            end_index = total_segments - 1

        else:
            # Search for a nearby sentence-ending boundary.
            search_start = max(start_index, end_index - 3)
            search_end = min(total_segments - 1, end_index + 3)

            candidates = [
                index
                for index in range(search_start, search_end + 1)
                if is_sentence_end(segments[index]["text"])
            ]

            if candidates:
                best_candidate = min(
                    candidates,
                    key=lambda index: abs(
                        segments[index]["end"] - target_end
                    )
                )

                if abs(
                    segments[best_candidate]["end"] - target_end
                ) <= 5:
                    end_index = best_candidate

        # IMPORTANT: Always initialize selected_segments.
        selected_segments = segments[
            start_index:end_index + 1
        ]

        chunk_end = selected_segments[-1]["end"]

        chunk_text = " ".join(
            segment["text"]
            for segment in selected_segments
        )

        chunks.append({
            "chunk_number": chunk_number,
            "start": round(chunk_start, 3),
            "end": round(chunk_end, 3),
            "duration": round(chunk_end - chunk_start, 3),
            "text": chunk_text,
            "segment_count": len(selected_segments)
        })

        chunk_number += 1

        # Stop if all transcript segments are included.
        if end_index >= total_segments - 1:
            break

        # Calculate the next chunk's approximate overlap.
        overlap_start_time = chunk_end - overlap_seconds

        next_start_index = start_index + 1

        while (
            next_start_index < total_segments
            and segments[next_start_index]["end"] <= overlap_start_time
        ):
            next_start_index += 1

        # Guarantee forward progress.
        if next_start_index <= start_index:
            next_start_index = start_index + 1

        start_index = next_start_index

    return chunks




# ============================================================
# ADD VIDEO METADATA
# ============================================================

def enrich_chunks(
    chunks: list[dict],
    transcript: dict
) -> list[dict]:

    video_id = transcript["video_id"]

    video_title = transcript.get(
        "video_title",
        "Unknown title"
    )

    video_url = transcript.get(
        "video_url",
        f"https://www.youtube.com/watch?v={video_id}"
    )

    enriched_chunks = []

    for chunk in chunks:

        start_timestamp = int(chunk["start"])

        youtube_url = (
            f"{video_url.split('&t=')[0]}"
            f"&t={start_timestamp}s"
            if "?" in video_url
            else f"{video_url}?t={start_timestamp}s"
        )

        enriched_chunks.append({
            "chunk_id": (
                f"{video_id}_{chunk['chunk_number']:03d}"
            ),
            "playlist_id": transcript["playlist_id"],
            "video_id": video_id,
            "video_title": video_title,
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": chunk["duration"],
            "text": chunk["text"],
            "segment_count": chunk["segment_count"],
            "youtube_url": youtube_url
        })

    return enriched_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(
    playlist_id: str,
    video_id: str,
    chunks: list[dict]
) -> Path:

    output_dir = CHUNK_DIR / playlist_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / f"{video_id}.json"

    data = {
        "playlist_id": playlist_id,
        "video_id": video_id,
        "total_chunks": len(chunks),
        "chunking_config": {
            "target_seconds": TARGET_CHUNK_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS
        },
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "chunks": chunks
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

        temporary_path.replace(output_path)

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Saved %s chunks: %s",
        len(chunks),
        output_path
    )

    return output_path


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    playlist_id: str,
    video_id: str
) -> Path:

    transcript = load_transcript(
        playlist_id,
        video_id
    )

    segments = validate_segments(
        transcript["segments"]
    )

    chunks = create_chunks(
        segments=segments
    )

    if not chunks:
        raise ChunkingError(
            f"No chunks generated for video {video_id}"
        )

    enriched_chunks = enrich_chunks(
        chunks,
        transcript
    )

    return save_chunks(
        playlist_id,
        video_id,
        enriched_chunks
    )


# ============================================================
# PROCESS ENTIRE PLAYLIST
# ============================================================

def process_playlist(
    playlist_id: str
) -> None:

    playlist_transcript_dir = (
        TRANSCRIPT_DIR / playlist_id
    )

    if not playlist_transcript_dir.exists():
        raise ChunkingError(
            f"Transcript directory not found: "
            f"{playlist_transcript_dir}"
        )

    transcript_files = sorted(
        playlist_transcript_dir.glob("*.json")
    )

    if not transcript_files:
        raise ChunkingError(
            "No transcript files found."
        )

    succeeded = 0
    failed = 0

    for index, transcript_path in enumerate(
        transcript_files,
        start=1
    ):

        video_id = transcript_path.stem

        logger.info(
            "Processing video %s/%s: %s",
            index,
            len(transcript_files),
            video_id
        )

        try:

            process_video(
                playlist_id,
                video_id
            )

            succeeded += 1

        except Exception:

            logger.exception(
                "Chunking failed for video %s",
                video_id
            )

            failed += 1

    logger.info(
        "\n========== CHUNKING SUMMARY =========="
    )

    logger.info(
        "Total transcripts: %s",
        len(transcript_files)
    )

    logger.info(
        "Successfully chunked: %s",
        succeeded
    )

    logger.info(
        "Failed: %s",
        failed
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Create timestamp-aware transcript chunks."
    )

    parser.add_argument(
        "playlist_id",
        help="YouTube playlist ID"
    )

    parser.add_argument(
        "video_id",
        nargs="?",
        help="Optional video ID for single-video chunking"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available transcripts"
    )

    args = parser.parse_args()

    if args.all and args.video_id:
        parser.error(
            "Do not provide video_id with --all."
        )

    if not args.all and not args.video_id:
        parser.error(
            "Provide video_id or use --all."
        )

    if args.all:

        process_playlist(args.playlist_id)

    else:

        output_path = process_video(
            args.playlist_id,
            args.video_id
        )

        print(f"\nChunks saved: {output_path}")


if __name__ == "__main__":
    main()
