
import argparse
import json
import logging
import os

from datetime import datetime, timezone
from pathlib import Path

from app.processing.embeddings import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    generate_embeddings,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNK_DIR = BASE_DIR / "data" / "chunks"
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"

BATCH_SIZE = 32


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# CUSTOM EXCEPTION
# --------------------------------------------------

class EmbeddingPipelineError(Exception):
    """Raised when the embedding pipeline fails."""


# --------------------------------------------------
# 1. LOAD CHUNK FILE
# --------------------------------------------------

def load_chunks(playlist_id: str, video_id: str) -> dict:
    chunk_path = (
        CHUNK_DIR
        / playlist_id
        / f"{video_id}.json"
    )

    if not chunk_path.exists():
        raise EmbeddingPipelineError(
            f"Chunk file not found: {chunk_path}"
        )

    try:
        data = json.loads(chunk_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmbeddingPipelineError(
            f"Invalid JSON in {chunk_path}"
        ) from exc

    if not isinstance(data.get("chunks"), list):
        raise EmbeddingPipelineError(
            "Chunk file must contain a chunks list."
        )

    return data


# --------------------------------------------------
# 2. VALIDATE CHUNKS
# --------------------------------------------------

def validate_chunks(chunks: list[dict]) -> list[dict]:
    valid_chunks = []

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            logger.warning("Skipping invalid chunk at index %s", index)
            continue

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            logger.warning("Skipping empty chunk at index %s", index)
            continue

        valid_chunks.append(chunk)

    return valid_chunks


# --------------------------------------------------
# 3. GENERATE EMBEDDINGS FOR ONE VIDEO
# --------------------------------------------------

def embed_video(
    playlist_id: str,
    video_id: str,
    force: bool = False,
) -> Path:

    output_dir = EMBEDDING_DIR / playlist_id
    output_path = output_dir / f"{video_id}.json"

    if output_path.exists() and not force:
        logger.info(
            "Embedding file already exists. Skipping %s",
            video_id,
        )
        return output_path

    chunk_data = load_chunks(playlist_id, video_id)

    chunks = validate_chunks(chunk_data["chunks"])

    if not chunks:
        raise EmbeddingPipelineError(
            f"No valid chunks found for video {video_id}"
        )

    logger.info(
        "Generating embeddings for %s chunks",
        len(chunks),
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = generate_embeddings(
        texts,
        batch_size=BATCH_SIZE,
    )

    if len(vectors) != len(chunks):
        raise EmbeddingPipelineError(
            "Embedding count does not match chunk count."
        )

    records = []

    for chunk, vector in zip(chunks, vectors):

        if len(vector) != EMBEDDING_DIMENSION:
            raise EmbeddingPipelineError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"received {len(vector)}."
            )

        records.append({
            **chunk,
            "embedding": vector,
        })

    output_data = {
        "playlist_id": playlist_id,
        "video_id": video_id,
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "total_chunks": len(records),
        "embedded_at": datetime.now(timezone.utc).isoformat(),
        "chunks": records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(".tmp")

    try:
        temp_path.write_text(
            json.dumps(output_data, ensure_ascii=False),
            encoding="utf-8",
        )

        os.replace(temp_path, output_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    logger.info(
        "Saved %s embeddings to %s",
        len(records),
        output_path,
    )

    return output_path


# --------------------------------------------------
# 4. PROCESS ALL VIDEOS IN A PLAYLIST
# --------------------------------------------------

def embed_playlist(
    playlist_id: str,
    force: bool = False,
) -> dict:

    playlist_chunk_dir = CHUNK_DIR / playlist_id

    if not playlist_chunk_dir.exists():
        raise EmbeddingPipelineError(
            f"Chunk directory not found: {playlist_chunk_dir}"
        )

    chunk_files = sorted(
        playlist_chunk_dir.glob("*.json")
    )

    if not chunk_files:
        raise EmbeddingPipelineError(
            "No chunk JSON files found."
        )

    manifest = {
        "playlist_id": playlist_id,
        "total_videos": len(chunk_files),
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    for chunk_file in chunk_files:

        video_id = chunk_file.stem

        output_path = (
            EMBEDDING_DIR
            / playlist_id
            / f"{video_id}.json"
        )

        if output_path.exists() and not force:
            logger.info("Skipping existing video: %s", video_id)
            manifest["skipped"].append(video_id)
            continue

        try:
            embed_video(
                playlist_id=playlist_id,
                video_id=video_id,
                force=force,
            )

            manifest["successful"].append(video_id)

        except Exception as exc:
            logger.exception(
                "Embedding failed for video %s",
                video_id,
            )

            manifest["failed"].append({
                "video_id": video_id,
                "error": str(exc),
            })

    manifest_path = (
        EMBEDDING_DIR
        / f"{playlist_id}_embedding_manifest.json"
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    logger.info("Embedding pipeline completed.")
    logger.info("Successful: %s", len(manifest["successful"]))
    logger.info("Skipped: %s", len(manifest["skipped"]))
    logger.info("Failed: %s", len(manifest["failed"]))

    return manifest


# --------------------------------------------------
# 5. COMMAND LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate MiniLM embeddings for YouTube chunks."
    )

    parser.add_argument("playlist_id")

    parser.add_argument(
        "video_id",
        nargs="?",
        help="Process one video instead of the entire playlist.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every video with chunk files.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing embeddings.",
    )

    args = parser.parse_args()

    if args.video_id and args.all:
        parser.error("Use either video_id or --all, not both.")

    try:
        if args.video_id:
            embed_video(
                args.playlist_id,
                args.video_id,
                force=args.force,
            )

        elif args.all:
            embed_playlist(
                args.playlist_id,
                force=args.force,
            )

        else:
            parser.error("Provide a video_id or use --all.")

    except EmbeddingPipelineError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
