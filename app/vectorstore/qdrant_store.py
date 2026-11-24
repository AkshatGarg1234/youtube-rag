
import argparse
import json
import logging
import math
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.processing.embeddings import EMBEDDING_DIMENSION


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "youtube_playlist_chunks"
)

VECTOR_SIZE = EMBEDDING_DIMENSION

EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"

BATCH_SIZE = 100

PLAYLIST_ID = "PLot-Xpze53leF0FeHz2X0aG3zd0mr1AW_"


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# CLIENT CONNECTION
# --------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant Cloud client."""

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL is missing from .env")

    if not api_key:
        raise ValueError("QDRANT_API_KEY is missing from .env")

    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=30
    )

    client.get_collections()

    logger.info("Connected successfully to Qdrant Cloud.")

    return client


# --------------------------------------------------
# COLLECTION MANAGEMENT
# --------------------------------------------------

def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME
) -> None:
    """Create collection if it does not exist."""

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)

        existing_vectors = info.config.params.vectors

        if isinstance(existing_vectors, dict):
            raise ValueError(
                "Collection uses named vectors. "
                "Please use a separate collection."
            )

        if existing_vectors.size != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions, "
                f"found {existing_vectors.size}."
            )

        if existing_vectors.distance != Distance.COSINE:
            raise ValueError(
                "Collection distance must be COSINE."
            )

        logger.info(
            "Collection '%s' already exists with correct configuration.",
            collection_name
        )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    logger.info(
        "Collection '%s' created successfully.",
        collection_name
    )


# --------------------------------------------------
# COLLECTION INFORMATION
# --------------------------------------------------

def get_collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME
) -> dict:
    """Return basic collection information."""

    info = client.get_collection(collection_name)

    return {
        "collection_name": collection_name,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
        "vector_size": VECTOR_SIZE,
        "distance": "COSINE"
    }


# --------------------------------------------------
# DETERMINISTIC POINT ID
# --------------------------------------------------

def generate_point_id(
    playlist_id: str,
    video_id: str,
    chunk_id: str
) -> str:
    """Generate a stable UUID for every chunk."""

    unique_key = f"{playlist_id}:{video_id}:{chunk_id}"

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, unique_key)
    )


# --------------------------------------------------
# LOAD EMBEDDING FILES
# --------------------------------------------------

def load_embedding_files(playlist_id: str) -> list[Path]:
    """Find all per-video embedding JSON files."""

    directory = EMBEDDING_DIR / playlist_id

    if not directory.exists():
        raise FileNotFoundError(
            f"Embedding directory not found: {directory}"
        )

    files = sorted(directory.glob("*.json"))

    if not files:
        raise ValueError("No embedding files found.")

    logger.info(
        "Found %d embedding files.",
        len(files)
    )

    return files


# --------------------------------------------------
# CONVERT CHUNKS INTO QDRANT POINTS
# --------------------------------------------------

def load_points_from_file(
    file_path: Path
) -> list[PointStruct]:
    """Convert embedded chunks into Qdrant points."""

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    points = []

    for chunk in data.get("chunks", []):

        vector = chunk.get("embedding")

        if not isinstance(vector, list):
            raise ValueError(
                f"Missing embedding in {file_path.name}"
            )

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Invalid vector dimension in {file_path.name}: "
                f"{len(vector)}"
            )

        vector = [float(value) for value in vector]

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                f"Non-finite vector value in {file_path.name}"
            )

        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "Skipping empty text chunk in %s",
                file_path.name
            )
            continue

        playlist_id = str(chunk["playlist_id"])
        video_id = str(chunk["video_id"])
        chunk_id = str(chunk["chunk_id"])

        point_id = generate_point_id(
            playlist_id,
            video_id,
            chunk_id
        )

        # Embedding belongs in the vector field,
        # not in the payload.
        payload = {
            key: value
            for key, value in chunk.items()
            if key != "embedding"
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    return points


# --------------------------------------------------
# BATCH UPLOAD
# --------------------------------------------------

def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct]
) -> int:
    """Upload points in batches using upsert."""

    uploaded = 0

    for start in range(0, len(points), BATCH_SIZE):

        batch = points[start:start + BATCH_SIZE]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True
        )

        uploaded += len(batch)

        logger.info(
            "Uploaded %d / %d points",
            uploaded,
            len(points)
        )

    return uploaded


# --------------------------------------------------
# PLAYLIST EMBEDDING UPLOAD
# --------------------------------------------------

def upload_playlist_embeddings(
    client: QdrantClient,
    playlist_id: str,
    collection_name: str = COLLECTION_NAME
) -> int:
    """Upload all video embeddings into Qdrant."""

    embedding_files = load_embedding_files(playlist_id)

    total_uploaded = 0

    for index, file_path in enumerate(embedding_files, start=1):

        logger.info(
            "Processing video %d / %d: %s",
            index,
            len(embedding_files),
            file_path.stem
        )

        points = load_points_from_file(file_path)

        if not points:
            logger.warning(
                "No valid points found in %s",
                file_path.name
            )
            continue

        uploaded = upload_points(
            client,
            collection_name,
            points
        )

        total_uploaded += uploaded

        logger.info(
            "Video %s uploaded: %d chunks",
            file_path.stem,
            uploaded
        )

    return total_uploaded


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(upload: bool = False) -> None:

    client = get_qdrant_client()

    create_collection(client)

    if upload:
        uploaded = upload_playlist_embeddings(
            client=client,
            playlist_id=PLAYLIST_ID,
            collection_name=COLLECTION_NAME
        )

        logger.info(
            "Total points upserted: %d",
            uploaded
        )

    info = get_collection_info(client)

    logger.info(
        "Collection information: %s",
        info
    )

    if upload:
        exact_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        ).count

        logger.info(
            "Verified Qdrant point count: %d",
            exact_count
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload playlist embeddings to Qdrant"
    )

    args = parser.parse_args()

    main(upload=args.upload)
