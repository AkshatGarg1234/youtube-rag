
import argparse
import logging
from typing import Any

from qdrant_client import models

from app.processing.embeddings import generate_embedding
from app.vectorstore.qdrant_store import (
    COLLECTION_NAME,
    get_qdrant_client,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def search_chunks(
    question: str,
    top_k: int = 5,
    playlist_id: str | None = None,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search Qdrant for chunks semantically similar
    to the user's question.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # 1. Convert question into an embedding.
    query_vector = generate_embedding(question)

    # 2. Build optional metadata filters.
    conditions = []

    if playlist_id:
        conditions.append(
            models.FieldCondition(
                key="playlist_id",
                match=models.MatchValue(value=playlist_id),
            )
        )

    if video_id:
        conditions.append(
            models.FieldCondition(
                key="video_id",
                match=models.MatchValue(value=video_id),
            )
        )

    query_filter = (
        models.Filter(must=conditions)
        if conditions
        else None
    )

    # 3. Connect to Qdrant.
    client = get_qdrant_client()

    # 4. Search for similar vectors.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    # 5. Convert results into clean dictionaries.
    results = []

    for point in response.points:
        payload = point.payload or {}

        result = {
            "id": str(point.id),
            "score": float(point.score),
            **payload,
        }

        results.append(result)

    logger.info(
        "Retrieved %d chunks for the question.",
        len(results),
    )

    return results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search YouTube playlist knowledge using Qdrant."
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve.",
    )

    args = parser.parse_args()

    results = search_chunks(
        question=args.question,
        top_k=args.top_k,
    )

    if not results:
        print("\nNo relevant chunks found.")
        return

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for index, result in enumerate(results, start=1):
        print(f"\nResult #{index}")
        print(f"Score: {result['score']:.4f}")
        print(f"Video: {result.get('video_title', 'Unknown')}")
        print(f"Video ID: {result.get('video_id', 'Unknown')}")
        print(f"Start: {result.get('start', 0)} seconds")
        print(f"Text: {result.get('text', '')[:500]}")
        print(f"URL: {result.get('youtube_url', 'Unavailable')}")
        print("-" * 70)


if __name__ == "__main__":
    main()
