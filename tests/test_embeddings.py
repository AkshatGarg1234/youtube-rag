
from app.processing.embeddings import (
    generate_embedding,
    generate_embeddings,
    EMBEDDING_DIMENSION,
)


def test_single_embedding_dimension():
    embedding = generate_embedding(
        "Monotonic stack helps find the largest rectangle."
    )

    assert len(embedding) == EMBEDDING_DIMENSION


def test_batch_embedding_count():
    texts = [
        "Python is a programming language.",
        "Machine learning uses data to learn patterns.",
        "Qdrant is a vector database.",
    ]

    embeddings = generate_embeddings(texts)

    assert len(embeddings) == 3

    assert all(
        len(embedding) == EMBEDDING_DIMENSION
        for embedding in embeddings
    )


def test_empty_batch():
    assert generate_embeddings([]) == []
