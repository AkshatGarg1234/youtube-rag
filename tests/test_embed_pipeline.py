
import json

import pytest

from app.processing import embed_pipeline


def test_validate_chunks():
    chunks = [
        {"text": "Valid transcript"},
        {"text": ""},
        {"text": "   "},
        {"text": "Another valid transcript"},
    ]

    result = embed_pipeline.validate_chunks(chunks)

    assert len(result) == 2


def test_embed_video(tmp_path, monkeypatch):
    playlist_id = "test_playlist"
    video_id = "test_video"

    chunk_dir = tmp_path / "chunks" / playlist_id
    chunk_dir.mkdir(parents=True)

    embedding_dir = tmp_path / "embeddings"

    chunk_data = {
        "chunks": [
            {
                "chunk_id": "test_video_001",
                "text": "Python is a programming language.",
                "start": 0,
                "end": 75,
                "youtube_url": "https://youtube.com/watch?v=test_video&t=0s",
            },
            {
                "chunk_id": "test_video_002",
                "text": "Machine learning uses data.",
                "start": 65,
                "end": 140,
                "youtube_url": "https://youtube.com/watch?v=test_video&t=65s",
            },
        ]
    }

    (chunk_dir / f"{video_id}.json").write_text(
        json.dumps(chunk_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        embed_pipeline,
        "CHUNK_DIR",
        tmp_path / "chunks",
    )

    monkeypatch.setattr(
        embed_pipeline,
        "EMBEDDING_DIR",
        embedding_dir,
    )

    fake_vectors = [
        [0.1] * embed_pipeline.EMBEDDING_DIMENSION,
        [0.2] * embed_pipeline.EMBEDDING_DIMENSION,
    ]

    monkeypatch.setattr(
        embed_pipeline,
        "generate_embeddings",
        lambda texts, batch_size: fake_vectors,
    )

    output_path = embed_pipeline.embed_video(
        playlist_id,
        video_id,
    )

    assert output_path.exists()

    result = json.loads(output_path.read_text())

    assert result["total_chunks"] == 2
    assert len(result["chunks"][0]["embedding"]) == 384
    assert result["chunks"][0]["text"] == chunk_data["chunks"][0]["text"]
    assert result["chunks"][0]["youtube_url"] == chunk_data["chunks"][0]["youtube_url"]


def test_missing_chunk_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        embed_pipeline,
        "CHUNK_DIR",
        tmp_path,
    )

    with pytest.raises(embed_pipeline.EmbeddingPipelineError):
        embed_pipeline.load_chunks(
            "unknown_playlist",
            "unknown_video",
        )
