
from app.processing.chunker import create_chunks


def generate_segments(count=12):
    """Generate synthetic 5-second transcript segments."""
    return [
        {
            "start": i * 5,
            "end": (i + 1) * 5,
            "text": f"Segment {i}"
        }
        for i in range(count)
    ]


def test_chunk_duration():
    segments = generate_segments()

    chunks = create_chunks(
        segments,
        target_seconds=20,
        overlap_seconds=5
    )

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 20


def test_chunks_have_expected_overlap():
    segments = generate_segments()

    chunks = create_chunks(
        segments,
        target_seconds=20,
        overlap_seconds=5
    )

    assert len(chunks) >= 2

    overlap = chunks[0]["end"] - chunks[1]["start"]

    assert overlap == 5


def test_last_chunk_contains_remaining_content():
    segments = generate_segments()

    chunks = create_chunks(
        segments,
        target_seconds=20,
        overlap_seconds=5
    )

    assert chunks[-1]["end"] == 60


def test_empty_segments():
    chunks = create_chunks([])

    assert chunks == []
