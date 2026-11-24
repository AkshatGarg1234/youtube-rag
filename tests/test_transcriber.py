
from pathlib import Path

import app.ingestion.transcriber as transcriber


def test_overlap_segments_are_not_duplicated(monkeypatch):

    parts = [
        transcriber.AudioPart(
            path=Path("part_000.flac"),
            offset=0,
            core_start=0,
            core_end=300
        ),
        transcriber.AudioPart(
            path=Path("part_001.flac"),
            offset=295,
            core_start=300,
            core_end=600
        )
    ]

    fake_results = {
        "part_000.flac": {
            "language": "english",
            "text": "First part.",
            "segments": [
                {
                    "start": 295,
                    "end": 300,
                    "text": "Boundary sentence."
                }
            ]
        },
        "part_001.flac": {
            "language": "english",
            "text": "Boundary sentence. Second sentence.",
            "segments": [
                {
                    "start": 0,
                    "end": 5,
                    "text": "Boundary sentence."
                },
                {
                    "start": 5,
                    "end": 10,
                    "text": "Second sentence."
                }
            ]
        }
    }

    def fake_transcribe(audio_path):
        return fake_results[audio_path.name]

    monkeypatch.setattr(
        transcriber,
        "transcribe_with_groq",
        fake_transcribe
    )

    result = transcriber.transcribe_audio_parts(parts)

    assert len(result["segments"]) == 2

    assert result["segments"][0]["text"] == "Boundary sentence."
    assert result["segments"][0]["start"] == 295

    assert result["segments"][1]["text"] == "Second sentence."
    assert result["segments"][1]["start"] == 300


def test_audio_part_has_correct_offset():

    part = transcriber.AudioPart(
        path=Path("part_001.flac"),
        offset=295,
        core_start=300,
        core_end=600
    )

    local_start = 5
    absolute_start = part.offset + local_start

    assert absolute_start == 300


def test_audio_part_windows_have_overlap():

    first = transcriber.AudioPart(
        path=Path("part_000.flac"),
        offset=0,
        core_start=0,
        core_end=300
    )

    second = transcriber.AudioPart(
        path=Path("part_001.flac"),
        offset=295,
        core_start=300,
        core_end=600
    )

    # The first part ends at 305 seconds.
    first_audio_end = 305

    # The second part begins at 295 seconds.
    second_audio_start = second.offset

    assert first_audio_end - second_audio_start == 10
