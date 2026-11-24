
import pytest

from app.ingestion.playlist_extractor import (
    extract_playlist_id,
    normalize_video,
    extract_playlist,
)


def test_extract_playlist_id():

    url = (
        "https://www.youtube.com/playlist"
        "?list=PL123456"
    )

    assert extract_playlist_id(url) == "PL123456"


def test_invalid_playlist_url():

    with pytest.raises(ValueError):

        extract_playlist_id(
            "https://www.youtube.com/watch?v=abc"
        )


def test_normalize_video():

    entry = {
        "id": "abc123",
        "title": "Introduction to RAG",
        "duration": 120,
        "availability": "public",
    }

    video = normalize_video(entry, 1)

    assert video.video_id == "abc123"

    assert video.title == "Introduction to RAG"

    assert video.playlist_position == 1

    assert video.duration == 120


def test_missing_video_id():

    entry = {
        "title": "Some Video"
    }

    with pytest.raises(ValueError):

        normalize_video(entry, 1)


def test_extract_playlist_with_duplicates(monkeypatch):

    class FakeYoutubeDL:

        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):

            return {
                "title": "Test Playlist",
                "entries": [
                    {
                        "id": "video1",
                        "title": "Video One"
                    },
                    {
                        "id": "video1",
                        "title": "Video One Duplicate"
                    },
                    {
                        "id": "video2",
                        "title": "Video Two"
                    }
                ]
            }

    monkeypatch.setattr(
        "app.ingestion.playlist_extractor.yt_dlp.YoutubeDL",
        FakeYoutubeDL
    )

    result = extract_playlist(
        "https://www.youtube.com/playlist?list=PL123"
    )

    assert result.total_videos == 3

    assert result.available_videos == 2

    assert len(result.extraction_issues) == 1

    assert result.videos[0].playlist_position == 1

    assert result.videos[1].playlist_position == 3
