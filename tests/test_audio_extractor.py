
import json

from types import SimpleNamespace

from app.ingestion import audio_extractor


def test_validate_audio(monkeypatch, tmp_path):

    audio_file = tmp_path / "test.wav"

    audio_file.write_bytes(b"fake audio")

    fake_output = {
        "streams": [
            {
                "codec_name": "pcm_s16le",
                "sample_rate": "16000",
                "channels": 1
            }
        ],
        "format": {
            "duration": "120.5"
        }
    }

    def fake_run(*args, **kwargs):

        return SimpleNamespace(
            stdout=json.dumps(fake_output),
            stderr="",
            returncode=0
        )

    monkeypatch.setattr(
        audio_extractor.subprocess,
        "run",
        fake_run
    )

    result = audio_extractor.validate_audio(
        audio_file
    )

    assert result["duration"] == 120.5

    assert result["sample_rate"] == 16000

    assert result["channels"] == 1


def test_validate_audio_missing_file(tmp_path):

    missing_file = tmp_path / "missing.wav"

    try:

        audio_extractor.validate_audio(
            missing_file
        )

        assert False, "Expected an exception"

    except audio_extractor.AudioExtractionError:

        assert True


def test_playlist_processing_continues_after_failure(
    monkeypatch
):

    videos = [
        {
            "video_id": "video1",
            "title": "Video One"
        },
        {
            "video_id": "video2",
            "title": "Video Two"
        }
    ]

    monkeypatch.setattr(
        audio_extractor,
        "check_dependencies",
        lambda: None
    )

    monkeypatch.setattr(
        audio_extractor,
        "load_playlist_metadata",
        lambda playlist_id: {
            "playlist_title": "Test Playlist",
            "videos": videos
        }
    )

    def fake_extract(video, playlist_id, force):

        if video["video_id"] == "video1":

            raise Exception("Simulated download failure")

        return audio_extractor.AudioExtractionResult(
            video_id="video2",
            video_title="Video Two",
            status="success"
        )

    monkeypatch.setattr(
        audio_extractor,
        "extract_audio_for_video",
        fake_extract
    )

    results = audio_extractor.extract_playlist_audio(
        "PL123"
    )

    assert len(results) == 2

    assert results[0].status == "failed"

    assert results[1].status == "success"


def test_manifest_creation(monkeypatch, tmp_path):

    monkeypatch.setattr(
        audio_extractor,
        "METADATA_DIR",
        tmp_path
    )

    results = [

        audio_extractor.AudioExtractionResult(
            video_id="video1",
            video_title="Video One",
            status="success"
        ),

        audio_extractor.AudioExtractionResult(
            video_id="video2",
            video_title="Video Two",
            status="failed",
            error="Download failed"
        )
    ]

    audio_extractor.save_audio_manifest(
        "PL123",
        results
    )

    manifest_path = (
        tmp_path / "PL123_audio_manifest.json"
    )

    assert manifest_path.exists()

    with open(manifest_path) as file:

        manifest = json.load(file)

    assert manifest["successful"] == 1

    assert manifest["failed"] == 1
