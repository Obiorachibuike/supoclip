"""Reusable caches: audio-peak analysis and transcript reuse."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.media import timeline


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"video-bytes")
    return path


def test_audio_peaks_are_cached_and_reused(monkeypatch, video):
    calls = []

    def compute(path, max_peaks):
        calls.append(path)
        return [1.0, 9.5]

    monkeypatch.setattr(timeline, "_compute_audio_peak_times", compute)
    monkeypatch.setattr(
        timeline, "get_config", lambda: SimpleNamespace(transcription_cache_enabled=True)
    )

    assert timeline.detect_audio_peak_times(video) == [1.0, 9.5]
    assert timeline.detect_audio_peak_times(video) == [1.0, 9.5]
    assert len(calls) == 1

    payload = json.loads(video.with_suffix(".audio_peaks.json").read_text())
    assert payload["peaks"] == [1.0, 9.5]
    assert payload["max_peaks"] == 8
    assert payload["version"] == timeline.AUDIO_PEAK_CACHE_VERSION


def test_changed_source_invalidates_the_peak_cache(monkeypatch, video):
    calls = []
    monkeypatch.setattr(
        timeline,
        "_compute_audio_peak_times",
        lambda path, max_peaks: calls.append(path) or [4.0],
    )
    monkeypatch.setattr(
        timeline, "get_config", lambda: SimpleNamespace(transcription_cache_enabled=True)
    )

    timeline.detect_audio_peak_times(video)
    video.write_bytes(b"different-bytes")
    timeline.detect_audio_peak_times(video)

    assert len(calls) == 2


def test_different_peak_counts_do_not_share_an_entry(monkeypatch, video):
    calls = []
    monkeypatch.setattr(
        timeline,
        "_compute_audio_peak_times",
        lambda path, max_peaks: calls.append(max_peaks) or [float(max_peaks)],
    )
    monkeypatch.setattr(
        timeline, "get_config", lambda: SimpleNamespace(transcription_cache_enabled=True)
    )

    assert timeline.detect_audio_peak_times(video, max_peaks=4) == [4.0]
    assert timeline.detect_audio_peak_times(video, max_peaks=8) == [8.0]
    assert calls == [4, 8]


def test_cache_can_be_disabled(monkeypatch, video):
    calls = []
    monkeypatch.setattr(
        timeline,
        "_compute_audio_peak_times",
        lambda path, max_peaks: calls.append(path) or [2.0],
    )
    monkeypatch.setattr(
        timeline, "get_config", lambda: SimpleNamespace(transcription_cache_enabled=False)
    )

    timeline.detect_audio_peak_times(video)
    timeline.detect_audio_peak_times(video)

    assert len(calls) == 2
    assert not video.with_suffix(".audio_peaks.json").exists()


def test_corrupt_cache_is_ignored(monkeypatch, video):
    video.with_suffix(".audio_peaks.json").write_text("{not json")
    monkeypatch.setattr(
        timeline, "get_config", lambda: SimpleNamespace(transcription_cache_enabled=True)
    )
    monkeypatch.setattr(timeline, "_compute_audio_peak_times", lambda path, max_peaks: [3.0])

    assert timeline.detect_audio_peak_times(video) == [3.0]
