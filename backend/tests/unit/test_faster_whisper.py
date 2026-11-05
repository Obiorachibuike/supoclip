"""Faster-Whisper engine, timestamp fidelity and local-transcription fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.media import faster_whisper_engine as engine
from src.media import transcription
from src.media.transcription import TranscriptionError


class FakeWord:
    def __init__(self, word: str, start: float, end: float, probability: float = 0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class FakeSegment:
    def __init__(self, start: float, end: float, text: str, words):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


def fake_model(segments):
    def transcribe(path, **kwargs):
        assert Path(path).exists()
        duration = segments[-1].end if segments else 0.0
        return iter(segments), SimpleNamespace(duration=duration, language="en", language_probability=0.98)

    return SimpleNamespace(transcribe=transcribe, kwargs={})


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"audio")
    return path


@pytest.fixture(autouse=True)
def clear_engine_cache():
    engine.clear_model_cache()
    yield
    engine.clear_model_cache()


def test_engine_reports_unavailable_when_package_is_missing(monkeypatch):
    monkeypatch.setattr(engine, "is_available", lambda: False)

    assert engine.is_available() is False
    with pytest.raises(engine.FasterWhisperUnavailable, match="not installed"):
        engine.get_model("small")


def test_auto_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(engine, "_cuda_available", lambda: False)

    assert engine.resolve_device("auto") == "cpu"
    assert engine.resolve_device("cuda") == "cpu"
    assert engine.resolve_device("cpu") == "cpu"
    assert engine.resolve_compute_type("cpu", "auto") == "int8"


def test_cuda_is_used_when_available(monkeypatch):
    monkeypatch.setattr(engine, "_cuda_available", lambda: True)

    assert engine.resolve_device("auto") == "cuda"
    assert engine.resolve_device("cuda") == "cuda"
    assert engine.resolve_compute_type("cuda", "auto") == "float16"
    # An explicit compute type always wins.
    assert engine.resolve_compute_type("cuda", "int8_float16") == "int8_float16"


def test_model_aliases_and_defaults():
    assert engine.normalize_model_size(None) == "medium"
    assert engine.normalize_model_size("large") == "large-v3"
    assert engine.normalize_model_size("small") == "small"


def test_transcription_preserves_word_level_timestamps(monkeypatch, audio_file):
    segments = [
        FakeSegment(
            0.0,
            2.4,
            " Hello there. How are you?",
            [
                FakeWord("Hello", 0.0, 0.4),
                FakeWord("there.", 0.4, 0.9),
                FakeWord("How", 1.2, 1.4),
                FakeWord("are", 1.4, 1.6),
                FakeWord("you?", 1.6, 2.0),
            ],
        ),
        FakeSegment(2.4, 4.0, " Fine.", [FakeWord("Fine.", 2.4, 3.0)]),
    ]
    monkeypatch.setattr(engine, "get_model", lambda *args, **kwargs: fake_model(segments))

    result = engine.transcribe_audio(audio_file, model_size="small", language="en_us")

    assert result["text"] == "Hello there. How are you? Fine."
    assert result["engine"] == "faster_whisper"
    # Word timings stay in seconds and in order: subtitles depend on this.
    assert [(word["word"], word["start"], word["end"]) for word in result["words"]] == [
        ("Hello", 0.0, 0.4),
        ("there.", 0.4, 0.9),
        ("How", 1.2, 1.4),
        ("are", 1.4, 1.6),
        ("you?", 1.6, 2.0),
        ("Fine.", 2.4, 3.0),
    ]
    # Analysis lines break on sentence punctuation, not on engine segments.
    assert [segment["text"] for segment in result["analysis_segments"]] == [
        "Hello there.",
        "How are you?",
        "Fine.",
    ]
    assert result["analysis_segments"][1]["start"] == 1.2


def test_transcript_cache_keeps_millisecond_word_timings(monkeypatch, audio_file, tmp_path):
    segments = [
        FakeSegment(0.0, 0.9, " Hello there.", [FakeWord("Hello", 0.0, 0.4), FakeWord("there.", 0.4, 0.9)])
    ]
    monkeypatch.setattr(engine, "get_model", lambda *args, **kwargs: fake_model(segments))

    result = engine.transcribe_audio(audio_file)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    transcription.cache_transcript_data(video_path, result)

    cached = transcription.load_cached_transcript_data(video_path)
    assert cached["words"][0] == {
        "text": "Hello",
        "start": 0,
        "end": 400,
        "confidence": 0.9,
        "speaker": None,
    }
    assert cached["utterances"][0]["text"] == "Hello there."
    assert cached["language_code"] == "en"


def test_analysis_lines_prefer_sentence_segments():
    words = [
        {"word": "One", "start": 0.0, "end": 0.2},
        {"word": "two.", "start": 0.2, "end": 0.4},
        {"word": "Three", "start": 0.4, "end": 0.6},
    ]
    grouped = engine.build_analysis_segments([{"start": 0.0, "end": 0.6, "text": "One two. Three"}], words)

    assert [segment["text"] for segment in grouped] == ["One two.", "Three"]


def test_long_word_runs_are_capped_into_lines():
    words = [{"word": f"word{index}", "start": index * 0.2, "end": index * 0.2 + 0.2} for index in range(90)]

    grouped = engine.build_analysis_segments([], words)

    assert len(grouped) == 3
    assert all(len(segment["text"].split()) <= engine.MAX_WORDS_PER_ANALYSIS_SEGMENT for segment in grouped)
    assert grouped[0]["start"] == 0.0
    assert grouped[-1]["end"] == words[-1]["end"]


def test_empty_audio_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")

    with pytest.raises(engine.FasterWhisperTranscriptionError, match="missing or empty"):
        engine.transcribe_audio(empty)


def test_silent_audio_is_reported(monkeypatch, audio_file):
    monkeypatch.setattr(engine, "get_model", lambda *args, **kwargs: fake_model([]))

    with pytest.raises(engine.FasterWhisperTranscriptionError, match="no speech"):
        engine.transcribe_audio(audio_file)


def test_cuda_load_failure_retries_on_cpu(monkeypatch):
    created = []
    monkeypatch.setattr(engine, "_cuda_available", lambda: True)

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type, download_root=None, cpu_threads=0):
            created.append((model_size, device, compute_type))
            if device == "cuda":
                raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(engine, "is_available", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    model = engine.get_model("small", device="cuda")

    assert isinstance(model, FakeWhisperModel)
    assert created == [("small", "cuda", "float16"), ("small", "cpu", "int8")]
    # The CPU model is cached, so a second request does not reload weights.
    engine.get_model("small", device="cuda")
    assert len(created) == 2


def test_missing_model_download_is_explained(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot download weights")

    monkeypatch.setattr(engine, "is_available", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    with pytest.raises(engine.FasterWhisperUnavailable, match="FASTER_WHISPER_DOWNLOAD_ROOT"):
        engine.get_model("small", device="cpu")


def test_config_defaults_for_local_transcription(monkeypatch):
    from src.config import Config

    for name in ("FASTER_WHISPER_MODEL", "WHISPER_MODEL_SIZE", "FASTER_WHISPER_DEVICE", "TRANSCRIPTION_PROVIDER", "AI_PROVIDER_MODE", "LLM", "ASSEMBLY_AI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    config = Config()

    assert config.transcription_provider == "faster_whisper"
    assert config.faster_whisper_model == "medium"
    assert config.faster_whisper_device == "auto"
    assert config.faster_whisper_compute_type == "auto"


def test_whisper_model_size_env_is_honoured_by_faster_whisper(monkeypatch):
    from src.config import Config

    monkeypatch.setenv("WHISPER_MODEL_SIZE", "small")
    monkeypatch.delenv("FASTER_WHISPER_MODEL", raising=False)

    assert Config().faster_whisper_model == "small"


@pytest.mark.asyncio
async def test_dispatch_transcribes_locally_and_caches(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    config = SimpleNamespace(
        transcription_provider="faster_whisper",
        transcription_language="",
        faster_whisper_model="small",
        faster_whisper_device="cpu",
        faster_whisper_compute_type="int8",
        faster_whisper_beam_size=5,
        faster_whisper_vad_filter=True,
        faster_whisper_download_root=None,
        faster_whisper_cpu_threads=0,
        transcription_cache_enabled=True,
    )
    monkeypatch.setattr(transcription, "get_config", lambda: config)
    monkeypatch.setattr(transcription, "faster_whisper_available", lambda: True)
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: path)
    monkeypatch.setattr(
        transcription,
        "faster_whisper_transcribe",
        Mock(
            return_value={
                "text": "Hello there.",
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello there."}],
                "analysis_segments": [{"start": 0.0, "end": 1.0, "text": "Hello there."}],
                "words": [{"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.9}],
            }
        ),
    )
    cache = Mock()
    monkeypatch.setattr(transcription, "cache_transcript_data", cache)

    result = transcription.get_video_transcript(video)

    assert result == "[00:00 - 00:01] Hello there."
    cache.assert_called_once()
    assert cache.call_args.args[0] == video


def test_local_failure_explains_itself_and_never_uses_a_paid_provider(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    config = SimpleNamespace(
        transcription_provider="faster_whisper",
        transcription_language="",
        faster_whisper_model="small",
        transcription_cache_enabled=True,
    )
    monkeypatch.setattr(transcription, "get_config", lambda: config)
    monkeypatch.setattr(transcription, "faster_whisper_available", lambda: True)
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: path)
    monkeypatch.setattr(
        transcription,
        "faster_whisper_transcribe",
        Mock(side_effect=engine.FasterWhisperTranscriptionError("decoder crashed")),
    )
    paid = Mock()
    monkeypatch.setattr(transcription, "_get_transcript_with_assemblyai", paid)

    with pytest.raises(TranscriptionError) as caught:
        transcription.get_video_transcript(video)

    assert "decoder crashed" in str(caught.value)
    assert "No paid provider" in str(caught.value)
    assert "assemblyai" in str(caught.value)
    paid.assert_not_called()


def test_missing_engine_does_not_silently_switch_engines(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        transcription,
        "get_config",
        lambda: SimpleNamespace(transcription_provider="faster_whisper"),
    )
    monkeypatch.setattr(transcription, "faster_whisper_available", lambda: False)
    paid = Mock()
    monkeypatch.setattr(transcription, "_get_transcript_with_assemblyai", paid)

    with pytest.raises(TranscriptionError, match="not installed"):
        transcription.get_video_transcript(video)

    paid.assert_not_called()
