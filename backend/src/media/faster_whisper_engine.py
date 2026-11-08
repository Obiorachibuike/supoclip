"""Faster-Whisper transcription engine.

Faster-Whisper (CTranslate2) is the default *free* transcription engine: it runs
locally, needs no API key, and returns word-level timestamps that the clip
boundaries and burned-in subtitles depend on.

Design notes
------------
* The model is loaded lazily and cached per (size, device, compute type) so a
  long-lived ARQ worker never reloads it between tasks.
* Device selection is explicit, with a CPU fallback: ``auto`` probes CUDA and
  drops to CPU when no device is visible, and a CUDA load failure retries on CPU
  with an integer compute type instead of failing the task.
* Nothing here calls a paid provider. If local transcription fails, the error is
  raised to the user with an explanation; SupoClip never silently switches to
  AssemblyAI.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "medium"
# Sizes Faster-Whisper documents. Aliases are normalised onto real model ids.
SUPPORTED_MODEL_SIZES: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v3",
)
MODEL_ALIASES = {"large": "large-v3"}
CPU_COMPUTE_TYPE = "int8"
CUDA_COMPUTE_TYPE = "float16"
VALID_COMPUTE_TYPES = frozenset(
    {"auto", "default", "int8", "int8_float16", "float16", "float32"}
)
# Words are grouped into analysis lines of this size at most, so transcript
# lines stay sentence-sized and clip boundaries can snap onto them.
MAX_WORDS_PER_ANALYSIS_SEGMENT = 40


class FasterWhisperUnavailable(RuntimeError):
    """The engine or its weights are not available in this deployment."""


class FasterWhisperTranscriptionError(RuntimeError):
    """Local transcription failed for a specific audio file."""


_model_cache: Dict[tuple[str, str, str, str | None], Any] = {}
_cache_lock = threading.Lock()
# Set once a CUDA load has failed so later tasks go straight to CPU instead of
# retrying a GPU that is not usable in this container.
_cuda_load_failed = False


def is_available() -> bool:
    """Whether ``faster-whisper`` (and CTranslate2) can be imported."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        return False
    return True


def _cuda_available() -> bool:
    """True when CTranslate2 reports at least one visible CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - CPU-only installs
        return False


def normalize_model_size(model_size: str | None) -> str:
    normalized = (model_size or "").strip().lower()
    if not normalized:
        return DEFAULT_MODEL_SIZE
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    if normalized not in SUPPORTED_MODEL_SIZES:
        logger.warning(
            "Unrecognised Faster-Whisper model '%s'; Faster-Whisper will try to load "
            "it as a local path or Hugging Face id",
            model_size,
        )
    return normalized


def resolve_device(requested: str | None) -> str:
    """Resolve ``auto``/``cuda``/``cpu`` into a device the engine can use."""
    global _cuda_load_failed
    if _cuda_load_failed:
        return "cpu"
    normalized = (requested or "auto").strip().lower()
    if normalized == "cuda":
        # An explicit cuda request still needs a visible device, otherwise the
        # load fails half way through a long task.
        return "cuda" if _cuda_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    return "cuda" if _cuda_available() else "cpu"


def resolve_compute_type(device: str, requested: str | None) -> str:
    normalized = (requested or "auto").strip().lower()
    if normalized in VALID_COMPUTE_TYPES and normalized not in {"auto", "default"}:
        return normalized
    return CUDA_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE


def get_model(
    model_size: str | None = None,
    *,
    device: str = "auto",
    compute_type: str = "auto",
    download_root: str | None = None,
    cpu_threads: int = 0,
):
    """Load and cache a Faster-Whisper model."""
    global _cuda_load_failed
    if not is_available():
        raise FasterWhisperUnavailable(
            "Faster-Whisper is not installed in this deployment. Install the backend "
            "with its default dependencies (uv sync) or set "
            "TRANSCRIPTION_PROVIDER=whisper / assemblyai."
        )
    from faster_whisper import WhisperModel

    resolved_device = resolve_device(device)
    resolved_compute_type = resolve_compute_type(resolved_device, compute_type)
    resolved_model = normalize_model_size(model_size)
    key = (resolved_model, resolved_device, resolved_compute_type, download_root or None)

    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        logger.info(
            "Loading Faster-Whisper model %s (device=%s, compute_type=%s)",
            resolved_model,
            resolved_device,
            resolved_compute_type,
        )
        started = time.monotonic()
        try:
            model = WhisperModel(
                resolved_model,
                device=resolved_device,
                compute_type=resolved_compute_type,
                download_root=download_root,
                cpu_threads=max(0, cpu_threads),
            )
        except Exception as error:
            if resolved_device != "cuda":
                raise FasterWhisperUnavailable(
                    f"Faster-Whisper could not load the model '{resolved_model}' "
                    f"({type(error).__name__}: {error}). Check FASTER_WHISPER_MODEL and "
                    "that the model can be downloaded, or set "
                    "FASTER_WHISPER_DOWNLOAD_ROOT to a directory that already holds it."
                ) from error
            logger.warning(
                "CUDA load of Faster-Whisper model %s failed (%s); retrying on CPU",
                resolved_model,
                type(error).__name__,
            )
            _cuda_load_failed = True
            key = (resolved_model, "cpu", CPU_COMPUTE_TYPE, download_root or None)
            cached = _model_cache.get(key)
            if cached is not None:
                return cached
            try:
                model = WhisperModel(
                    resolved_model,
                    device="cpu",
                    compute_type=CPU_COMPUTE_TYPE,
                    download_root=download_root,
                    cpu_threads=max(0, cpu_threads),
                )
            except Exception as cpu_error:
                raise FasterWhisperUnavailable(
                    f"Faster-Whisper could not load the model '{resolved_model}' on GPU "
                    f"or CPU ({type(cpu_error).__name__}: {cpu_error})."
                ) from cpu_error

        _model_cache[key] = model
        logger.info(
            "Faster-Whisper model %s ready in %.1fs", resolved_model, time.monotonic() - started
        )
        return model


def clear_model_cache() -> None:
    """Drop loaded models (tests and memory-constrained deployments)."""
    global _cuda_load_failed
    with _cache_lock:
        _model_cache.clear()
        _cuda_load_failed = False


def transcribe_audio(
    audio_path: Path,
    *,
    model_size: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    download_root: str | None = None,
    cpu_threads: int = 0,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Transcribe one audio file and return a Whisper-compatible result dict.

    The returned mapping has the same shape ``cache_transcript_data`` and
    ``format_transcript_for_analysis`` already understand (``segments`` with
    second-based timings plus a flat ``words`` list), so word timings survive
    into subtitle rendering without a second conversion layer.
    """
    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FasterWhisperTranscriptionError(
            f"Audio file for transcription is missing or empty: {path.name}"
        )

    model = get_model(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=cpu_threads,
    )
    # Whisper expects an ISO 639-1 code; AssemblyAI-style values such as
    # ``en_us`` are reduced to their base language.
    whisper_language = language.split("_", 1)[0] if language else None
    logger.info(
        "Starting Faster-Whisper transcription (model=%s, language=%s, vad=%s)",
        normalize_model_size(model_size),
        whisper_language or "auto",
        vad_filter,
    )

    try:
        segments_iter, info = model.transcribe(
            str(path),
            language=whisper_language,
            beam_size=max(1, beam_size),
            word_timestamps=True,
            vad_filter=vad_filter,
        )
    except Exception as error:
        raise FasterWhisperTranscriptionError(
            f"Faster-Whisper failed to start transcription ({type(error).__name__}: {error})."
        ) from error

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    words: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    try:
        for segment in segments_iter:
            segment_text = (getattr(segment, "text", "") or "").strip()
            if segment_text:
                text_parts.append(segment_text)
            segment_words: List[Dict[str, Any]] = []
            for word in getattr(segment, "words", None) or []:
                word_text = (getattr(word, "word", "") or "").strip()
                if not word_text:
                    continue
                entry = {
                    "word": word_text,
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "probability": float(getattr(word, "probability", 1.0) or 1.0),
                }
                segment_words.append(entry)
                words.append(entry)
            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0) or 0.0),
                    "end": float(getattr(segment, "end", 0.0) or 0.0),
                    "text": segment_text,
                    # Word timings stay attached to their own segment so the
                    # transcript cache keeps millisecond-precise subtitle data.
                    "words": segment_words,
                }
            )
            if on_progress is not None and duration > 0:
                try:
                    on_progress(
                        min(1.0, float(getattr(segment, "end", 0.0) or 0.0) / duration)
                    )
                except Exception:  # pragma: no cover - progress must never fail a task
                    pass
    except Exception as error:
        raise FasterWhisperTranscriptionError(
            f"Faster-Whisper failed while decoding audio "
            f"({type(error).__name__}: {error})."
        ) from error

    if not words and not segments:
        raise FasterWhisperTranscriptionError(
            "Faster-Whisper returned no speech for this audio. The source may be "
            "silent, music-only, or the wrong audio track."
        )

    return {
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": duration,
        "segments": segments,
        # Sentence-sized lines for the clip-selection prompt; the raw segments
        # above stay authoritative for word-level subtitle timing.
        "analysis_segments": build_analysis_segments(segments, words),
        "words": words,
        "engine": "faster_whisper",
    }


def build_analysis_segments(
    segments: Iterable[Dict[str, Any]], words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group words into sentence-sized analysis lines.

    Whisper's own segments are sometimes a whole paragraph, which makes the
    transcript prompt's line boundaries useless for grounding clip start/end
    times. Grouping the word timings at sentence punctuation (with a hard cap)
    gives the model lines it can snap clip boundaries onto while keeping the
    original segment text intact.
    """
    if not words:
        return list(segments)

    grouped: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        grouped.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": _join_words([word["word"] for word in current]),
            }
        )
        current = []

    for word in words:
        current.append(word)
        ends_sentence = word["word"].endswith((".", "!", "?", "。", "！", "？"))
        if ends_sentence or len(current) >= MAX_WORDS_PER_ANALYSIS_SEGMENT:
            flush()
    flush()
    return grouped or list(segments)


def _join_words(tokens: List[str]) -> str:
    """Join word tokens without inventing spaces around punctuation."""
    text = " ".join(token.strip() for token in tokens if token.strip())
    for before, after in (
        (" ,", ","),
        (" .", "."),
        (" !", "!"),
        (" ?", "?"),
        (" ;", ";"),
        (" :", ":"),
        (" n't", "n't"),
        (" 're", "'re"),
        (" 've", "'ve"),
        (" 'll", "'ll"),
        (" 'd", "'d"),
        (" 'm", "'m"),
        (" 's", "'s"),
    ):
        text = text.replace(before, after)
    return text.strip()
