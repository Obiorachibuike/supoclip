"""Processing modes, provider catalogue and paid-provider protection."""

from __future__ import annotations

import pytest

from src.ai_providers import (
    AICapacityError,
    AIProviderQuotaExhausted,
    PaidProviderNotAuthorized,
    ProviderPolicy,
    default_transcription_provider_for_mode,
    get_provider_policy,
    normalize_ai_mode,
    normalize_transcription_provider,
    provider_label,
)
from src.config import Config

PROVIDER_ENV = (
    "AI_PROVIDER_MODE",
    "AI_ALLOW_PAID_PROVIDERS",
    "AI_FREE_FALLBACK_ENABLED",
    "LLM",
    "TRANSCRIPTION_PROVIDER",
    "AGNES_API_KEY",
    "AGNES_MODEL",
    "ASSEMBLY_AI_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL",
    "WHISPER_MODEL_SIZE",
    "FASTER_WHISPER_MODEL",
)


def clean_config(monkeypatch, **env) -> Config:
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Config()


def test_brand_new_install_defaults_to_free_mode(monkeypatch):
    config = clean_config(monkeypatch)

    assert config.ai_provider_mode == "free"
    assert config.llm == "agnes:agnes-2.5-flash"
    assert config.transcription_provider == "faster_whisper"
    assert config.faster_whisper_model == "medium"
    assert get_provider_policy(config).paid_authorized is False


def test_legacy_deployment_keeps_its_configured_provider(monkeypatch):
    config = clean_config(
        monkeypatch,
        LLM="google-gla:gemini-3-flash-preview",
        GOOGLE_API_KEY="google-key",
        TRANSCRIPTION_PROVIDER="assemblyai",
        ASSEMBLY_AI_API_KEY="assembly-key",
    )

    assert config.ai_provider_mode == "custom"
    assert config.llm == "google-gla:gemini-3-flash-preview"
    assert config.transcription_provider == "assemblyai"
    assert get_provider_policy(config).paid_authorized is True


@pytest.mark.parametrize("value", ["free", "premium", "local", "custom", "PREMIUM", "premium-mode", "free mode"])
def test_modes_are_normalised(value):
    normalized = normalize_ai_mode(value)
    assert normalized in {"free", "premium", "local", "custom"}


def test_unknown_mode_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="AI_PROVIDER_MODE"):
        clean_config(monkeypatch, AI_PROVIDER_MODE="cheap")


def test_free_mode_ignores_an_explicit_paid_llm(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        LLM="openai:gpt-5.2",
        OPENAI_API_KEY="openai-key",
        AGNES_API_KEY="agnes-key",
    )

    assert config.llm == "agnes:agnes-2.5-flash"


def test_free_mode_keeps_an_explicit_free_local_model(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        LLM="ollama:llama3.2:3b",
        OLLAMA_MODEL="llama3.2:3b",
    )

    assert config.llm == "ollama:llama3.2:3b"


def test_paid_llm_is_allowed_when_paid_providers_are_authorized(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        AI_ALLOW_PAID_PROVIDERS="true",
        LLM="openai:gpt-5.2",
        OPENAI_API_KEY="openai-key",
    )

    assert config.llm == "openai:gpt-5.2"
    assert get_provider_policy(config).paid_authorized is True


def test_free_mode_replaces_an_explicit_paid_transcription_engine(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        TRANSCRIPTION_PROVIDER="assemblyai",
        ASSEMBLY_AI_API_KEY="assembly-key",
    )

    assert config.transcription_provider == "faster_whisper"


def test_premium_mode_keeps_the_explicit_paid_transcription_engine(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="premium",
        TRANSCRIPTION_PROVIDER="assemblyai",
        ASSEMBLY_AI_API_KEY="assembly-key",
        LLM="openai:gpt-5.2",
        OPENAI_API_KEY="openai-key",
    )

    assert config.transcription_provider == "assemblyai"
    assert config.llm == "openai:gpt-5.2"
    assert get_provider_policy(config).requires_paid_provider is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("faster_whisper", "faster_whisper"),
        ("faster-whisper", "faster_whisper"),
        ("whisper", "whisper"),
        ("youtube_captions", "youtube_captions"),
        ("assemblyai", "assemblyai"),
        ("nonsense", None),
        ("", None),
    ],
)
def test_transcription_provider_normalisation(raw, expected):
    assert normalize_transcription_provider(raw) == expected


@pytest.mark.parametrize(
    "mode,expected", [("free", "faster_whisper"), ("local", "faster_whisper"), ("premium", "assemblyai")]
)
def test_mode_transcription_defaults(mode, expected):
    assert default_transcription_provider_for_mode(mode) == expected


def test_local_mode_uses_ollama_and_local_transcription(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="local",
        OLLAMA_MODEL="gpt-oss:20b",
        OLLAMA_BASE_URL="http://127.0.0.1:11434/v1",
    )

    assert config.llm == "ollama:gpt-oss:20b"
    assert config.transcription_provider == "faster_whisper"
    assert ProviderPolicy(config).paid_authorized is False


def test_paid_provider_is_refused_without_authorization(monkeypatch):
    config = clean_config(monkeypatch, AI_PROVIDER_MODE="free", AGNES_API_KEY="k")
    policy = ProviderPolicy(config)

    with pytest.raises(PaidProviderNotAuthorized, match="never spends paid credits"):
        policy.assert_paid_authorization("openai", purpose="Clip analysis")

    # Free providers never need authorization.
    policy.assert_paid_authorization("agnes", purpose="Clip analysis")


def test_paid_provider_is_allowed_in_premium_mode(monkeypatch):
    config = clean_config(
        monkeypatch, AI_PROVIDER_MODE="premium", LLM="openai:gpt-5.2", OPENAI_API_KEY="k"
    )
    ProviderPolicy(config).assert_paid_authorization("openai", purpose="Clip analysis")


def test_free_fallback_requires_a_configured_local_model(monkeypatch):
    config = clean_config(monkeypatch, AGNES_API_KEY="k")
    assert ProviderPolicy(config).free_fallback_model() is None

    config = clean_config(monkeypatch, AGNES_API_KEY="k", OLLAMA_MODEL="gpt-oss:20b")
    assert ProviderPolicy(config).free_fallback_model() == "ollama:gpt-oss:20b"


def test_free_fallback_can_be_disabled(monkeypatch):
    config = clean_config(
        monkeypatch,
        AGNES_API_KEY="k",
        OLLAMA_MODEL="gpt-oss:20b",
        AI_FREE_FALLBACK_ENABLED="false",
    )

    assert ProviderPolicy(config).free_fallback_model() is None


def test_free_fallback_is_not_the_already_selected_model(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        LLM="ollama:gpt-oss:20b",
        OLLAMA_MODEL="gpt-oss:20b",
    )

    assert ProviderPolicy(config).free_fallback_model() is None


def test_capacity_error_explains_free_and_premium_ways_forward(monkeypatch):
    config = clean_config(monkeypatch, AGNES_API_KEY="k")
    error = ProviderPolicy(config).capacity_error(
        AIProviderQuotaExhausted("quota gone", provider_id="agnes")
    )

    assert isinstance(error, AICapacityError)
    message = str(error)
    assert "agnes" in message
    assert "OLLAMA_MODEL" in message
    assert "AI_PROVIDER_MODE=premium" in message
    assert "No paid provider was used" in message
    # Never claim a dollar amount or a guaranteed outcome.
    assert "$" not in message


def test_describe_reports_mode_providers_and_warnings_without_secrets(monkeypatch):
    config = clean_config(monkeypatch, AGNES_API_KEY="super-secret-key")
    payload = ProviderPolicy(config).describe()

    assert payload["mode"] == "free"
    assert payload["reasoning"]["provider"] == "agnes"
    assert payload["reasoning"]["configured"] is True
    assert payload["transcription"]["provider"] == "faster_whisper"
    assert payload["requires_paid_provider"] is False
    assert payload["paid_providers_authorized"] is False
    assert "super-secret-key" not in str(payload)
    ids = {provider["id"] for provider in payload["providers"]}
    assert {"agnes", "openai", "faster_whisper", "assemblyai"} <= ids
    paid = {provider["id"]: provider for provider in payload["providers"]}
    assert paid["openai"]["authorized"] is False
    assert paid["assemblyai"]["authorized"] is False
    for provider in payload["providers"]:
        assert "api_key" not in provider


def test_describe_warns_when_the_free_key_is_missing(monkeypatch):
    config = clean_config(monkeypatch)
    payload = ProviderPolicy(config).describe()

    assert payload["reasoning"]["configured"] is False
    assert any("AGNES_API_KEY" in warning for warning in payload["warnings"])


def test_describe_warns_about_unused_paid_credentials(monkeypatch):
    config = clean_config(
        monkeypatch,
        AI_PROVIDER_MODE="free",
        AGNES_API_KEY="k",
        OPENAI_API_KEY="openai-key",
        LLM="agnes:agnes-2.5-flash",
    )
    payload = ProviderPolicy(config).describe()

    assert any("will not be used" in warning for warning in payload["warnings"])


def test_provider_label_is_human_readable(monkeypatch):
    assert provider_label("agnes:agnes-2.5-flash") == "Agnes AI"
    assert provider_label("ollama:gpt-oss:20b") == "Ollama (local)"
    assert provider_label("vendor-custom:model") == "vendor-custom"
