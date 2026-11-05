"""The AI provider status endpoint: mode, availability, usage, no secrets."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.routes.ai_providers import get_ai_providers
from src.config import Config, set_config_override


def build_request(headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/ai/providers",
            "headers": [
                (key.lower().encode("utf-8"), value.encode("utf-8"))
                for key, value in (headers or {}).items()
            ],
        }
    )


@pytest.fixture
def config(monkeypatch):
    for name in (
        "AI_PROVIDER_MODE",
        "LLM",
        "TRANSCRIPTION_PROVIDER",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "ASSEMBLY_AI_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGNES_API_KEY", "agnes-secret-value")
    runtime_config = Config()
    runtime_config.backend_auth_secret = None
    runtime_config.allow_unsigned_backend_auth = True
    set_config_override(runtime_config)
    yield runtime_config
    set_config_override(None)


@pytest.mark.asyncio
async def test_reports_free_mode_without_leaking_the_key(config):
    payload = await get_ai_providers(
        build_request({"x-supoclip-user-id": "user-1"})
    )

    assert payload["mode"] == "free"
    assert payload["reasoning"]["provider"] == "agnes"
    assert payload["requires_paid_provider"] is False
    assert payload["transcription"]["provider"] == "faster_whisper"
    assert payload["transcription_settings"]["engine"] == "faster_whisper"
    assert payload["usage"]["scope"] == "process"
    assert "agnes-secret-value" not in str(payload)


@pytest.mark.asyncio
async def test_requires_authentication(config):
    with pytest.raises(HTTPException) as caught:
        await get_ai_providers(build_request())

    assert caught.value.status_code in {401, 503}
