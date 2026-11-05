"""Agnes AI service: model factory, chat completions, error mapping, usage."""

from __future__ import annotations

import httpx
import pytest

from src import agnes_ai
from src.ai_providers import (
    AIProviderConfigurationError,
    AIProviderError,
    AIProviderQuotaExhausted,
    AIProviderUnavailable,
)
from src.ai_usage import reset_usage_tracker, usage_snapshot
from src.config import Config


@pytest.fixture(autouse=True)
def _clean_usage():
    reset_usage_tracker()
    yield
    reset_usage_tracker()


def make_config(monkeypatch, **env) -> Config:
    """Config with a free-tier Agnes key and no inherited legacy settings."""
    for name in (
        "LLM",
        "TRANSCRIPTION_PROVIDER",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "ASSEMBLY_AI_API_KEY",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_PROVIDER_MODE", "free")
    monkeypatch.setenv("AGNES_API_KEY", "agnes-test-key")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Config()


def completion_payload(content: str = "Hello from Agnes", model: str = "agnes-2.5-flash"):
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def service_with_transport(config: Config, handler) -> agnes_ai.AgnesAIService:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return agnes_ai.AgnesAIService(config, http_client=client)


def test_agnes_is_the_free_mode_default_llm(monkeypatch):
    config = make_config(monkeypatch)

    assert config.ai_provider_mode == "free"
    assert config.llm == "agnes:agnes-2.5-flash"
    assert config.transcription_provider == "faster_whisper"


def test_build_model_points_at_the_documented_base_url(monkeypatch):
    config = make_config(monkeypatch)

    model = agnes_ai.build_agnes_model(config)

    assert model.model_name == "agnes-2.5-flash"
    assert str(model.provider.client.base_url).rstrip("/") == "https://apihub.agnes-ai.com/v1"
    assert model.provider.client.api_key == "agnes-test-key"
    # Native JSON-schema output is not documented for Agnes; tool calling is.
    assert model.profile.supports_json_schema_output is False
    assert model.profile.supports_tools is True


def test_build_model_rejects_missing_key(monkeypatch):
    config = make_config(monkeypatch)
    config.agnes_api_key = None

    with pytest.raises(AIProviderConfigurationError, match="AGNES_API_KEY"):
        agnes_ai.build_agnes_model(config)


def test_model_name_and_base_url_are_configurable(monkeypatch):
    config = make_config(
        monkeypatch,
        AGNES_MODEL="agnes-3.0-flash",
        AGNES_BASE_URL="https://gateway.example/v1/",
    )

    model = agnes_ai.build_agnes_model(config)

    assert model.model_name == "agnes-3.0-flash"
    assert str(model.provider.client.base_url).rstrip("/") == "https://gateway.example/v1"


@pytest.mark.asyncio
async def test_chat_completion_parses_content_and_records_usage(monkeypatch):
    config = make_config(monkeypatch)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read().decode()
        return httpx.Response(200, json=completion_payload("  A caption.  "))

    service = service_with_transport(config, handler)
    try:
        completion = await service.chat(
            [{"role": "user", "content": "Hi"}], max_tokens=64, source="test"
        )
    finally:
        await service.aclose()

    assert completion.content == "A caption."
    assert completion.usage["total_tokens"] == 18
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer agnes-test-key"
    assert '"max_tokens":64' in seen["body"]

    snapshot = usage_snapshot()
    assert snapshot["entries"] == [
        {
            "provider": "agnes",
            "model": "agnes-2.5-flash",
            "tier": "free",
            "requests": 1,
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "errors": 0,
        }
    ]


@pytest.mark.asyncio
async def test_content_blocks_are_joined(monkeypatch):
    config = make_config(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "agnes-2.5-flash",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Part one. "},
                                {"type": "text", "text": "Part two."},
                            ],
                        }
                    }
                ],
            },
        )

    service = service_with_transport(config, handler)
    try:
        completion = await service.complete_text("system", "prompt")
    finally:
        await service.aclose()

    assert completion == "Part one. Part two."


@pytest.mark.parametrize(
    "status,expected,message",
    [
        (402, AIProviderQuotaExhausted, "quota"),
        (429, AIProviderQuotaExhausted, "rate limit"),
        (401, AIProviderConfigurationError, "AGNES_API_KEY"),
        (403, AIProviderConfigurationError, "rejected the credentials"),
        (404, AIProviderConfigurationError, "AGNES_BASE_URL"),
        (400, AIProviderError, "rejected the request"),
        (500, AIProviderUnavailable, "temporarily unavailable"),
        (503, AIProviderUnavailable, "temporarily unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_error_responses_map_to_actionable_errors(
    monkeypatch, status, expected, message
):
    config = make_config(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "provider detail"}})

    service = service_with_transport(config, handler)
    try:
        with pytest.raises(expected, match=message):
            await service.chat([{"role": "user", "content": "Hi"}])
    finally:
        await service.aclose()

    # Failures are counted separately so the UI can flag a broken provider.
    assert usage_snapshot()["entries"][0]["errors"] == 1
    if status in {402, 429}:
        assert usage_snapshot()["entries"][0]["requests"] == 0


@pytest.mark.asyncio
async def test_network_failures_are_reported_as_unavailable(monkeypatch):
    config = make_config(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    service = service_with_transport(config, handler)
    try:
        with pytest.raises(AIProviderUnavailable, match="could not be reached"):
            await service.chat([{"role": "user", "content": "Hi"}])
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_response_without_content_is_an_error(monkeypatch):
    config = make_config(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    service = service_with_transport(config, handler)
    try:
        with pytest.raises(AIProviderUnavailable, match="without assistant content"):
            await service.chat([{"role": "user", "content": "Hi"}])
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_chat_without_key_raises_configuration_error(monkeypatch):
    config = make_config(monkeypatch)
    config.agnes_api_key = None
    service = agnes_ai.AgnesAIService(config)

    with pytest.raises(AIProviderConfigurationError, match="not configured"):
        await service.chat([{"role": "user", "content": "Hi"}])


def test_provider_error_message_is_truncated_to_avoid_dumping_bodies():
    error = agnes_ai.map_agnes_error(400, {"error": {"message": "x" * 500}})

    assert len(str(error)) < 400
    assert "x" * 201 not in str(error)
    assert "x" * 200 in str(error)
