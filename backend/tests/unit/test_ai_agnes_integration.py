"""Agnes AI wired into the transcript-analysis agents, plus quota protection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from src import ai
from src.ai_providers import (
    AICapacityError,
    AIProviderQuotaExhausted,
    AIProviderUnavailable,
    PaidProviderNotAuthorized,
)
from src.ai_usage import reset_usage_tracker, usage_snapshot
from src.config import Config, set_config_override

TRANSCRIPT = "[00:00 - 00:30] A useful standalone explanation of a common mistake.\n[00:30 - 01:00] And the fix nobody tells you about."

PROVIDER_ENV = (
    "AI_PROVIDER_MODE",
    "AI_ALLOW_PAID_PROVIDERS",
    "AI_FREE_FALLBACK_ENABLED",
    "LLM",
    "TRANSCRIPTION_PROVIDER",
    "AGNES_API_KEY",
    "ASSEMBLY_AI_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_agents():
    """Agent instances are cached globally; keep tests from sharing them."""
    reset_usage_tracker()
    saved = (
        ai._transcript_agent,
        ai._transcript_agent_signature,
        ai._rerank_agent,
        ai._rerank_agent_signature,
        ai._broll_agent,
        ai._broll_agent_signature,
    )
    ai._transcript_agent = ai._transcript_agent_signature = None
    ai._rerank_agent = ai._rerank_agent_signature = None
    ai._broll_agent = ai._broll_agent_signature = None
    yield
    (
        ai._transcript_agent,
        ai._transcript_agent_signature,
        ai._rerank_agent,
        ai._rerank_agent_signature,
        ai._broll_agent,
        ai._broll_agent_signature,
    ) = saved
    set_config_override(None)
    reset_usage_tracker()


def free_config(monkeypatch, **env) -> Config:
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_PROVIDER_MODE", "free")
    monkeypatch.setenv("AGNES_API_KEY", "agnes-key")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = Config()
    set_config_override(config)
    return config


def analysis_result():
    return ai.TranscriptAnalysis(
        most_relevant_segments=[
            ai.TranscriptSegment(
                start_time="00:00",
                end_time="00:30",
                text="A useful standalone explanation of a common mistake.",
                hook_title="The mistake nobody notices",
            )
        ],
        summary="A common mistake.",
        key_topics=["mistakes"],
    )


def test_agnes_is_a_supported_provider():
    assert "agnes" in ai.SUPPORTED_LLM_PROVIDERS
    assert ai._split_llm_name("agnes:agnes-2.5-flash") == ("agnes", "agnes-2.5-flash")


def test_default_agent_is_built_on_agnes(monkeypatch):
    free_config(monkeypatch)

    agent = ai.get_transcript_agent()

    model = agent.model
    assert model.model_name == "agnes-2.5-flash"
    assert str(model.provider.client.base_url).rstrip("/") == "https://apihub.agnes-ai.com/v1"


def test_missing_agnes_key_has_an_actionable_message(monkeypatch):
    config = free_config(monkeypatch)
    config.agnes_api_key = None

    error = ai._get_missing_llm_key_error("agnes:agnes-2.5-flash", config)

    assert "AGNES_API_KEY" in error
    assert "OLLAMA_MODEL" in error
    assert "AI_PROVIDER_MODE=premium" in error


def test_paid_model_is_refused_in_free_mode(monkeypatch):
    config = free_config(monkeypatch)
    config.llm = "openai:gpt-5.2"
    config.openai_api_key = "openai-key"

    with pytest.raises(PaidProviderNotAuthorized, match="never spends paid credits"):
        ai._build_transcript_model(config)


def test_paid_model_runs_when_authorized(monkeypatch):
    config = free_config(monkeypatch, AI_ALLOW_PAID_PROVIDERS="true")
    config.llm = "openai:gpt-5.2"
    config.openai_api_key = "openai-key"

    # The factory builds a pydantic-ai model for the paid provider; no request is
    # made here, so no credits are spent by constructing it.
    assert ai._build_transcript_model(config) is not None


@pytest.mark.asyncio
async def test_quota_exhaustion_is_not_retried(monkeypatch):
    free_config(monkeypatch)
    agent = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(402, "agnes-2.5-flash")))
    monkeypatch.setattr(ai, "get_transcript_agent", lambda *args, **kwargs: agent)
    sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", sleep)

    with pytest.raises(AIProviderQuotaExhausted, match="did not switch to a paid provider"):
        await ai._run_transcript_analysis(agent, "prompt")

    assert agent.run.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_usage_is_recorded(monkeypatch):
    free_config(monkeypatch)
    result = SimpleNamespace(
        output=analysis_result(),
        response=SimpleNamespace(model_name="agnes-2.5-flash"),
        usage=lambda: SimpleNamespace(input_tokens=120, output_tokens=40, total_tokens=160),
    )
    agent = SimpleNamespace(run=AsyncMock(return_value=result))
    monkeypatch.setattr(ai, "get_transcript_agent", lambda *args, **kwargs: agent)

    await ai.get_most_relevant_parts_by_transcript(TRANSCRIPT)

    entries = usage_snapshot()["entries"]
    assert entries[0]["provider"] == "agnes"
    assert entries[0]["tier"] == "free"
    assert entries[0]["total_tokens"] == 160


@pytest.mark.asyncio
async def test_quota_exhaustion_uses_the_configured_free_local_fallback(monkeypatch):
    free_config(monkeypatch, OLLAMA_MODEL="gpt-oss:20b")
    primary = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(402, "agnes-2.5-flash")))
    fallback = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(output=analysis_result())))
    used_models = []

    def factory(clip_length="auto", model_name=None):
        used_models.append(model_name)
        return fallback if model_name else primary

    monkeypatch.setattr(ai, "get_transcript_agent", factory)
    monkeypatch.setattr(ai, "get_rerank_agent", lambda *args, **kwargs: fallback)

    result = await ai.get_most_relevant_parts_by_transcript(TRANSCRIPT)

    assert used_models == [None, "ollama:gpt-oss:20b"]
    assert len(result.most_relevant_segments) == 1
    assert fallback.run.await_count == 1


@pytest.mark.asyncio
async def test_quota_exhaustion_without_a_fallback_stops_with_an_explanation(monkeypatch):
    free_config(monkeypatch)
    agent = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(402, "agnes-2.5-flash")))
    monkeypatch.setattr(ai, "get_transcript_agent", lambda *args, **kwargs: agent)

    with pytest.raises(AICapacityError) as caught:
        await ai.get_most_relevant_parts_by_transcript(TRANSCRIPT)

    message = str(caught.value)
    assert "out of capacity" in message
    assert "OLLAMA_MODEL" in message
    assert "AI_PROVIDER_MODE=premium" in message
    assert "No paid provider was used" in message


@pytest.mark.asyncio
async def test_failed_fallback_also_explains_itself(monkeypatch):
    free_config(monkeypatch, OLLAMA_MODEL="gpt-oss:20b")
    primary = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(402, "agnes-2.5-flash")))
    fallback = SimpleNamespace(run=AsyncMock(side_effect=ConnectionError("ollama down")))

    monkeypatch.setattr(
        ai, "get_transcript_agent", lambda clip_length="auto", model_name=None: fallback if model_name else primary
    )

    with pytest.raises(AICapacityError, match="out of capacity"):
        await ai.get_most_relevant_parts_by_transcript(TRANSCRIPT)


@pytest.mark.asyncio
async def test_provider_outage_also_triggers_the_free_fallback(monkeypatch):
    free_config(monkeypatch, OLLAMA_MODEL="gpt-oss:20b")
    primary = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(503, "agnes-2.5-flash")))
    fallback = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(output=analysis_result())))
    monkeypatch.setattr(
        ai, "get_transcript_agent", lambda clip_length="auto", model_name=None: fallback if model_name else primary
    )
    monkeypatch.setattr(ai, "get_rerank_agent", lambda *args, **kwargs: fallback)
    ui_sleep = AsyncMock()
    monkeypatch.setattr(ai.asyncio, "sleep", ui_sleep)

    # A 503 is retried first; after the retry budget the fallback model is used.
    result = await ai.get_most_relevant_parts_by_transcript(TRANSCRIPT)

    assert len(result.most_relevant_segments) == 1


def test_quota_errors_are_translated_but_retryable_ones_are_not(monkeypatch):
    config = free_config(monkeypatch)

    assert isinstance(
        ai._translate_provider_error(ModelHTTPError(402, "agnes-2.5-flash"), config),
        AIProviderQuotaExhausted,
    )
    for status in (400, 401, 403, 500, 429):
        error = ModelHTTPError(status, "agnes-2.5-flash")
        assert ai._translate_provider_error(error, config) is error


@pytest.mark.asyncio
async def test_rate_limit_surviving_all_retries_becomes_a_quota_error(monkeypatch):
    free_config(monkeypatch)
    monkeypatch.setattr(ai, "TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS", 2)
    agent = SimpleNamespace(run=AsyncMock(side_effect=ModelHTTPError(429, "agnes-2.5-flash")))
    monkeypatch.setattr(ai.asyncio, "sleep", AsyncMock())

    with pytest.raises(AIProviderQuotaExhausted, match="rate limiting"):
        await ai._run_transcript_analysis(agent, "prompt")


@pytest.mark.asyncio
async def test_unavailable_error_is_not_retried_when_it_is_our_own_type(monkeypatch):
    free_config(monkeypatch)
    agent = SimpleNamespace(run=AsyncMock(side_effect=AIProviderUnavailable("down", provider_id="agnes")))

    with pytest.raises(AIProviderUnavailable):
        await ai._run_transcript_analysis(agent, "prompt")

    assert agent.run.await_count == 1
