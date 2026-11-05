"""Agnes AI integration.

Agnes AI exposes an OpenAI-compatible API:

* Base URL: ``https://apihub.agnes-ai.com/v1``
* Text endpoint: ``POST /v1/chat/completions``
* Auth: ``Authorization: Bearer YOUR_API_KEY``
* Default text model: ``agnes-2.5-flash`` (``agnes-2.0-flash`` is deprecated)
* Tool calling and streaming are supported; provider usage is reported in
  ``usage.prompt_tokens`` / ``usage.completion_tokens`` / ``usage.total_tokens``.

Documented error codes: 400 (bad request), 401 (bad key), 402 (quota or balance
exhausted), 403 (no access to the model), 404 (wrong base URL or model), 408 /
429 / 5xx (retry later).

Every Agnes call in SupoClip goes through this module so credentials, timeouts,
error translation and usage accounting stay in one place.  The model factory is
consumed by the pydantic-ai agents in :mod:`src.ai`; the HTTP client serves the
smaller text operations that do not need a schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import httpx

from .ai_providers import (
    AIProviderConfigurationError,
    AIProviderError,
    AIProviderQuotaExhausted,
    AIProviderUnavailable,
)
from .ai_usage import record_ai_error, record_ai_usage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai.models import Model

    from .config import Config

logger = logging.getLogger(__name__)

PROVIDER_ID = "agnes"
DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_MODEL = "agnes-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 600.0
# Agnes `usage` keys; the same names OpenAI-compatible APIs use.
QUOTA_STATUS_CODES = frozenset({402, 429})
AUTH_STATUS_CODES = frozenset({401, 403})
TRANSIENT_STATUS_CODES = frozenset({408, 409, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class AgnesCompletion:
    """One Agnes chat completion."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: Mapping[str, Any] | None = None


def agnes_is_configured(config: "Config") -> bool:
    return bool(getattr(config, "agnes_api_key", None))


def agnes_base_url(config: "Config") -> str:
    return (getattr(config, "agnes_base_url", "") or DEFAULT_BASE_URL).rstrip("/")


def agnes_model_name(config: "Config", model_name: str | None = None) -> str:
    return model_name or getattr(config, "agnes_model", None) or DEFAULT_MODEL


def build_agnes_model(config: "Config", model_name: str | None = None) -> "Model":
    """Build the pydantic-ai model that talks to Agnes.

    Structured output uses tool calling, which Agnes documents as supported.
    Native JSON-schema response format is not documented, so it stays disabled
    and pydantic-ai falls back to a tool result instead of a request the API may
    reject.
    """
    try:
        # Chat Completions model (Agnes is Chat-Completions compatible). Newer
        # pydantic-ai releases renamed OpenAIModel to OpenAIChatModel.
        from pydantic_ai.models.openai import OpenAIChatModel as _ChatModel
    except ImportError:  # pragma: no cover - older pydantic-ai
        from pydantic_ai.models.openai import OpenAIModel as _ChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    if not agnes_is_configured(config):
        raise AIProviderConfigurationError(
            "Agnes AI is selected but AGNES_API_KEY is not set. Create a free key at "
            "platform.agnes-ai.com and set AGNES_API_KEY, or switch AI_PROVIDER_MODE."
        )

    profile = OpenAIModelProfile(
        supports_json_schema_output=False,
        supports_json_object_output=False,
        # Agnes documents tool calling but not OpenAI's strict schema extension.
        openai_supports_strict_tool_definition=False,
        openai_supports_tool_choice_required=False,
    )
    return _ChatModel(
        agnes_model_name(config, model_name),
        provider=OpenAIProvider(
            base_url=agnes_base_url(config),
            api_key=config.agnes_api_key,
        ),
        profile=profile,
    )


def map_agnes_error(
    status_code: int, payload: Any = None, *, model: str = DEFAULT_MODEL
) -> AIProviderError:
    """Translate an Agnes HTTP failure into a typed, user-actionable error.

    Provider error bodies can echo request content, so the message is taken from
    ``error.message`` only and truncated.
    """
    detail = _error_message(payload)
    suffix = f" ({detail})" if detail else ""

    if status_code in AUTH_STATUS_CODES:
        return AIProviderConfigurationError(
            f"Agnes AI rejected the credentials (HTTP {status_code}){suffix}. Check "
            "AGNES_API_KEY, or confirm the key has access to the model "
            f"'{model}'."
        )
    if status_code == 402:
        return AIProviderQuotaExhausted(
            "Agnes AI free quota or account balance is exhausted (HTTP 402)"
            f"{suffix}. SupoClip did not switch to a paid provider automatically.",
            provider_id=PROVIDER_ID,
            status_code=status_code,
        )
    if status_code == 429:
        return AIProviderQuotaExhausted(
            f"Agnes AI rate limit reached (HTTP 429){suffix}. Wait for the limit "
            "window to reset; no paid provider was used.",
            provider_id=PROVIDER_ID,
            status_code=status_code,
        )
    if status_code == 404:
        return AIProviderConfigurationError(
            f"Agnes AI could not find the endpoint or model (HTTP 404){suffix}. "
            "Check AGNES_BASE_URL (https://apihub.agnes-ai.com/v1) and AGNES_MODEL."
        )
    if status_code == 400:
        return AIProviderError(
            f"Agnes AI rejected the request (HTTP 400){suffix}. The prompt may exceed "
            "the model context limit."
        )
    if status_code in TRANSIENT_STATUS_CODES or status_code >= 500:
        return AIProviderUnavailable(
            f"Agnes AI is temporarily unavailable (HTTP {status_code}){suffix}.",
            provider_id=PROVIDER_ID,
            status_code=status_code,
        )
    return AIProviderError(f"Agnes AI request failed (HTTP {status_code}){suffix}.")


class AgnesAIService:
    """The single Agnes AI entry point used by SupoClip."""

    def __init__(
        self,
        config: "Config",
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | None = None,
    ):
        self.config = config
        self._client = http_client
        self._owns_client = http_client is None
        self.timeout = timeout or getattr(
            config, "agnes_timeout_seconds", DEFAULT_TIMEOUT_SECONDS
        )

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_configured(self) -> bool:
        return agnes_is_configured(self.config)

    def build_model(self, model_name: str | None = None) -> "Model":
        return build_agnes_model(self.config, model_name)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- chat ---------------------------------------------------------------
    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        source: str = "reasoning",
    ) -> AgnesCompletion:
        """Call ``/v1/chat/completions`` and return the assistant text."""
        if not self.is_configured:
            raise AIProviderConfigurationError(
                "Agnes AI is not configured: AGNES_API_KEY is missing."
            )
        selected_model = agnes_model_name(self.config, model)
        request: dict[str, Any] = {
            "model": selected_model,
            "messages": list(messages),
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        url = f"{agnes_base_url(self.config)}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.agnes_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._get_client().post(
                url, json=request, headers=headers, timeout=self.timeout
            )
        except httpx.TimeoutException as error:
            record_ai_error(PROVIDER_ID, selected_model, tier="free")
            raise AIProviderUnavailable(
                f"Agnes AI did not respond within {self.timeout:.0f}s.",
                provider_id=PROVIDER_ID,
            ) from error
        except httpx.HTTPError as error:
            record_ai_error(PROVIDER_ID, selected_model, tier="free")
            raise AIProviderUnavailable(
                f"Agnes AI could not be reached ({type(error).__name__}).",
                provider_id=PROVIDER_ID,
            ) from error

        if response.status_code >= 400:
            record_ai_error(PROVIDER_ID, selected_model, tier="free")
            raise map_agnes_error(
                response.status_code, _safe_json(response), model=selected_model
            )

        payload = _safe_json(response) or {}
        content = _extract_content(payload)
        if content is None:
            record_ai_error(PROVIDER_ID, selected_model, tier="free")
            raise AIProviderUnavailable(
                "Agnes AI returned a response without assistant content.",
                provider_id=PROVIDER_ID,
            )

        usage = payload.get("usage") if isinstance(payload, dict) else None
        record_ai_usage(
            PROVIDER_ID,
            payload.get("model") or selected_model,
            usage if isinstance(usage, Mapping) else None,
            tier="free",
            source=source,
        )
        choice = (payload.get("choices") or [{}])[0] if payload.get("choices") else {}
        return AgnesCompletion(
            content=content.strip(),
            model=payload.get("model") or selected_model,
            finish_reason=choice.get("finish_reason"),
            usage=usage if isinstance(usage, Mapping) else None,
        )

    async def complete_text(
        self,
        system_prompt: str,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        source: str = "text",
    ) -> str:
        completion = await self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            source=source,
        )
        return completion.content


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _error_message(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    error = payload.get("error")
    if isinstance(error, Mapping):
        message = error.get("message") or error.get("code") or ""
    elif isinstance(error, str):
        message = error
    else:
        message = payload.get("message") or ""
    text = str(message).strip().replace("\n", " ")
    return text[:200]


def _extract_content(payload: Mapping[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") in {"text", "output_text"}
        ]
        joined = "".join(parts).strip()
        return joined or None
    return None
