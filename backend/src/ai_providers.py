"""Provider catalogue, processing modes and paid-provider protection.

SupoClip can reason about a transcript with several models and transcribe audio
with several engines.  This module is the single source of truth for

* which providers exist and whether they are free, local or paid,
* what ``AI_PROVIDER_MODE`` means,
* when a paid provider is allowed to run, and
* which free alternative may replace a provider that ran out of quota.

It intentionally has no import of :mod:`src.config` at module load time so both
``config`` and the AI services can depend on it without a cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config

logger = logging.getLogger(__name__)

ProviderTier = Literal["free", "local", "paid"]
ProviderKind = Literal["reasoning", "transcription"]

AI_MODES: tuple[str, ...] = ("free", "premium", "local", "custom")

MODE_LABELS: dict[str, str] = {
    "free": "Free mode",
    "premium": "Premium mode",
    "local": "Local mode",
    "custom": "Custom mode",
}

MODE_DESCRIPTIONS: dict[str, str] = {
    "free": (
        "Agnes AI for clip reasoning, Faster-Whisper for local transcription, "
        "FFmpeg and OpenCV/MediaPipe for media processing. No paid API key required."
    ),
    "premium": (
        "Use the paid provider you selected (OpenAI, Google Gemini, Anthropic, "
        "OpenRouter or AssemblyAI). Selecting the provider is your authorization."
    ),
    "local": (
        "Ollama for reasoning and local Whisper for transcription. No external AI "
        "API calls at all."
    ),
    "custom": (
        "Mix providers independently (for example Agnes for reasoning and "
        "AssemblyAI for transcription)."
    ),
}


class AIProviderError(RuntimeError):
    """Base class for AI provider problems that a user can act on."""


class AIProviderConfigurationError(AIProviderError):
    """A provider is selected but missing credentials or a model name."""


class PaidProviderNotAuthorized(AIProviderError):
    """A paid provider was requested without the user opting in."""


class AIProviderQuotaExhausted(AIProviderError):
    """The provider refused the request because quota or balance ran out."""

    def __init__(self, message: str, *, provider_id: str, status_code: int = 402):
        super().__init__(message)
        self.provider_id = provider_id
        self.status_code = status_code


class AIProviderUnavailable(AIProviderError):
    """The provider timed out or returned a server-side failure."""

    def __init__(self, message: str, *, provider_id: str, status_code: int | None = None):
        super().__init__(message)
        self.provider_id = provider_id
        self.status_code = status_code


class AICapacityError(AIProviderError):
    """Every provider that could serve this operation is unavailable.

    The pipeline stops instead of silently spending money, and the message tells
    the operator exactly which switch to flip.
    """


@dataclass(frozen=True)
class ReasoningProvider:
    id: str
    label: str
    tier: ProviderTier
    description: str
    api_key_env: str | None = None
    default_model: str | None = None
    model_example: str | None = None


@dataclass(frozen=True)
class TranscriptionProvider:
    id: str
    label: str
    tier: ProviderTier
    description: str
    api_key_env: str | None = None


# ``google`` and ``google-gla`` are the two prefixes pydantic-ai accepts for the
# same Google GenAI key; both stay listed so existing ``LLM`` values keep working.
REASONING_PROVIDERS: tuple[ReasoningProvider, ...] = (
    ReasoningProvider(
        id="agnes",
        label="Agnes AI",
        tier="free",
        description=(
            "Default free reasoning provider. OpenAI-compatible chat completions "
            "at apihub.agnes-ai.com."
        ),
        api_key_env="AGNES_API_KEY",
        default_model="agnes-2.5-flash",
        model_example="agnes:agnes-2.5-flash",
    ),
    ReasoningProvider(
        id="ollama",
        label="Ollama (local)",
        tier="local",
        description="Local or self-hosted Ollama model. No API key, no per-request cost.",
        model_example="ollama:gpt-oss:20b",
    ),
    ReasoningProvider(
        id="openai",
        label="OpenAI",
        tier="paid",
        description="Premium reasoning provider.",
        api_key_env="OPENAI_API_KEY",
        model_example="openai:gpt-5.2",
    ),
    ReasoningProvider(
        id="google",
        label="Google Gemini",
        tier="paid",
        description="Premium reasoning provider (pydantic-ai google-gla).",
        api_key_env="GOOGLE_API_KEY",
        model_example="google-gla:gemini-3-flash-preview",
    ),
    ReasoningProvider(
        id="google-gla",
        label="Google Gemini",
        tier="paid",
        description="Premium reasoning provider (pydantic-ai google-gla).",
        api_key_env="GOOGLE_API_KEY",
        model_example="google-gla:gemini-3-flash-preview",
    ),
    ReasoningProvider(
        id="anthropic",
        label="Anthropic Claude",
        tier="paid",
        description="Premium reasoning provider.",
        api_key_env="ANTHROPIC_API_KEY",
        model_example="anthropic:claude-4-sonnet",
    ),
    ReasoningProvider(
        id="openrouter",
        label="OpenRouter",
        tier="paid",
        description="Premium router with vendor/model ids and fallback models.",
        api_key_env="OPENROUTER_API_KEY",
        model_example="openrouter:anthropic/claude-sonnet-5.5",
    ),
)

TRANSCRIPTION_PROVIDERS: tuple[TranscriptionProvider, ...] = (
    TranscriptionProvider(
        id="faster_whisper",
        label="Faster-Whisper (local)",
        tier="free",
        description=(
            "Default free transcription engine. CTranslate2 Whisper with "
            "word-level timestamps; CPU or CUDA."
        ),
    ),
    TranscriptionProvider(
        id="whisper",
        label="OpenAI Whisper (local)",
        tier="free",
        description="Reference openai-whisper implementation running locally.",
    ),
    TranscriptionProvider(
        id="youtube_captions",
        label="YouTube captions",
        tier="free",
        description="Free captions for YouTube sources only; no word-level timings.",
    ),
    TranscriptionProvider(
        id="assemblyai",
        label="AssemblyAI",
        tier="paid",
        description=(
            "Optional premium transcription with diarization. Never selected "
            "automatically and never used as a silent fallback."
        ),
        api_key_env="ASSEMBLY_AI_API_KEY",
    ),
)

REASONING_PROVIDER_IDS = frozenset(provider.id for provider in REASONING_PROVIDERS)
TRANSCRIPTION_PROVIDER_IDS = frozenset(provider.id for provider in TRANSCRIPTION_PROVIDERS)
PAID_REASONING_PROVIDERS = frozenset(
    provider.id for provider in REASONING_PROVIDERS if provider.tier == "paid"
)
PAID_TRANSCRIPTION_PROVIDERS = frozenset(
    provider.id for provider in TRANSCRIPTION_PROVIDERS if provider.tier == "paid"
)
MODES_WITHOUT_PAID_ACCESS = frozenset({"free", "local"})


def normalize_ai_mode(value: str | None) -> str | None:
    """Return a canonical mode, or None for blank/auto."""
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    # Accept "free mode" / "premium-mode" spellings.
    if normalized.endswith("_mode"):
        normalized = normalized[: -len("_mode")]
    if normalized in {"", "auto"}:
        return None
    if normalized not in AI_MODES:
        raise ValueError(
            f"AI_PROVIDER_MODE must be one of {', '.join(AI_MODES)} (got '{value}')."
        )
    return normalized


def normalize_transcription_provider(value: str | None) -> str | None:
    """Canonical transcription provider id, or None when unset/auto.

    Unknown values return ``None`` so the mode default applies instead of
    silently transcribing with an engine the operator did not ask for.
    """
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "auto"}:
        return None
    if normalized in {"fasterwhisper", "faster_whisper", "fw"}:
        return "faster_whisper"
    if normalized in TRANSCRIPTION_PROVIDER_IDS:
        return normalized
    logger.warning(
        "Unrecognised TRANSCRIPTION_PROVIDER '%s'; using the mode default", value
    )
    return None


def split_model_name(model_name: str) -> tuple[str, str | None]:
    """Split ``provider:model`` (model names may themselves contain colons)."""
    if ":" not in model_name:
        return model_name.strip().lower(), None
    provider, provider_model_name = model_name.split(":", 1)
    return provider.strip().lower(), provider_model_name.strip() or None


def provider_of_model(model_name: str) -> str:
    return split_model_name(model_name)[0]


def reasoning_provider(model_name: str) -> ReasoningProvider | None:
    provider_id = provider_of_model(model_name)
    for provider in REASONING_PROVIDERS:
        if provider.id == provider_id:
            return provider
    return None


def transcription_provider(provider_id: str) -> TranscriptionProvider | None:
    for provider in TRANSCRIPTION_PROVIDERS:
        if provider.id == provider_id:
            return provider
    return None


def provider_label(model_name: str) -> str:
    """Human-readable label for a ``provider:model`` string."""
    provider = reasoning_provider(model_name)
    return provider.label if provider else provider_of_model(model_name)


def is_paid_reasoning_provider(model_name: str) -> bool:
    return provider_of_model(model_name) in PAID_REASONING_PROVIDERS


def tier_of_reasoning_model(model_name: str) -> ProviderTier:
    provider = reasoning_provider(model_name)
    return provider.tier if provider else "paid"


def default_transcription_provider_for_mode(mode: str) -> str:
    """Free and local modes transcribe locally; premium/custom keep AssemblyAI."""
    if mode in {"free", "local"}:
        return "faster_whisper"
    return "assemblyai"


def default_reasoning_model_for_mode(mode: str, config: "Config") -> str | None:
    """Mode default used when ``LLM`` is not explicitly set."""
    if mode == "free":
        if config.ollama_model and not config.agnes_api_key:
            # A free tier that is explicitly configured for a local model may
            # still run without an Agnes key; the policy reports the reason.
            return f"ollama:{config.ollama_model}"
        return f"agnes:{config.agnes_model}"
    if mode == "local":
        if config.ollama_model:
            return f"ollama:{config.ollama_model}"
        return None
    return None


class ProviderPolicy:
    """Answer "may this provider run?" for the configured mode.

    The policy never mutates configuration and never contacts a provider, so it
    is safe to call from request handlers, workers and tests.
    """

    def __init__(self, config: "Config"):
        self.config = config

    # -- mode ---------------------------------------------------------------
    @property
    def mode(self) -> str:
        # Configs built by integrations or tests may not carry the newer
        # attributes; an explicit legacy configuration behaves like custom mode.
        return getattr(self.config, "ai_provider_mode", None) or "custom"

    @property
    def reasoning_model(self) -> str:
        return getattr(self.config, "llm", "") or ""

    @property
    def transcription_provider(self) -> str:
        return getattr(self.config, "transcription_provider", "assemblyai")

    @property
    def requires_paid_provider(self) -> bool:
        """Whether the resolved configuration spends money per request."""
        return (
            is_paid_reasoning_provider(self.reasoning_model)
            or self.transcription_provider in PAID_TRANSCRIPTION_PROVIDERS
        )

    # -- authorization ------------------------------------------------------
    @property
    def paid_authorized(self) -> bool:
        """True when the operator explicitly opted into paid providers."""
        if getattr(self.config, "allow_paid_ai_providers", False):
            return True
        return self.mode in {"premium", "custom"}

    def requires_paid_authorization(self, provider_id: str) -> bool:
        return provider_id in PAID_REASONING_PROVIDERS or provider_id in PAID_TRANSCRIPTION_PROVIDERS

    def assert_paid_authorization(self, provider_id: str, *, purpose: str) -> None:
        """Raise unless a paid provider may be used for ``purpose``."""
        if not self.requires_paid_authorization(provider_id):
            return
        if self.paid_authorized:
            return
        raise PaidProviderNotAuthorized(
            f"{purpose} would use the paid provider '{provider_id}', but the platform is in "
            f"{MODE_LABELS.get(self.mode, self.mode)} and no explicit authorization is set. "
            "SupoClip never spends paid credits without your consent. Select premium mode "
            "(AI_PROVIDER_MODE=premium) with that provider's API key, or set "
            "AI_ALLOW_PAID_PROVIDERS=true."
        )

    # -- free fallback ------------------------------------------------------
    def free_fallback_model(self) -> Optional[str]:
        """An eligible free local reasoning model, when one is configured."""
        if not getattr(self.config, "ai_free_fallback_enabled", True):
            return None
        if self.mode in {"premium", "custom"} and not self.paid_authorized:
            return None
        current = provider_of_model(self.config.llm)
        ollama_model = (getattr(self.config, "ollama_model", None) or "").strip()
        if not ollama_model:
            return None
        candidate = f"ollama:{ollama_model}"
        if provider_of_model(candidate) == current:
            return None
        return candidate

    def capacity_error(self, primary_error: AIProviderError) -> AICapacityError:
        """Explain how to continue after a provider ran out of capacity."""
        provider_id = getattr(primary_error, "provider_id", None) or provider_of_model(
            self.reasoning_model
        )
        suggestions = []
        if not (getattr(self.config, "ollama_model", None) or "").strip():
            suggestions.append(
                "set OLLAMA_MODEL (for example OLLAMA_MODEL=gpt-oss:20b) for a free local fallback"
            )
        if getattr(self.config, "agnes_api_key", None):
            suggestions.append("wait for the Agnes free-tier quota window to reset")
        if self.paid_authorized:
            suggestions.append("retry once the premium provider is available again")
        else:
            suggestions.append(
                "add a premium provider key (OPENAI_API_KEY, GOOGLE_API_KEY or "
                "ANTHROPIC_API_KEY) and set AI_PROVIDER_MODE=premium"
            )
        return AICapacityError(
            f"AI clip selection could not run: the {provider_id} provider is out of capacity "
            f"({primary_error}). No paid provider was used automatically. To continue, "
            + "; or ".join(suggestions)
            + "."
        )

    # -- reporting ----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """A secret-free snapshot for the settings API and the UI."""
        config = self.config
        selected_reasoning = provider_of_model(self.reasoning_model)
        warnings: list[str] = []

        reasoning_meta = reasoning_provider(self.reasoning_model)
        if reasoning_meta is None:
            warnings.append(
                f"The configured LLM '{self.reasoning_model}' is not a recognised provider."
            )
        if selected_reasoning == "agnes" and not config.agnes_api_key:
            warnings.append(
                "Agnes AI is selected but AGNES_API_KEY is not set, so AI clip selection "
                "will fail until a key is added (free keys: platform.agnes-ai.com)."
            )
        if selected_reasoning == "ollama" and not (
            getattr(config, "ollama_model", None) or ""
        ).strip():
            warnings.append(
                "A local Ollama model is selected without OLLAMA_MODEL; make sure the "
                "OLLAMA_BASE_URL endpoint serves the model name in LLM."
            )
        if self.transcription_provider == "assemblyai" and not getattr(
            config, "assembly_ai_api_key", None
        ):
            warnings.append(
                "AssemblyAI is selected but ASSEMBLY_AI_API_KEY is not set. Set the key, "
                "or switch TRANSCRIPTION_PROVIDER to faster_whisper for free local transcription."
            )
        if (
            self.transcription_provider in {"faster_whisper", "whisper"}
            and not config.transcription_local_available()
        ):
            warnings.append(
                f"The local transcription engine '{self.transcription_provider}' is not "
                "installed in this deployment; install the backend extra or choose another "
                "TRANSCRIPTION_PROVIDER."
            )
        if self.mode in MODES_WITHOUT_PAID_ACCESS:
            unpaid_paid = [
                provider.id
                for provider in REASONING_PROVIDERS
                if provider.tier == "paid"
                and provider.api_key_env
                and getattr(config, provider.api_key_env.lower(), None)
            ]
            if unpaid_paid:
                warnings.append(
                    "Paid provider credentials are present but the platform is in "
                    f"{MODE_LABELS[self.mode]}; they will not be used without "
                    "AI_PROVIDER_MODE=premium or AI_ALLOW_PAID_PROVIDERS=true."
                )

        reasoning_providers = []
        for provider in REASONING_PROVIDERS:
            key_present = bool(
                getattr(config, provider.api_key_env.lower(), None)
                if provider.api_key_env
                else provider.id == "ollama"
            )
            reasoning_providers.append(
                {
                    "id": provider.id,
                    "label": provider.label,
                    "tier": provider.tier,
                    "kind": "reasoning",
                    "description": provider.description,
                    "model_example": provider.model_example,
                    "configured": key_present or provider.id == "ollama",
                    "selected": provider.id == selected_reasoning,
                    "paid": provider.tier == "paid",
                    "authorized": provider.tier != "paid" or self.paid_authorized,
                }
            )

        transcription_providers = []
        for transcription_meta in TRANSCRIPTION_PROVIDERS:
            if transcription_meta.tier == "paid":
                configured = bool(config.assembly_ai_api_key)
            elif transcription_meta.id == "youtube_captions":
                configured = True
            else:
                configured = config.transcription_local_available()
            transcription_providers.append(
                {
                    "id": transcription_meta.id,
                    "label": transcription_meta.label,
                    "tier": transcription_meta.tier,
                    "kind": "transcription",
                    "description": transcription_meta.description,
                    "configured": configured,
                    "selected": transcription_meta.id == self.transcription_provider,
                    "paid": transcription_meta.tier == "paid",
                    "authorized": transcription_meta.tier != "paid" or self.paid_authorized,
                }
            )

        fallback_model = self.free_fallback_model()
        selected_transcription = transcription_provider(self.transcription_provider)
        return {
            "mode": self.mode,
            "mode_label": MODE_LABELS.get(self.mode, self.mode),
            "mode_description": MODE_DESCRIPTIONS.get(self.mode, ""),
            "modes": [
                {
                    "id": mode,
                    "label": MODE_LABELS[mode],
                    "description": MODE_DESCRIPTIONS[mode],
                    "selected": mode == self.mode,
                }
                for mode in AI_MODES
            ],
            "reasoning": {
                "provider": selected_reasoning,
                "model": self.reasoning_model,
                "label": reasoning_meta.label if reasoning_meta else selected_reasoning,
                "tier": reasoning_meta.tier if reasoning_meta else "paid",
                "configured": reasoning_meta is not None
                and (
                    not reasoning_meta.api_key_env
                    or bool(getattr(config, reasoning_meta.api_key_env.lower(), None))
                ),
            },
            "transcription": {
                "provider": self.transcription_provider,
                "label": (
                    selected_transcription.label
                    if selected_transcription
                    else self.transcription_provider
                ),
                "tier": (
                    selected_transcription.tier if selected_transcription else "paid"
                ),
                "configured": self.transcription_provider
                not in {"faster_whisper", "whisper"}
                or config.transcription_local_available(),
            },
            "requires_paid_provider": (
                is_paid_reasoning_provider(self.reasoning_model)
                or self.transcription_provider in PAID_TRANSCRIPTION_PROVIDERS
            ),
            "paid_providers_authorized": self.paid_authorized,
            "free_fallback": {
                "provider": provider_of_model(fallback_model) if fallback_model else None,
                "model": fallback_model,
                "configured": fallback_model is not None,
            },
            "providers": [*reasoning_providers, *transcription_providers],
            "warnings": warnings,
        }


def get_provider_policy(config: "Config | None" = None) -> ProviderPolicy:
    if config is None:
        from .config import get_config

        config = get_config()
    return ProviderPolicy(config)
