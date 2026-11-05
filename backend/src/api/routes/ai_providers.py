"""Read-only AI provider status for the settings UI.

Nothing here returns a credential: the response reports which providers exist,
which one is selected, whether a task would use a paid provider, and the usage
counters this process observed. Provider selection itself is operator
configuration (environment or the admin runtime-settings screen), so there is no
user-level "switch my key" endpoint that could leak or bill a shared secret.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ...ai_providers import get_provider_policy
from ...ai_usage import usage_snapshot
from ...auth_headers import get_authenticated_user_id
from ...config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/providers")
async def get_ai_providers(request: Request):
    """Current processing mode, providers, availability and usage."""
    config = get_config()
    # Requires a signed-in user; the answer is identical for everyone, but it
    # should not be world-readable.
    get_authenticated_user_id(request, config)
    policy = get_provider_policy(config)
    payload = policy.describe()
    payload["usage"] = usage_snapshot()
    payload["transcription_settings"] = {
        "engine": config.transcription_provider,
        "model_size": getattr(config, "faster_whisper_model", None),
        "device": getattr(config, "faster_whisper_device", None),
        "compute_type": getattr(config, "faster_whisper_compute_type", None),
        "cache_enabled": getattr(config, "transcription_cache_enabled", True),
    }
    return payload
