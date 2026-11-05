"""Lightweight AI usage accounting.

Providers that report token usage are the only place SupoClip can observe real
consumption, so this module records what the provider told us and nothing is
estimated.  Totals are kept per process: the API pod and every ARQ worker keep
their own counters, and each record is also emitted as a structured ``ai_usage``
log line so a log pipeline can aggregate across processes.

These numbers are *not* billing amounts.  They are request and token counts the
provider returned, reported separately for free (Agnes) and paid providers.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class UsageTotals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.requests += 1
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)
        self.total_tokens += max(0, total_tokens)

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
        }


@dataclass
class UsageEntry:
    provider: str
    model: str
    tier: str
    totals: UsageTotals = field(default_factory=UsageTotals)


_lock = threading.Lock()
_entries: dict[tuple[str, str, str], UsageEntry] = {}


def record_ai_usage(
    provider: str,
    model: str,
    usage: Mapping[str, Any] | None,
    *,
    tier: str = "free",
    source: str = "reasoning",
) -> UsageTotals:
    """Record one provider call.

    ``usage`` is the provider's own usage object (prompt/completion tokens) or
    None when the provider did not report any.  The call is still counted.
    """
    payload = usage or {}
    prompt_tokens = _coerce_int(
        payload.get("prompt_tokens", payload.get("input_tokens", 0))
    )
    completion_tokens = _coerce_int(
        payload.get("completion_tokens", payload.get("output_tokens", 0))
    )
    total_tokens = _coerce_int(
        payload.get("total_tokens", prompt_tokens + completion_tokens)
    )
    key = (provider, model, tier)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            entry = UsageEntry(provider=provider, model=model, tier=tier)
            _entries[key] = entry
        entry.totals.add(prompt_tokens, completion_tokens, total_tokens)
        snapshot = entry.totals.as_dict()

    logger.info(
        "ai_usage provider=%s model=%s tier=%s source=%s prompt_tokens=%d "
        "completion_tokens=%d total_tokens=%d",
        provider,
        model,
        tier,
        source,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )
    return entry.totals


def record_ai_error(provider: str, model: str, *, tier: str = "free") -> None:
    """Count a failed call so the UI can show that a provider is failing."""
    key = (provider, model, tier)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            entry = UsageEntry(provider=provider, model=model, tier=tier)
            _entries[key] = entry
        entry.totals.errors += 1


def usage_snapshot() -> dict[str, Any]:
    """Secret-free per-provider counters for the current process."""
    with _lock:
        entries = [
            {
                "provider": entry.provider,
                "model": entry.model,
                "tier": entry.tier,
                **entry.totals.as_dict(),
            }
            for entry in _entries.values()
        ]
    entries.sort(key=lambda item: (item["tier"], item["provider"], item["model"]))
    return {
        "scope": "process",
        "note": (
            "Counts reported by providers since this process started. Not a billing "
            "statement; processes do not share counters."
        ),
        "entries": entries,
    }


def reset_usage_tracker() -> None:
    """Clear counters (tests and long-lived admin tooling only)."""
    with _lock:
        _entries.clear()


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
