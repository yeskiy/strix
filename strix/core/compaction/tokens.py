"""Context-window lookup and calibrated token estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_OUTPUT = 16000
CHARS_PER_TOKEN = 4

# Context windows for models litellm has not cataloged yet. Keys are lowercase
# and match the normalized model name (and its provider-stripped suffix).
KNOWN_CONTEXT_WINDOWS = {
    "z-ai/glm-5.2": 1_048_576,
}


def _normalize_model_name(model: str) -> str:
    name = model.strip().lower()
    for prefix in ("litellm/", "any-llm/", "openai/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _model_cost_entry(model: str) -> dict[str, Any] | None:
    import litellm

    name = _normalize_model_name(model)
    entry = litellm.model_cost.get(name)
    if entry is None and "/" in name:
        entry = litellm.model_cost.get(name.rsplit("/", 1)[1])
    return entry if isinstance(entry, dict) else None


def _known_context_window(model: str) -> int | None:
    name = _normalize_model_name(model)
    candidates = [name]
    if "/" in name:
        candidates.append(name.split("/", 1)[1])
    for candidate in candidates:
        value = KNOWN_CONTEXT_WINDOWS.get(candidate)
        if value is not None:
            return value
    return None


def context_window(model: str, override: int | None = None) -> int:
    if override is not None and override > 0:
        return override
    known = _known_context_window(model)
    if known is not None:
        return known
    entry = _model_cost_entry(model)
    if entry is None:
        return DEFAULT_CONTEXT_WINDOW
    for key in ("max_input_tokens", "max_tokens"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_CONTEXT_WINDOW


def max_output_for(model: str) -> int:
    entry = _model_cost_entry(model)
    if entry is None:
        return DEFAULT_MAX_OUTPUT
    for key in ("max_output_tokens", "max_tokens"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_MAX_OUTPUT


def char_count(items: list[Any]) -> int:
    return len(json.dumps(items, ensure_ascii=False, default=str))


def tokens_from_chars(chars: int, ratio: float | None = None) -> int:
    effective = ratio if ratio is not None and ratio > 0 else 1.0 / CHARS_PER_TOKEN
    return int(chars * effective)


def estimate_tokens(items: list[Any], ratio: float | None = None) -> int:
    return tokens_from_chars(char_count(items), ratio)


@dataclass
class Calibrator:
    ratio: float | None = None

    def update(self, *, real_tokens: int, chars: int) -> None:
        if real_tokens > 0 and chars > 0:
            self.ratio = real_tokens / chars

    def estimate(self, items: list[Any]) -> int:
        return estimate_tokens(items, self.ratio)
