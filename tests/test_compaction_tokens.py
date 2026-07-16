"""Tests for token accounting helpers."""

from __future__ import annotations

from strix.core.compaction.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    Calibrator,
    char_count,
    context_window,
    estimate_tokens,
    max_output_for,
    tokens_from_chars,
)


def test_context_window_known_model() -> None:
    assert context_window("openai/gpt-4o") == 128000
    assert context_window("gpt-4o") == 128000


def test_context_window_strips_routing_prefixes() -> None:
    assert context_window("litellm/gpt-4o") == 128000
    assert context_window("any-llm/openai/gpt-4o") == 128000


def test_context_window_unknown_model_falls_back() -> None:
    assert context_window("totally/unknown-model-xyz") == DEFAULT_CONTEXT_WINDOW


def test_context_window_known_override_map() -> None:
    assert context_window("z-ai/glm-5.2") == 1_048_576
    assert context_window("openrouter/z-ai/glm-5.2") == 1_048_576


def test_context_window_explicit_override_wins() -> None:
    assert context_window("some/unknown-model-xyz") == 128000
    assert context_window("some/unknown-model-xyz", override=500000) == 500000


def test_context_window_override_zero_or_none_falls_through() -> None:
    assert context_window("z-ai/glm-5.2", override=0) == 1_048_576
    assert context_window("z-ai/glm-5.2", override=None) == 1_048_576
    assert context_window("some/unknown-model-xyz", override=0) == DEFAULT_CONTEXT_WINDOW


def test_max_output_for_known_model() -> None:
    assert max_output_for("gpt-4o") == 16384


def test_char_and_token_estimate() -> None:
    items = [{"role": "user", "content": "x" * 400}]
    chars = char_count(items)
    assert chars >= 400
    # default heuristic is char / 4
    assert estimate_tokens(items) == tokens_from_chars(chars)
    assert estimate_tokens(items) == chars // 4


def test_calibrator_tracks_provider_ratio() -> None:
    cal = Calibrator()
    assert cal.ratio is None
    cal.update(real_tokens=1000, chars=2000)
    assert cal.ratio == 0.5
    items = [{"role": "user", "content": "x" * 800}]
    # with ratio 0.5 the estimate is chars * 0.5
    assert cal.estimate(items) == int(char_count(items) * 0.5)


def test_calibrator_ignores_empty_samples() -> None:
    cal = Calibrator()
    cal.update(real_tokens=0, chars=2000)
    assert cal.ratio is None
    cal.update(real_tokens=1000, chars=0)
    assert cal.ratio is None
