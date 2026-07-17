# tests/test_compaction_summarize.py
"""Tests for the summarizer LLM call (litellm mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.prompt import PREVIOUS_SUMMARY_OPEN
from strix.core.compaction.summarize import summarize_head


def _cfg(**over: Any) -> CompactionConfig:
    base = {
        "enabled": True,
        "threshold": 0.90,
        "hysteresis": 0.10,
        "reserved_output": 1000,
        "tail_turns": 2,
        "preserve_recent_tokens": 16000,
        "prune_enabled": True,
        "prune_protect_tokens": 40000,
        "prune_min_reclaim": 20000,
        "session_model": "openrouter/z-ai/glm-4.6",
        "summary_model": "litellm/openai/gpt-4o",
        "usable_window": 100000,
    }
    base.update(over)
    return CompactionConfig(**base)


def _fake_response(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.mark.asyncio
async def test_summarize_head_calls_litellm_and_returns_text() -> None:
    mock = AsyncMock(return_value=_fake_response("## Objective\n- test"))
    with patch("litellm.acompletion", mock):
        out = await summarize_head([{"role": "user", "content": "hi"}], None, _cfg())
    assert out == "## Objective\n- test"
    kwargs = mock.await_args.kwargs
    # normalized model: litellm/ prefix stripped
    assert kwargs["model"] == "openai/gpt-4o"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_previous_summary_included_when_present() -> None:
    mock = AsyncMock(return_value=_fake_response("SUM"))
    with patch("litellm.acompletion", mock):
        await summarize_head([{"role": "user", "content": "hi"}], "PRIOR SUMMARY", _cfg())
    user_content = mock.await_args.kwargs["messages"][1]["content"]
    assert PREVIOUS_SUMMARY_OPEN in user_content
    assert "PRIOR SUMMARY" in user_content


@pytest.mark.asyncio
async def test_empty_content_returns_empty_string() -> None:
    mock = AsyncMock(return_value=_fake_response(None))
    with patch("litellm.acompletion", mock):
        out = await summarize_head([{"role": "user", "content": "x"}], None, _cfg())
    assert out == ""


@pytest.mark.asyncio
async def test_summarize_head_openrouter_sets_middle_out() -> None:
    mock = AsyncMock(return_value=_fake_response("SUM"))
    with patch("litellm.acompletion", mock):
        await summarize_head(
            [{"role": "user", "content": "hi"}],
            None,
            _cfg(summary_model="openrouter/z-ai/glm-5.2"),
        )
    assert mock.await_args.kwargs.get("extra_body") == {"transforms": ["middle-out"]}


@pytest.mark.asyncio
async def test_summarize_head_non_openrouter_no_transform() -> None:
    mock = AsyncMock(return_value=_fake_response("SUM"))
    with patch("litellm.acompletion", mock):
        await summarize_head(
            [{"role": "user", "content": "hi"}],
            None,
            _cfg(summary_model="openai/gpt-4o"),
        )
    assert "extra_body" not in mock.await_args.kwargs
