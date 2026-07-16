# tests/test_compaction_filter.py
"""Tests for the call_model_input_filter orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from agents.run_config import CallModelData, ModelInputData

from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.filter import build_compaction_filter
from strix.core.compaction.store import CompactionStore


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
        "session_model": "openai/gpt-4o",
        "summary_model": "openai/gpt-4o",
        "usable_window": 1000,
    }
    base.update(over)
    return CompactionConfig(**base)


def _payload(items: list[Any], agent_id: str = "root") -> CallModelData[Any]:
    model_data = ModelInputData(input=items, instructions="SYS")
    agent = type("A", (), {"name": "strix"})()
    return CallModelData(model_data=model_data, agent=agent, context={"agent_id": agent_id})


@pytest.mark.asyncio
async def test_identity_pass_below_threshold_returns_same_object() -> None:
    store = CompactionStore()
    filt = build_compaction_filter(_cfg(usable_window=1_000_000), store)
    payload = _payload([{"role": "user", "content": "hi"}])
    result = await filt(payload)
    assert result is payload.model_data


@pytest.mark.asyncio
async def test_over_threshold_triggers_summary() -> None:
    store = CompactionStore()
    # small window so any real history trips it; big history so head is non-empty
    filt = build_compaction_filter(_cfg(usable_window=50, prune_enabled=False), store)
    items = [
        {"role": "user", "content": "t1 " + "x" * 500},
        {"role": "assistant", "content": "a1 " + "y" * 500},
        {"role": "user", "content": "t2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "t3"},
    ]
    payload = _payload(items)
    with patch(
        "strix.core.compaction.filter.summarize_head",
        AsyncMock(return_value="## Objective\n- ok"),
    ):
        result = await filt(payload)
    assert result.instructions == "SYS"
    assert result.input[0]["role"] == "user"
    assert result.input[0]["content"].startswith("Context summary of earlier work")
    # a summary is cached and the anchor advanced
    st = store.state("root")
    assert st.summary == "## Objective\n- ok"
    assert st.anchor > 0


@pytest.mark.asyncio
async def test_force_flag_triggers_compaction_then_clears() -> None:
    store = CompactionStore()
    filt = build_compaction_filter(_cfg(usable_window=1_000_000, prune_enabled=False), store)
    items = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "t3"},
    ]
    store.request_compaction("root")
    payload = _payload(items)
    with patch(
        "strix.core.compaction.filter.summarize_head",
        AsyncMock(return_value="SUMMARY"),
    ):
        result = await filt(payload)
    assert result.input[0]["content"].endswith("SUMMARY")
    assert store.consume_force("root") is False


@pytest.mark.asyncio
async def test_cached_summary_reused_below_threshold() -> None:
    store = CompactionStore()
    store.set_summary("root", "CACHED", 2)
    filt = build_compaction_filter(_cfg(usable_window=1_000_000), store)
    items = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
    ]
    mock = AsyncMock(return_value="NEW")
    with patch("strix.core.compaction.filter.summarize_head", mock):
        result = await filt(_payload(items))
    # summarizer not called; cached summary applied as byte-stable prefix
    mock.assert_not_awaited()
    assert result.input[0]["content"].endswith("CACHED")
    assert result.input[1:] == items[2:]


@pytest.mark.asyncio
async def test_summarizer_failure_degrades_gracefully() -> None:
    store = CompactionStore()
    filt = build_compaction_filter(_cfg(usable_window=50, prune_enabled=False), store)
    items = [
        {"role": "user", "content": "t1 " + "x" * 500},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "t3"},
    ]
    with patch(
        "strix.core.compaction.filter.summarize_head",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = await filt(_payload(items))
    # never raises; returns a valid ModelInputData
    assert isinstance(result, ModelInputData)
    assert result.instructions == "SYS"
