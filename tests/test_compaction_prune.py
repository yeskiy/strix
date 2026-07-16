"""Tests for the LLM-free tool-output prune stage."""

from __future__ import annotations

from typing import Any

from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.prune import PRUNED_PREFIX, prune_tool_outputs


def _cfg(**over: Any) -> CompactionConfig:
    base = {
        "enabled": True,
        "threshold": 0.90,
        "hysteresis": 0.10,
        "reserved_output": 1000,
        "tail_turns": 2,
        "preserve_recent_tokens": 16000,
        "prune_enabled": True,
        "prune_protect_tokens": 0,
        "prune_min_reclaim": 10,
        "session_model": "openai/gpt-4o",
        "summary_model": "openai/gpt-4o",
        "usable_window": 100000,
    }
    base.update(over)
    return CompactionConfig(**base)


def _tool_pair(call_id: str, name: str, output: str) -> list[dict[str, Any]]:
    return [
        {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def test_prunes_old_large_output_and_keeps_pairing() -> None:
    items: list[dict[str, Any]] = [
        {"role": "user", "content": "start"},
        *_tool_pair("c1", "browser", "A" * 4000),
        {"role": "user", "content": "turn2"},
        {"role": "user", "content": "turn3"},
    ]
    new_items, reclaimed = prune_tool_outputs(items, _cfg(), ratio=None)
    assert reclaimed > 0
    # the function_call is untouched, its output is now a placeholder
    assert new_items[1]["type"] == "function_call"
    assert new_items[2]["type"] == "function_call_output"
    assert new_items[2]["call_id"] == "c1"
    assert new_items[2]["output"].startswith(PRUNED_PREFIX)
    assert "browser" in new_items[2]["output"]


def test_protects_recent_tail_turns() -> None:
    items: list[dict[str, Any]] = [
        {"role": "user", "content": "start"},
        {"role": "user", "content": "turn2"},
        *_tool_pair("c1", "browser", "A" * 4000),
        {"role": "user", "content": "turn3"},
    ]
    # tail_turns=2 protects turn2 (index 1) onward, so the c1 output is protected
    new_items, reclaimed = prune_tool_outputs(items, _cfg(tail_turns=2), ratio=None)
    assert reclaimed == 0
    assert new_items[3]["output"] == "A" * 4000


def test_below_min_reclaim_is_noop() -> None:
    items: list[dict[str, Any]] = [
        {"role": "user", "content": "start"},
        *_tool_pair("c1", "browser", "A" * 40),
        {"role": "user", "content": "turn2"},
        {"role": "user", "content": "turn3"},
    ]
    new_items, reclaimed = prune_tool_outputs(items, _cfg(prune_min_reclaim=100000), ratio=None)
    assert reclaimed == 0
    assert new_items == items


def test_idempotent() -> None:
    items: list[dict[str, Any]] = [
        {"role": "user", "content": "start"},
        *_tool_pair("c1", "browser", "A" * 4000),
        {"role": "user", "content": "turn2"},
        {"role": "user", "content": "turn3"},
    ]
    once, _r1 = prune_tool_outputs(items, _cfg(), ratio=None)
    twice, r2 = prune_tool_outputs(once, _cfg(), ratio=None)
    assert r2 == 0
    assert twice == once


def test_prune_disabled_returns_input() -> None:
    items: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
    new_items, reclaimed = prune_tool_outputs(items, _cfg(prune_enabled=False), ratio=None)
    assert new_items is items
    assert reclaimed == 0
