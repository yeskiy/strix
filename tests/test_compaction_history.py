"""Tests for turn grouping, head/tail split, and image stripping."""

from __future__ import annotations

from typing import Any

from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.history import (
    iter_turn_start_indices,
    split_head_tail,
    strip_images,
)


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
        "usable_window": 100000,
    }
    base.update(over)
    return CompactionConfig(**base)


def test_turn_start_indices() -> None:
    items = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert iter_turn_start_indices(items) == [0, 2]


def test_split_keeps_last_tail_turns() -> None:
    items = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "t3"},
        {"role": "assistant", "content": "a3"},
    ]
    head, tail = split_head_tail(items, _cfg(tail_turns=2))
    # tail starts at t2 (index 2)
    assert head == items[:2]
    assert tail == items[2:]


def test_split_guard_returns_empty_head_when_too_few_turns() -> None:
    items = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
    ]
    head, tail = split_head_tail(items, _cfg(tail_turns=2))
    assert head == []
    assert tail == items


def test_split_never_orphans_tool_output() -> None:
    # one oversized single tail turn: split must not start the tail on a
    # function_call_output whose function_call would land in head.
    items = [
        {"role": "user", "content": "t1"},
        {"role": "user", "content": "big"},
        {"type": "function_call", "call_id": "c1", "name": "browser", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "Z" * 200000},
        {"role": "assistant", "content": "done"},
    ]
    _head, tail = split_head_tail(items, _cfg(tail_turns=1, preserve_recent_tokens=10))
    # whatever the split, no function_call_output may be the first tail item
    # unless its function_call is also in tail
    if tail and isinstance(tail[0], dict) and tail[0].get("type") == "function_call_output":
        call_ids_in_tail = {
            it.get("call_id")
            for it in tail
            if isinstance(it, dict) and it.get("type") == "function_call"
        }
        assert tail[0].get("call_id") in call_ids_in_tail


def test_strip_images_replaces_image_blocks() -> None:
    items = [
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_image", "image_url": "data:..."}],
        }
    ]
    stripped = strip_images(items)
    block = stripped[0]["output"][0]
    assert block["type"] == "input_text"
    assert "screenshot" in block["text"].lower() or "image" in block["text"].lower()
    # original untouched
    assert items[0]["output"][0]["type"] == "input_image"
