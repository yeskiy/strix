"""Tests for turn grouping, head/tail split, and image stripping."""

from __future__ import annotations

from typing import Any

from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.history import (
    iter_turn_start_indices,
    merge_consecutive_roles,
    split_head_tail,
    strip_images,
)
from strix.core.compaction.tokens import estimate_tokens


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


def test_split_by_token_budget_at_safe_boundary() -> None:
    # Large turns with a small budget: the split is driven by the token budget,
    # not the user-turn count. Head must be non-empty and the tail must start on
    # a safe boundary (a role message or a function_call), never mid tool pair.
    items = [
        {"role": "user", "content": "t1 " + "x" * 4000},
        {"role": "assistant", "content": "a1 " + "y" * 4000},
        {"role": "user", "content": "t2 " + "x" * 4000},
        {"role": "assistant", "content": "a2 " + "y" * 4000},
        {"role": "user", "content": "t3"},
        {"role": "assistant", "content": "a3"},
    ]
    head, tail = split_head_tail(items, _cfg(preserve_recent_tokens=200))
    assert head, "large history over budget must produce a non-empty head"
    assert estimate_tokens(tail) <= 200
    assert isinstance(tail[0], dict)
    assert tail[0].get("role") in {"user", "assistant"} or tail[0].get("type") == "function_call"
    assert head + tail == items


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


def test_split_head_tail_single_giant_turn() -> None:
    # One long autonomous turn: assistant/function_call/function_call_output
    # triples with large tool outputs, plus two user messages -> only two user
    # turns, yet the token budget must still carve out a non-empty head.
    big = "Z" * 5000
    items: list[Any] = [{"role": "user", "content": "kick off the run"}]
    for i in range(16):
        if i == 8:
            items.append({"role": "user", "content": "keep going"})
        items.append({"role": "assistant", "content": f"step {i}"})
        items.append(
            {"type": "function_call", "call_id": f"c{i}", "name": "browser", "arguments": "{}"}
        )
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": big})

    cfg = _cfg(preserve_recent_tokens=8000, tail_turns=2)
    head, tail = split_head_tail(items, cfg)

    assert head, "head must be non-empty for an oversized single-turn history"
    assert estimate_tokens(tail) <= cfg.preserve_recent_tokens
    assert not (isinstance(tail[0], dict) and tail[0].get("type") == "function_call_output")
    # nothing is lost: head + tail reconstruct the original order
    assert head + tail == items


def test_merge_consecutive_roles_merges_user_user() -> None:
    items = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    merged = merge_consecutive_roles(items)
    assert len(merged) == 1
    assert merged[0]["role"] == "user"
    assert "a" in merged[0]["content"]
    assert "b" in merged[0]["content"]
    roles = [it.get("role") for it in merged if isinstance(it, dict)]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_merge_consecutive_roles_leaves_tool_items() -> None:
    items = [
        {"role": "assistant", "content": "calling the browser"},
        {"type": "function_call", "call_id": "c1", "name": "browser", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
    ]
    assert merge_consecutive_roles(items) == items


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
