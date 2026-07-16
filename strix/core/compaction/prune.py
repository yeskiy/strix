"""Cheap tool-output eviction that shrinks old function_call_output text."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from strix.core.compaction.tokens import tokens_from_chars


if TYPE_CHECKING:
    from strix.core.compaction.config import CompactionConfig


PRUNED_PREFIX = "[pruned:"


def _recent_boundary(items: list[Any], tail_turns: int) -> int:
    """Index of start of last ``tail_turns`` user turns; items at/after are protected."""
    seen = 0
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if isinstance(item, dict) and item.get("role") == "user":
            seen += 1
            if seen >= tail_turns:
                return i
    return 0


def _call_id_names(items: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if call_id and isinstance(name, str):
                mapping[str(call_id)] = name
    return mapping


def _output_str(output: Any) -> str:
    return (
        output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    )


def prune_tool_outputs(
    items: list[Any], cfg: CompactionConfig, ratio: float | None = None
) -> tuple[list[Any], int]:
    if not cfg.prune_enabled or not items:
        return items, 0

    boundary = _recent_boundary(items, cfg.tail_turns)
    names = _call_id_names(items)
    protect_budget = cfg.prune_protect_tokens

    replacements: list[tuple[int, str]] = []
    total_reclaim = 0

    for i in range(boundary - 1, -1, -1):
        item = items[i]
        if not (isinstance(item, dict) and item.get("type") == "function_call_output"):
            continue
        out_str = _output_str(item.get("output"))
        if out_str.startswith(PRUNED_PREFIX):
            continue
        out_tokens = tokens_from_chars(len(out_str), ratio)
        if protect_budget > 0:
            protect_budget -= out_tokens
            continue
        tool = names.get(str(item.get("call_id") or ""), "tool")
        placeholder = f"{PRUNED_PREFIX} ~{out_tokens} tokens of {tool} output]"
        reclaimed = out_tokens - tokens_from_chars(len(placeholder), ratio)
        if reclaimed <= 0:
            continue
        replacements.append((i, placeholder))
        total_reclaim += reclaimed

    if total_reclaim < cfg.prune_min_reclaim:
        return items, 0

    new_items = list(items)
    for i, placeholder in replacements:
        new_items[i] = {**items[i], "output": placeholder}
    return new_items, total_reclaim
