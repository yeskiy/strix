"""Turn grouping, head/tail split, and image stripping for compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strix.core.compaction.tokens import estimate_tokens
from strix.core.sessions import scrub_images_from_items


if TYPE_CHECKING:
    from strix.core.compaction.config import CompactionConfig


def iter_turn_start_indices(items: list[Any]) -> list[int]:
    return [
        i for i, item in enumerate(items) if isinstance(item, dict) and item.get("role") == "user"
    ]


def _is_safe_boundary(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("role") in {"user", "assistant"}:
        return True
    return item.get("type") == "function_call"


def _safe_inner_split(items: list[Any], start: int, budget: int) -> int:
    """Largest tail (smallest safe index > start) whose estimate fits ``budget``."""
    for b in range(start + 1, len(items)):
        if _is_safe_boundary(items[b]) and estimate_tokens(items[b:]) <= budget:
            return b
    return start


def split_head_tail(items: list[Any], cfg: CompactionConfig) -> tuple[list[Any], list[Any]]:
    starts = iter_turn_start_indices(items)
    if len(starts) <= cfg.tail_turns:
        return [], items

    tail_start = starts[-cfg.tail_turns]

    # Drop older tail turns into head while the tail exceeds its token budget.
    idx = len(starts) - cfg.tail_turns
    while (
        tail_start != starts[-1]
        and estimate_tokens(items[tail_start:]) > cfg.preserve_recent_tokens
    ):
        idx += 1
        tail_start = starts[idx]

    # A single oversized tail turn: split inside it at a safe boundary.
    if estimate_tokens(items[tail_start:]) > cfg.preserve_recent_tokens:
        tail_start = _safe_inner_split(items, tail_start, cfg.preserve_recent_tokens)

    return items[:tail_start], items[tail_start:]


def strip_images(items: list[Any]) -> list[Any]:
    return scrub_images_from_items(items)
