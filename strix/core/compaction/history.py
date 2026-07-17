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


def _safe_boundaries(items: list[Any]) -> list[int]:
    return [i for i in range(len(items)) if _is_safe_boundary(items[i])]


def split_head_tail(items: list[Any], cfg: CompactionConfig) -> tuple[list[Any], list[Any]]:
    """Split by a token budget at safe boundaries, independent of user-turn count.

    Strix agents pile huge context into a single user turn, so splitting on the
    number of user turns is wrong. The tail is the largest suffix that starts on a
    safe boundary and whose estimate fits ``cfg.preserve_recent_tokens``; the head
    is everything before it. When even the newest safe boundary already exceeds the
    budget (one oversized item), the tail collapses to that boundary so the head
    still receives everything earlier. When the whole history fits, the tail is the
    whole history and the head is empty (nothing to summarize).
    """
    boundaries = _safe_boundaries(items)
    if not boundaries:
        return [], items

    budget = cfg.preserve_recent_tokens
    tail_start = boundaries[-1]
    for start in reversed(boundaries):
        if estimate_tokens(items[start:]) <= budget:
            tail_start = start
        else:
            break
    return items[:tail_start], items[tail_start:]


def _role_of(item: Any) -> str | None:
    if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
        return str(item["role"])
    return None


def _as_blocks(content: Any, role: str) -> list[Any]:
    if isinstance(content, list):
        return content
    block_type = "output_text" if role == "assistant" else "input_text"
    return [{"type": block_type, "text": content if isinstance(content, str) else str(content)}]


def _merge_content(prev: dict[str, Any], curr: dict[str, Any], role: str) -> Any:
    prev_c, curr_c = prev.get("content"), curr.get("content")
    if isinstance(prev_c, str) and isinstance(curr_c, str):
        return f"{prev_c}\n\n{curr_c}"
    return [*_as_blocks(prev_c, role), *_as_blocks(curr_c, role)]


def merge_consecutive_roles(items: list[Any]) -> list[Any]:
    """Merge adjacent messages that share the same role so providers requiring
    strict user/assistant alternation (GLM/Z.AI) do not reject the payload.
    Only role-bearing messages (dicts with a "role" of user/assistant) are
    merged; function_call / function_call_output items are left untouched."""
    merged: list[Any] = []
    for item in items:
        role = _role_of(item)
        if role is not None and merged and _role_of(merged[-1]) == role:
            prev = merged[-1]
            merged[-1] = {**prev, "content": _merge_content(prev, item, role)}
        else:
            merged.append(item)
    return merged


def trim_front_to_budget(items: list[Any], budget: int, ratio: float | None = None) -> list[Any]:
    """Drop the oldest tail items (everything after ``items[0]``) at safe boundaries
    until the estimate fits ``budget`` or only ``items[0]`` remains. A hard ceiling
    for the case where a single retained item is still larger than the window."""
    if not items or estimate_tokens(items, ratio) <= budget:
        return items
    head, tail = items[:1], items[1:]
    while tail and estimate_tokens([*head, *tail], ratio) > budget:
        nxt = next((i for i in range(1, len(tail)) if _is_safe_boundary(tail[i])), None)
        if nxt is None:
            tail = []
            break
        tail = tail[nxt:]
    return [*head, *tail]


def strip_images(items: list[Any]) -> list[Any]:
    return scrub_images_from_items(items)
