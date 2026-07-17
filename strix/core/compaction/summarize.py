"""LLM summarization of the compacted head, via the shared LiteLLM route."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from strix.core.compaction.prompt import (
    PREVIOUS_SUMMARY_CLOSE,
    PREVIOUS_SUMMARY_OPEN,
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_TEMPLATE,
)


if TYPE_CHECKING:
    from strix.core.compaction.config import CompactionConfig


logger = logging.getLogger(__name__)


def _normalize_summary_model(model: str) -> str:
    name = model.strip()
    for prefix in ("litellm/", "any-llm/"):
        if name.lower().startswith(prefix):
            return name[len(prefix) :]
    return name


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


async def summarize_head(items: list[Any], prev_summary: str | None, cfg: CompactionConfig) -> str:
    import litellm

    history_json = json.dumps(items, ensure_ascii=False, default=str)
    user_content = SUMMARIZER_USER_TEMPLATE.format(history=history_json)
    if prev_summary:
        user_content = (
            f"{PREVIOUS_SUMMARY_OPEN}\n{prev_summary}\n{PREVIOUS_SUMMARY_CLOSE}\n\n{user_content}"
        )
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict[str, Any] = {
        "model": _normalize_summary_model(cfg.summary_model),
        "messages": messages,
        "stream": False,
        "drop_params": True,
        "temperature": 0,
    }
    # OpenRouter's middle-out transform lets an oversized head still summarize
    # instead of hard-failing the summarizer call on a context-window overflow.
    if "openrouter/" in cfg.summary_model.lower():
        kwargs["extra_body"] = {"transforms": ["middle-out"]}
    response = await litellm.acompletion(**kwargs)
    return _extract_text(response)
