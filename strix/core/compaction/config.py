"""Resolve STRIX_COMPACTION_* settings into an immutable config object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from strix.config.settings import Settings


HYSTERESIS_DEFAULT = 0.10
RESERVED_OUTPUT_CAP = 32000


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool
    threshold: float
    hysteresis: float
    reserved_output: int
    tail_turns: int
    preserve_recent_tokens: int
    prune_enabled: bool
    prune_protect_tokens: int
    prune_min_reclaim: int
    session_model: str
    summary_model: str
    usable_window: int

    @classmethod
    def from_env(cls, *, session_model: str, settings: Settings) -> CompactionConfig:
        from strix.core.compaction.tokens import context_window, max_output_for

        cs = settings.compaction
        reserved = (
            cs.reserved_output
            if cs.reserved_output is not None
            else min(RESERVED_OUTPUT_CAP, max_output_for(session_model))
        )
        usable = max(
            1, context_window(session_model, override=cs.context_window or None) - reserved
        )
        return cls(
            enabled=cs.enabled,
            threshold=cs.threshold,
            hysteresis=HYSTERESIS_DEFAULT,
            reserved_output=reserved,
            tail_turns=cs.tail_turns,
            preserve_recent_tokens=cs.preserve_recent_tokens,
            prune_enabled=cs.prune,
            prune_protect_tokens=cs.prune_protect_tokens,
            prune_min_reclaim=cs.prune_min_reclaim,
            session_model=session_model,
            summary_model=cs.model or session_model,
            usable_window=usable,
        )
