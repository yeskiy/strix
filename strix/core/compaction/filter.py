"""The call_model_input_filter closure that ties prune + summarize together."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agents.run_config import CallModelData, ModelInputData

from strix.core.compaction.history import (
    merge_consecutive_roles,
    split_head_tail,
    strip_images,
    trim_front_to_budget,
)
from strix.core.compaction.prompt import summary_message
from strix.core.compaction.prune import prune_tool_outputs
from strix.core.compaction.summarize import summarize_head
from strix.core.compaction.tokens import char_count, estimate_tokens


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from strix.core.compaction.config import CompactionConfig
    from strix.core.compaction.store import AgentCompactionState, CompactionStore


logger = logging.getLogger(__name__)


def _resolve_agent_id(data: CallModelData[Any]) -> str:
    ctx = data.context if isinstance(data.context, dict) else {}
    agent_id = ctx.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    name = getattr(data.agent, "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _effective(state: AgentCompactionState, items: list[Any]) -> list[Any]:
    if state.summary is None:
        return items
    return [summary_message(state.summary), *items[state.anchor :]]


def build_compaction_filter(
    cfg: CompactionConfig, store: CompactionStore
) -> Callable[[CallModelData[Any]], Awaitable[ModelInputData]]:
    threshold_tokens = cfg.usable_window * cfg.threshold
    relief_tokens = cfg.usable_window * (cfg.threshold - cfg.hysteresis)

    async def _filter(data: CallModelData[Any]) -> ModelInputData:
        model_data = data.model_data
        items = list(model_data.input)
        if not items:
            return model_data

        agent_id = _resolve_agent_id(data)
        state = store.state(agent_id)
        forced = store.consume_force(agent_id)
        ratio = state.calibrator.ratio

        eff = _effective(state, items)
        store.record_pending_chars(agent_id, char_count(eff))

        if not forced and estimate_tokens(eff, ratio) < threshold_tokens:
            if state.summary is None:
                return model_data
            return ModelInputData(
                input=merge_consecutive_roles(eff), instructions=model_data.instructions
            )

        logger.info(
            "compaction triggered: agent=%s est_tokens=%d threshold=%d forced=%s",
            agent_id,
            estimate_tokens(eff, ratio),
            int(threshold_tokens),
            forced,
        )
        result = merge_consecutive_roles(await _compact(agent_id, items, state, ratio, forced))
        logger.info("compaction done: agent=%s items %d -> %d", agent_id, len(items), len(result))
        store.record_pending_chars(agent_id, char_count(result))
        return ModelInputData(input=result, instructions=model_data.instructions)

    async def _compact(
        agent_id: str,
        items: list[Any],
        state: AgentCompactionState,
        ratio: float | None,
        forced: bool,
    ) -> list[Any]:
        pruned = items
        if cfg.prune_enabled:
            pruned, _ = prune_tool_outputs(items, cfg, ratio)
        eff_pruned = _effective(state, pruned)
        # Auto path: if pruning alone dropped below the hysteresis floor, stop
        # here with no LLM cost. A manual force always proceeds to summarize.
        if not forced and estimate_tokens(eff_pruned, ratio) < relief_tokens:
            return eff_pruned

        head, tail = split_head_tail(pruned, cfg)
        head_delta = head[state.anchor :] if state.summary is not None else head
        if not head_delta:
            return eff_pruned

        try:
            new_summary = await summarize_head(strip_images(head_delta), state.summary, cfg)
        except Exception:
            logger.exception("compaction summarizer failed for %s; using pruned input", agent_id)
            return eff_pruned

        if not new_summary.strip():
            return eff_pruned

        store.set_summary(agent_id, new_summary, len(head))
        summarized = [summary_message(new_summary), *strip_images(tail)]
        capped = trim_front_to_budget(summarized, cfg.usable_window, ratio)
        if len(capped) < len(summarized):
            logger.info(
                "compaction hard-trim: agent=%s items %d -> %d over usable_window=%d",
                agent_id,
                len(summarized),
                len(capped),
                cfg.usable_window,
            )
        return capped

    return _filter
