"""Message delivery bridge from TUI input to SDK-backed agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any


logger = logging.getLogger(__name__)

_COMPACT_COMMANDS = frozenset({"/compact", "/compress"})


def send_user_message_to_agent(
    *,
    coordinator: Any,
    loop: asyncio.AbstractEventLoop | None,
    live_view: Any,
    target_agent_id: str,
    message: str,
) -> bool:
    if loop is None or loop.is_closed():
        return False

    if message.strip().lower() in _COMPACT_COMMANDS:
        live_view.record_system_message(
            target_agent_id,
            "Compaction requested; it runs before this agent's next model call.",
        )
        future = asyncio.run_coroutine_threadsafe(
            coordinator.request_compaction(target_agent_id),
            loop,
        )
        future.add_done_callback(_log_delivery_failure)
        return True

    live_view.record_user_message(target_agent_id, message)
    future = asyncio.run_coroutine_threadsafe(
        coordinator.send(
            target_agent_id,
            {"from": "user", "content": message, "type": "instruction"},
        ),
        loop,
    )
    future.add_done_callback(_log_delivery_failure)
    return True


def _log_delivery_failure(future: Any) -> None:
    try:
        delivered = bool(future.result())
    except Exception:
        logger.exception("TUI user message delivery failed")
        return
    if not delivered:
        logger.warning("TUI user message was not persisted to the SDK session")
