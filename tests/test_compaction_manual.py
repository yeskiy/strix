"""Tests for the manual /compact command path."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from strix.core.agents import AgentCoordinator
from strix.core.compaction.store import get_store
from strix.interface.tui.live_view import TuiLiveView
from strix.interface.tui.messages import send_user_message_to_agent


@pytest.mark.asyncio
async def test_request_compaction_sets_force_flag_for_known_agent() -> None:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    ok = await coordinator.request_compaction("root")
    assert ok is True
    assert get_store().consume_force("root") is True


@pytest.mark.asyncio
async def test_request_compaction_unknown_agent_returns_false() -> None:
    coordinator = AgentCoordinator()
    assert await coordinator.request_compaction("nope") is False


def test_record_system_message_appends_system_role() -> None:
    view = TuiLiveView()
    view.record_system_message("root", "Compaction requested.")
    events = view.events_for_agent("root")
    assert events
    assert events[-1]["data"]["role"] == "system"
    assert "Compaction" in events[-1]["data"]["content"]


class _FakeCoordinator:
    def __init__(self) -> None:
        self.compact_calls: list[str] = []
        self.send_calls: list[Any] = []
        self.done = threading.Event()

    async def request_compaction(self, agent_id: str) -> bool:
        self.compact_calls.append(agent_id)
        self.done.set()
        return True

    async def send(self, target_agent_id: str, message: dict[str, Any]) -> bool:
        self.send_calls.append((target_agent_id, message))
        self.done.set()
        return True


def _loop_in_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


@pytest.mark.parametrize("command", ["/compact", "/compress", "  /COMPACT  "])
def test_compact_command_intercepted(command: str) -> None:
    loop, thread = _loop_in_thread()
    coordinator = _FakeCoordinator()
    view = TuiLiveView()
    try:
        submitted = send_user_message_to_agent(
            coordinator=coordinator,
            loop=loop,
            live_view=view,
            target_agent_id="root",
            message=command,
        )
        assert coordinator.done.wait(timeout=2) is True
    finally:
        _stop_loop(loop, thread)
    assert submitted is True
    assert coordinator.compact_calls == ["root"]
    assert coordinator.send_calls == []
    # a system line was recorded, not a user message
    assert any(e["data"].get("role") == "system" for e in view.events_for_agent("root"))


def test_normal_message_not_intercepted() -> None:
    loop, thread = _loop_in_thread()
    coordinator = _FakeCoordinator()
    view = TuiLiveView()
    try:
        send_user_message_to_agent(
            coordinator=coordinator,
            loop=loop,
            live_view=view,
            target_agent_id="root",
            message="hello there",
        )
        assert coordinator.done.wait(timeout=2) is True
    finally:
        _stop_loop(loop, thread)
    assert coordinator.compact_calls == []
    assert coordinator.send_calls and coordinator.send_calls[0][0] == "root"
