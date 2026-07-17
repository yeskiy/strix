"""Tests that run_strix_scan wires MCP servers into the scan lifecycle."""

from __future__ import annotations

import types
from typing import Any, ClassVar, Self

import pytest

import strix.tools.notes.tools as notes_tools
import strix.tools.todo.tools as todo_tools
from strix.agents import factory
from strix.config.settings import CompactionSettings, MCPServerConfig
from strix.core import runner
from strix.core.agents import AgentCoordinator


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name

    async def list_tools(self) -> list[Any]:
        return [object(), object()]


class _FakeManager:
    """Stand-in for agents.mcp.MCPServerManager."""

    instances: ClassVar[list[_FakeManager]] = []

    def __init__(self, servers: Any, **_kwargs: Any) -> None:
        self.servers = list(servers)
        self.failed_servers: list[Any] = []
        self.entered = False
        self.exited = False
        _FakeManager.instances.append(self)

    @property
    def active_servers(self) -> list[Any]:
        return list(self.servers)

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.exited = True


@pytest.fixture(autouse=True)
def _reset_mcp_registry() -> object:
    saved = list(factory._MCP_SERVERS)
    factory._MCP_SERVERS.clear()
    _FakeManager.instances.clear()
    try:
        yield
    finally:
        factory._MCP_SERVERS[:] = saved


def _patch_scaffold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    mcp_servers: list[MCPServerConfig],
    fake_servers: list[_FakeServer],
) -> dict[str, Any]:
    monkeypatch.setattr(runner, "run_dir_for", lambda _scan_id: tmp_path)
    monkeypatch.setattr(runner, "runtime_state_dir", lambda _run_dir: tmp_path)
    monkeypatch.setattr(runner, "setup_scan_logging", lambda _run_dir: lambda: None)
    monkeypatch.setattr(runner, "set_scan_id", lambda _scan_id: None)

    settings = types.SimpleNamespace(
        llm=types.SimpleNamespace(
            model="openai/gpt-4o",
            reasoning_effort="high",
            force_required_tool_choice=False,
        ),
        runtime=types.SimpleNamespace(max_context_images=3),
        compaction=CompactionSettings(),
        mcp_servers=mcp_servers,
    )
    monkeypatch.setattr(runner, "load_settings", lambda: settings)
    monkeypatch.setattr(runner, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        runner, "uses_chat_completions_tool_schema", lambda _model, _settings: False
    )

    monkeypatch.setattr(todo_tools, "hydrate_todos_from_disk", lambda _state_dir: None)
    monkeypatch.setattr(notes_tools, "hydrate_notes_from_disk", lambda _state_dir: None)

    async def _create_or_reuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"client": object(), "session": object(), "caido_client": None}

    async def _cleanup(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner.session_manager, "create_or_reuse", _create_or_reuse)
    monkeypatch.setattr(runner.session_manager, "cleanup", _cleanup)

    monkeypatch.setattr(runner, "build_root_task", lambda _scan_config: "task")
    monkeypatch.setattr(runner, "build_scope_context", lambda _scan_config: {"scope": "s"})
    monkeypatch.setattr(runner, "make_model_settings", lambda *_a, **_k: object())

    monkeypatch.setattr(runner, "build_mcp_servers", lambda _configs: list(fake_servers))
    monkeypatch.setattr(runner, "mcp_connect_timeout", lambda _configs: 30.0)
    monkeypatch.setattr(runner, "MCPServerManager", _FakeManager)

    captured: dict[str, Any] = {}

    def _build_strix_agent(**kwargs: Any) -> object:
        if kwargs.get("is_root"):
            captured["servers_at_build"] = list(factory._MCP_SERVERS)
        return object()

    monkeypatch.setattr(runner, "build_strix_agent", _build_strix_agent)
    monkeypatch.setattr(runner, "make_child_factory", lambda **_k: lambda **_kk: object())
    monkeypatch.setattr(runner, "open_agent_session", lambda _root_id, _db: object())

    async def _run_agent_loop(*_args: Any, **_kwargs: Any) -> None:
        captured["servers_during_run"] = list(factory._MCP_SERVERS)

    monkeypatch.setattr(runner, "run_agent_loop", _run_agent_loop)
    return captured


@pytest.mark.asyncio
async def test_mcp_servers_registered_before_build_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    fake_servers = [_FakeServer("cve"), _FakeServer("osint")]
    captured = _patch_scaffold(
        monkeypatch,
        tmp_path,
        mcp_servers=[MCPServerConfig(name="cve", command="npx")],
        fake_servers=fake_servers,
    )

    await runner.run_strix_scan(
        scan_config={"targets": [], "scan_mode": "deep"},
        scan_id="scan-mcp",
        image="img",
        coordinator=AgentCoordinator(),
    )

    # Registered with the manager's active servers before the root agent built.
    assert captured["servers_at_build"] == fake_servers
    assert captured["servers_during_run"] == fake_servers
    # Manager entered (connect) and exited (cleanup) exactly once.
    assert len(_FakeManager.instances) == 1
    assert _FakeManager.instances[0].entered is True
    assert _FakeManager.instances[0].exited is True
    # Registry cleared in finally.
    assert factory._MCP_SERVERS == []


@pytest.mark.asyncio
async def test_no_mcp_servers_skips_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    captured = _patch_scaffold(monkeypatch, tmp_path, mcp_servers=[], fake_servers=[])

    await runner.run_strix_scan(
        scan_config={"targets": [], "scan_mode": "deep"},
        scan_id="scan-nomcp",
        image="img",
        coordinator=AgentCoordinator(),
    )

    assert _FakeManager.instances == []
    assert captured["servers_at_build"] == []
    assert factory._MCP_SERVERS == []
