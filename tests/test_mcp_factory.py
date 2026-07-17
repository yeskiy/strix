"""Tests for MCP server registration in the agent factory."""

from __future__ import annotations

import pytest

from strix.agents import factory


class _FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture(autouse=True)
def _reset_mcp_registry() -> object:
    saved = list(factory._MCP_SERVERS)
    factory._MCP_SERVERS.clear()
    try:
        yield
    finally:
        factory._MCP_SERVERS[:] = saved


def test_register_mcp_servers_is_deduped() -> None:
    server = _FakeServer("cve")
    factory.register_mcp_servers([server])
    factory.register_mcp_servers([server])
    assert factory.registered_mcp_servers() == (server,)


def test_clear_mcp_servers_empties_registry() -> None:
    factory.register_mcp_servers([_FakeServer("cve")])
    factory.clear_mcp_servers()
    assert factory.registered_mcp_servers() == ()


def test_build_strix_agent_forwards_registered_servers() -> None:
    servers = [_FakeServer("cve"), _FakeServer("osint")]
    factory.register_mcp_servers(servers)

    agent = factory.build_strix_agent(is_root=True)

    assert list(agent.mcp_servers) == servers


def test_build_strix_agent_no_servers_by_default() -> None:
    agent = factory.build_strix_agent(is_root=True)
    assert list(agent.mcp_servers) == []


def test_mcp_servers_snapshot_is_independent() -> None:
    servers = [_FakeServer("cve")]
    factory.register_mcp_servers(servers)
    agent = factory.build_strix_agent(is_root=True)
    factory.register_mcp_servers([_FakeServer("osint")])
    # The agent captured a snapshot of the registry at build time.
    assert [s.name for s in agent.mcp_servers] == ["cve"]
