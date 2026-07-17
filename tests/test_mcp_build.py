"""Tests for strix.core.mcp: command normalization and server building."""

from __future__ import annotations

import pytest
from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from strix.config.settings import MCPServerConfig
from strix.core import mcp as mcp_module


# --- normalize_command ---


def test_normalize_command_wraps_shim_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_module.os, "name", "nt")
    command, args = mcp_module.normalize_command("npx", ["-y", "cve-mcp"])
    assert command == "cmd"
    assert args == ["/c", "npx", "-y", "cve-mcp"]


def test_normalize_command_passthrough_non_shim_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_module.os, "name", "nt")
    command, args = mcp_module.normalize_command("python", ["server.py"])
    assert command == "python"
    assert args == ["server.py"]


def test_normalize_command_passthrough_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_module.os, "name", "posix")
    command, args = mcp_module.normalize_command("npx", ["-y", "cve-mcp"])
    assert command == "npx"
    assert args == ["-y", "cve-mcp"]


# --- build_mcp_servers ---


def test_build_mcp_servers_skips_disabled() -> None:
    configs = [
        MCPServerConfig(name="off", command="npx", enabled=False),
        MCPServerConfig(name="on", command="npx", args=["-y", "cve-mcp"]),
    ]
    servers = mcp_module.build_mcp_servers(configs)
    assert [s.name for s in servers] == ["on"]


def test_build_mcp_servers_stdio_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_module.os, "name", "posix")
    config = MCPServerConfig(
        name="cve",
        command="npx",
        args=["-y", "cve-mcp"],
        cache_tools_list=True,
        connect_timeout_seconds=42,
    )
    server = mcp_module.build_mcp_servers([config])[0]
    assert isinstance(server, MCPServerStdio)
    assert server.name == "cve"
    assert server.params.command == "npx"
    assert server.params.args == ["-y", "cve-mcp"]
    assert server.cache_tools_list is True
    assert server.client_session_timeout_seconds == 42


def test_build_mcp_servers_stdio_normalizes_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_module.os, "name", "nt")
    config = MCPServerConfig(name="cve", command="npx", args=["-y", "cve-mcp"])
    server = mcp_module.build_mcp_servers([config])[0]
    assert isinstance(server, MCPServerStdio)
    assert server.params.command == "cmd"
    assert server.params.args == ["/c", "npx", "-y", "cve-mcp"]


def test_build_mcp_servers_allowed_tools_sets_filter() -> None:
    config = MCPServerConfig(name="cve", command="npx", allowed_tools=["lookup_cve"])
    server = mcp_module.build_mcp_servers([config])[0]
    assert server.tool_filter == {"allowed_tool_names": ["lookup_cve"]}


def test_build_mcp_servers_no_allowed_tools_no_filter() -> None:
    config = MCPServerConfig(name="cve", command="npx")
    server = mcp_module.build_mcp_servers([config])[0]
    assert server.tool_filter is None


def test_build_mcp_servers_sse_params() -> None:
    config = MCPServerConfig(
        name="remote",
        transport="sse",
        url="https://mcp.example.com/sse",
        headers={"Authorization": "Bearer x"},
    )
    server = mcp_module.build_mcp_servers([config])[0]
    assert isinstance(server, MCPServerSse)
    assert server.params["url"] == "https://mcp.example.com/sse"
    assert server.params["headers"] == {"Authorization": "Bearer x"}


def test_build_mcp_servers_streamable_http_params() -> None:
    config = MCPServerConfig(
        name="remote",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
    )
    server = mcp_module.build_mcp_servers([config])[0]
    assert isinstance(server, MCPServerStreamableHttp)
    assert server.params["url"] == "https://mcp.example.com/mcp"


def test_build_mcp_servers_stdio_without_command_raises() -> None:
    config = MCPServerConfig(name="bad", transport="stdio", command=None)
    with pytest.raises(ValueError, match="requires 'command'"):
        mcp_module.build_mcp_servers([config])


def test_build_mcp_servers_http_without_url_raises() -> None:
    config = MCPServerConfig(name="bad", transport="sse", url=None)
    with pytest.raises(ValueError, match="requires 'url'"):
        mcp_module.build_mcp_servers([config])


# --- mcp_connect_timeout ---


def test_mcp_connect_timeout_uses_max_enabled() -> None:
    configs = [
        MCPServerConfig(name="a", command="npx", connect_timeout_seconds=20),
        MCPServerConfig(name="b", command="npx", connect_timeout_seconds=60),
        MCPServerConfig(name="c", command="npx", connect_timeout_seconds=90, enabled=False),
    ]
    assert mcp_module.mcp_connect_timeout(configs) == 60.0


def test_mcp_connect_timeout_defaults_when_empty() -> None:
    assert mcp_module.mcp_connect_timeout([]) == 30.0
