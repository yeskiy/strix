"""Tests for MCPServerConfig and Settings.mcp_servers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strix.config.settings import MCPServerConfig, Settings


def test_mcp_server_config_defaults() -> None:
    cfg = MCPServerConfig(name="cve")
    assert cfg.enabled is True
    assert cfg.transport == "stdio"
    assert cfg.command is None
    assert cfg.args == []
    assert cfg.env == {}
    assert cfg.url is None
    assert cfg.headers == {}
    assert cfg.allowed_tools == []
    assert cfg.cache_tools_list is True
    assert cfg.connect_timeout_seconds == 30.0


def test_mcp_server_config_stdio_full() -> None:
    cfg = MCPServerConfig(
        name="cve",
        transport="stdio",
        command="npx",
        args=["-y", "cve-mcp"],
        env={"TOKEN": "secret"},
        allowed_tools=["lookup_cve"],
        cache_tools_list=False,
        connect_timeout_seconds=45,
    )
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "cve-mcp"]
    assert cfg.env == {"TOKEN": "secret"}
    assert cfg.allowed_tools == ["lookup_cve"]
    assert cfg.cache_tools_list is False
    assert cfg.connect_timeout_seconds == 45.0


def test_mcp_server_config_rejects_unknown_transport() -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig(name="x", transport="carrier-pigeon")


def test_settings_defaults_to_no_mcp_servers() -> None:
    assert Settings().mcp_servers == []


def test_settings_accepts_mcp_servers_list() -> None:
    settings = Settings(mcp_servers=[{"name": "cve", "command": "npx", "args": ["-y", "cve-mcp"]}])
    assert len(settings.mcp_servers) == 1
    assert isinstance(settings.mcp_servers[0], MCPServerConfig)
    assert settings.mcp_servers[0].name == "cve"
