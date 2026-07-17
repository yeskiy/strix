"""Tests for MCP config loading and persistence in strix.config.loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from strix.config import loader
from strix.config.settings import MCPServerConfig


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_cached", None)
    monkeypatch.setattr(loader, "_override", None)


def test_read_json_overrides_passes_mcp_servers_through(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(
        json.dumps(
            {
                "env": {"STRIX_LLM": "m"},
                "mcp_servers": [{"name": "cve", "command": "npx", "args": ["-y", "cve-mcp"]}],
            }
        ),
        encoding="utf-8",
    )
    result = loader._read_json_overrides(path)
    assert result["mcp_servers"] == [{"name": "cve", "command": "npx", "args": ["-y", "cve-mcp"]}]


def test_read_json_overrides_ignores_non_list_mcp_servers(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"mcp_servers": {"not": "a list"}}), encoding="utf-8")
    assert "mcp_servers" not in loader._read_json_overrides(path)


def test_load_settings_builds_mcp_server_configs(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "cve",
                        "command": "npx",
                        "args": ["-y", "cve-mcp"],
                        "allowed_tools": ["lookup_cve"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(path)
    settings = loader.load_settings()
    assert len(settings.mcp_servers) == 1
    server = settings.mcp_servers[0]
    assert isinstance(server, MCPServerConfig)
    assert server.name == "cve"
    assert server.allowed_tools == ["lookup_cve"]


def test_persist_current_preserves_mcp_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "persisted-model")
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"mcp_servers": [{"name": "cve", "command": "npx"}]}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)

    loader.persist_current()

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["env"] == {"STRIX_LLM": "persisted-model"}
    assert written["mcp_servers"] == [{"name": "cve", "command": "npx"}]
