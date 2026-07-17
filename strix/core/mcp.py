"""Config-driven MCP client layer.

Builds Agents-SDK MCP server objects from :class:`MCPServerConfig` entries so
the CLI can attach any stdio / SSE / streamable-HTTP MCP server to every scan
agent. stdio servers run as host subprocesses, outside the Docker sandbox;
their ``env`` / ``headers`` may hold secrets and are never logged. The caller
owns the connect / cleanup lifecycle (see ``agents.mcp.MCPServerManager``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agents.mcp import (
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from agents.mcp import (
        MCPServer,
        MCPServerSseParams,
        MCPServerStdioParams,
        MCPServerStreamableHttpParams,
    )

    from strix.config.settings import MCPServerConfig


# Node / uv launchers ship as Windows shims (.cmd) that CreateProcess cannot
# exec directly; they must be run through ``cmd /c``.
_WINDOWS_SHIMS = frozenset({"npx", "npm", "uvx"})

DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0


def normalize_command(command: str, args: list[str]) -> tuple[str, list[str]]:
    """Rewrite npm / npx / uvx shims to run via ``cmd /c`` on Windows.

    Elsewhere, and for any non-shim command, the command and args pass through
    unchanged.
    """
    if os.name == "nt" and command.lower() in _WINDOWS_SHIMS:
        return "cmd", ["/c", command, *args]
    return command, list(args)


def mcp_connect_timeout(configs: Iterable[MCPServerConfig]) -> float:
    """Connect timeout for the shared MCPServerManager.

    The manager applies one connect timeout to every server, so use the
    largest per-server ``connect_timeout_seconds`` among enabled configs; npx
    cold-starts can be slow on first run.
    """
    timeouts = [c.connect_timeout_seconds for c in configs if c.enabled]
    return max([DEFAULT_CONNECT_TIMEOUT_SECONDS, *timeouts])


def build_mcp_servers(configs: Iterable[MCPServerConfig]) -> list[MCPServer]:
    """Instantiate SDK MCP server objects for every enabled config.

    Disabled configs are skipped. Servers are NOT connected here.
    """
    return [_build_server(c) for c in configs if c.enabled]


def _build_server(config: MCPServerConfig) -> MCPServer:
    tool_filter = (
        create_static_tool_filter(allowed_tool_names=list(config.allowed_tools))
        if config.allowed_tools
        else None
    )

    if config.transport == "stdio":
        if not config.command:
            raise ValueError(f"MCP server '{config.name}': stdio transport requires 'command'.")
        command, args = normalize_command(config.command, list(config.args))
        stdio_params: MCPServerStdioParams = {"command": command, "args": args}
        if config.env:
            stdio_params["env"] = dict(config.env)
        return MCPServerStdio(
            params=stdio_params,
            name=config.name,
            cache_tools_list=config.cache_tools_list,
            client_session_timeout_seconds=config.connect_timeout_seconds,
            tool_filter=tool_filter,
        )

    if not config.url:
        raise ValueError(
            f"MCP server '{config.name}': {config.transport} transport requires 'url'."
        )

    if config.transport == "sse":
        sse_params: MCPServerSseParams = {"url": config.url}
        if config.headers:
            sse_params["headers"] = dict(config.headers)
        return MCPServerSse(
            params=sse_params,
            name=config.name,
            cache_tools_list=config.cache_tools_list,
            client_session_timeout_seconds=config.connect_timeout_seconds,
            tool_filter=tool_filter,
        )

    http_params: MCPServerStreamableHttpParams = {"url": config.url}
    if config.headers:
        http_params["headers"] = dict(config.headers)
    return MCPServerStreamableHttp(
        params=http_params,
        name=config.name,
        cache_tools_list=config.cache_tools_list,
        client_session_timeout_seconds=config.connect_timeout_seconds,
        tool_filter=tool_filter,
    )
