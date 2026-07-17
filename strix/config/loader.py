"""Settings loader, override switch, and disk persistence."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, BaseModel

from strix.config.settings import Settings


if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


logger = logging.getLogger(__name__)


_DEFAULT_PATH: Path = Path.home() / ".strix" / "cli-config.json"
_override: Path | None = None
_cached: Settings | None = None


def load_settings() -> Settings:
    """Resolve settings from env + JSON file + defaults. Memoized.

    Precedence: env vars win, then the JSON file, then field defaults.
    """
    global _cached  # noqa: PLW0603
    if _cached is None:
        source_path = _override or _DEFAULT_PATH
        init_kwargs: dict[str, Any] = _read_json_overrides(source_path)
        _cached = Settings(**init_kwargs)
        logger.debug(
            "load_settings: resolved (override=%s, file_used=%s, json_keys=%d)",
            _override is not None,
            source_path.exists(),
            sum(len(v) for v in init_kwargs.values()),
        )
    return _cached


def apply_config_override(path: Path) -> None:
    """Switch the JSON source to ``path`` and invalidate the cache."""
    global _override, _cached  # noqa: PLW0603
    _override = path
    _cached = None
    logger.info("config override applied: %s", path)


def persist_current() -> None:
    """Write currently-set env vars to the active config file (0o600).

    Preserves any ``mcp_servers`` array already in the file so persisting
    secrets never clobbers the user's MCP configuration.
    """
    s = load_settings()
    target = _override or _DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    env_block: dict[str, str] = {}
    for sub_name in s.model_fields:
        sub_model = getattr(s, sub_name)
        if not isinstance(sub_model, BaseModel):
            continue
        for finfo in type(sub_model).model_fields.values():
            for alias in _aliases_for(finfo):
                value = os.environ.get(alias.upper())
                if value:
                    env_block[alias.upper()] = value
                    break

    payload: dict[str, Any] = {"env": env_block}
    preserved = _existing_mcp_servers(target)
    if preserved is not None:
        payload["mcp_servers"] = preserved

    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        target.chmod(0o600)


def _existing_mcp_servers(path: Path) -> list[Any] | None:
    """Return the ``mcp_servers`` array already stored in ``path``, if any."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    servers = data.get("mcp_servers")
    return servers if isinstance(servers, list) else None


def _aliases_for(finfo: FieldInfo) -> list[str]:
    """Collect every env-var name that should populate ``finfo``."""
    aliases: list[str] = []
    if finfo.alias:
        aliases.append(finfo.alias)
    va = finfo.validation_alias
    if isinstance(va, AliasChoices):
        aliases.extend(c for c in va.choices if isinstance(c, str))
    elif isinstance(va, str):
        aliases.append(va)
    return aliases


def _read_json_overrides(path: Path) -> dict[str, Any]:
    """Read ``{"env": {...}, "mcp_servers": [...]}`` from ``path`` and remap.

    ``env`` maps to nested sub-model kwargs (an env var still wins over the
    file for any field it sets). ``mcp_servers`` is passed straight through to
    ``Settings.mcp_servers``. Only includes env keys whose var is NOT already
    set, so the process environment always wins over the persisted file.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    env_block = data.get("env", {})
    if not isinstance(env_block, dict):
        env_block = {}

    env_block_upper = {str(k).upper(): v for k, v in env_block.items()}
    env_present = {k.upper() for k in os.environ}

    nested: dict[str, Any] = {}
    for sub_name, sub_finfo in Settings.model_fields.items():
        sub_cls = sub_finfo.annotation
        if not (isinstance(sub_cls, type) and issubclass(sub_cls, BaseModel)):
            continue
        sub_data: dict[str, Any] = {}
        for fname, finfo in sub_cls.model_fields.items():
            aliases = [alias.upper() for alias in _aliases_for(finfo)]
            if any(alias in env_present for alias in aliases):
                continue  # env wins under some alias; skip the JSON file for this field
            for alias in aliases:
                if alias in env_block_upper:
                    sub_data[fname] = env_block_upper[alias]
                    break
        if sub_data:
            nested[sub_name] = sub_data

    mcp_servers = data.get("mcp_servers")
    if isinstance(mcp_servers, list):
        nested["mcp_servers"] = mcp_servers

    return nested
