"""Tests for compaction settings and CompactionConfig resolution."""

from __future__ import annotations

import pytest

from strix.config.settings import Settings
from strix.core.compaction.config import HYSTERESIS_DEFAULT, CompactionConfig


def test_defaults_are_on() -> None:
    settings = Settings()
    cs = settings.compaction
    assert cs.enabled is True
    assert cs.threshold == pytest.approx(0.90)
    assert cs.tail_turns == 2
    assert cs.preserve_recent_tokens == 16000
    assert cs.prune is True
    assert cs.prune_protect_tokens == 40000
    assert cs.prune_min_reclaim == 20000
    assert cs.reserved_output is None
    assert cs.model is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_COMPACTION", "false")
    monkeypatch.setenv("STRIX_COMPACTION_THRESHOLD", "0.75")
    monkeypatch.setenv("STRIX_COMPACTION_TAIL_TURNS", "3")
    monkeypatch.setenv("STRIX_COMPACTION_MODEL", "openai/gpt-5.6")
    cs = Settings().compaction
    assert cs.enabled is False
    assert cs.threshold == pytest.approx(0.75)
    assert cs.tail_turns == 3
    assert cs.model == "openai/gpt-5.6"


def test_from_env_resolves_summary_model_default() -> None:
    # reserved_output passed explicitly; usable_window still uses the live gpt-4o window.
    settings = Settings(compaction={"reserved_output": 1000})
    cfg = CompactionConfig.from_env(session_model="openai/gpt-4o", settings=settings)
    assert cfg.summary_model == "openai/gpt-4o"
    assert cfg.session_model == "openai/gpt-4o"
    assert cfg.reserved_output == 1000
    assert cfg.hysteresis == pytest.approx(HYSTERESIS_DEFAULT)
    assert cfg.usable_window > 0  # gpt-4o window (128000) - 1000


def test_from_env_summary_model_override() -> None:
    settings = Settings(compaction={"reserved_output": 1000, "model": "deepseek/deepseek-v4-flash"})
    cfg = CompactionConfig.from_env(session_model="openai/gpt-4o", settings=settings)
    assert cfg.summary_model == "deepseek/deepseek-v4-flash"
    assert cfg.session_model == "openai/gpt-4o"
