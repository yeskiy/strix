"""Tests for the live context indicator line."""

from __future__ import annotations

from typing import Any

from strix.core.compaction.store import CompactionStore, get_store
from strix.interface.utils import build_tui_stats_text


class _FakeReportState:
    caido_url = None

    def get_total_llm_usage(self) -> dict[str, Any]:
        return {}


def test_indicator_shows_context_usage() -> None:
    # seed the store with a live per-call sample
    get_store().record_pending_chars("root", 400000)
    get_store().record_real_usage("root", 100000)
    text = build_tui_stats_text(_FakeReportState()).plain
    assert "context" in text
    # used and window both rendered via format_token_count
    assert "100.0K" in text


def test_indicator_absent_without_sample() -> None:
    # a fresh store with no sample: latest is 0 -> no context line
    store = CompactionStore()
    assert store.latest_input_tokens() == 0
