"""Tests for the per-agent compaction store."""

from __future__ import annotations

from strix.core.compaction.store import CompactionStore, get_store


def test_force_flag_set_and_consumed() -> None:
    store = CompactionStore()
    assert store.consume_force("a") is False
    store.request_compaction("a")
    assert store.consume_force("a") is True
    # consuming clears it
    assert store.consume_force("a") is False


def test_summary_and_anchor_roundtrip() -> None:
    store = CompactionStore()
    store.set_summary("a", "SUMMARY", 12)
    st = store.state("a")
    assert st.summary == "SUMMARY"
    assert st.anchor == 12


def test_real_usage_updates_ratio_and_latest() -> None:
    store = CompactionStore()
    store.record_pending_chars("a", 2000)
    store.record_real_usage("a", 1000)
    assert store.ratio("a") == 0.5
    assert store.agent_input_tokens("a") == 1000
    assert store.latest_input_tokens() == 1000


def test_latest_tracks_most_recent_agent() -> None:
    store = CompactionStore()
    store.record_pending_chars("a", 400)
    store.record_real_usage("a", 100)
    store.record_pending_chars("b", 800)
    store.record_real_usage("b", 222)
    assert store.latest_input_tokens() == 222
    assert store.agent_input_tokens("a") == 100


def test_get_store_is_singleton() -> None:
    assert get_store() is get_store()
