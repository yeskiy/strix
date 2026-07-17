"""Tests for todo tool helpers."""

from __future__ import annotations

import pytest

from strix.tools.todo.tools import VALID_PRIORITIES, _normalize_priority


def test_alias_medium_maps_to_normal() -> None:
    assert _normalize_priority("medium") == "normal"


def test_alias_is_case_insensitive_and_stripped() -> None:
    assert _normalize_priority("  Medium  ") == "normal"


def test_alias_p_levels() -> None:
    assert _normalize_priority("P1") == "high"
    assert _normalize_priority("p0") == "critical"
    assert _normalize_priority("P3") == "low"


def test_canonical_values_pass_through() -> None:
    for value in VALID_PRIORITIES:
        assert _normalize_priority(value) == value


def test_high_stays_high() -> None:
    assert _normalize_priority("high") == "high"


def test_default_when_missing() -> None:
    assert _normalize_priority(None) == "normal"
    assert _normalize_priority("") == "normal"


def test_unknown_priority_raises() -> None:
    with pytest.raises(ValueError, match="Invalid priority"):
        _normalize_priority("bogus")
