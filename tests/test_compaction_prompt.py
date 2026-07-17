"""Tests for summarizer prompt constants."""

from __future__ import annotations

from strix.core.compaction.prompt import (
    PREVIOUS_SUMMARY_CLOSE,
    PREVIOUS_SUMMARY_OPEN,
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_TEMPLATE,
    SUMMARY_MARKER,
    SUMMARY_SECTIONS,
    SUMMARY_TEMPLATE,
    summary_message,
)


def test_required_sections_present_in_template() -> None:
    expected = (
        "## Objective",
        "## Important Details",
        "## Findings / Artifacts",
        "## Tested / Ruled Out",
        "## Work State",
        "## Next Move",
        "## Relevant Files",
    )
    assert expected == SUMMARY_SECTIONS
    for section in SUMMARY_SECTIONS:
        assert section in SUMMARY_TEMPLATE
    for sub in ("### Completed", "### Active", "### Blocked"):
        assert sub in SUMMARY_TEMPLATE


def test_tested_ruled_out_section_after_findings() -> None:
    assert "## Tested / Ruled Out" in SUMMARY_SECTIONS
    assert "## Tested / Ruled Out" in SUMMARY_TEMPLATE

    sections = list(SUMMARY_SECTIONS)
    assert sections.index("## Tested / Ruled Out") == sections.index("## Findings / Artifacts") + 1

    findings_at = SUMMARY_TEMPLATE.index("## Findings / Artifacts")
    tested_at = SUMMARY_TEMPLATE.index("## Tested / Ruled Out")
    work_state_at = SUMMARY_TEMPLATE.index("## Work State")
    assert findings_at < tested_at < work_state_at


def test_system_prompt_names_key_rules() -> None:
    lowered = SUMMARIZER_SYSTEM_PROMPT.lower()
    assert "previous-summary" in lowered
    assert "verbatim" in lowered
    # must instruct never to reveal that context was compacted
    assert "compact" in lowered


def test_user_template_has_history_placeholder() -> None:
    assert "{history}" in SUMMARIZER_USER_TEMPLATE


def test_previous_summary_tags() -> None:
    assert PREVIOUS_SUMMARY_OPEN == "<previous-summary>"
    assert PREVIOUS_SUMMARY_CLOSE == "</previous-summary>"


def test_summary_message_shape() -> None:
    msg = summary_message("BODY")
    assert msg["role"] == "user"
    assert msg["content"].startswith(SUMMARY_MARKER)
    assert "BODY" in msg["content"]
