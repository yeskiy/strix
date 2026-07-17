"""Summarizer prompt: system role, template, rules, and the summary carrier message."""

from __future__ import annotations


SUMMARY_MARKER = "Context summary of earlier work in this session (older turns were compacted):"

PREVIOUS_SUMMARY_OPEN = "<previous-summary>"
PREVIOUS_SUMMARY_CLOSE = "</previous-summary>"

SUMMARY_SECTIONS = (
    "## Objective",
    "## Important Details",
    "## Findings / Artifacts",
    "## Tested / Ruled Out",
    "## Work State",
    "## Next Move",
    "## Relevant Files",
)

SUMMARY_TEMPLATE = """## Objective
## Important Details
## Findings / Artifacts
## Tested / Ruled Out
## Work State
### Completed
### Active
### Blocked
## Next Move
## Relevant Files"""

SUMMARIZER_SYSTEM_PROMPT = f"""\
You compress a security-testing conversation into a compact working memory.

Summarize only the history you are given. If a {PREVIOUS_SUMMARY_OPEN} block is present, \
treat it as the current anchor: keep facts that are still true, drop facts that later turns \
made stale, and merge in what is new - but never drop confirmed findings or tested/ruled-out \
coverage, which are always still relevant. Do not re-derive from scratch.

Output only the following Markdown structure, in this exact order, keeping every section even \
when it is empty:

{SUMMARY_TEMPLATE}

Rules:
- Use terse bullet points, not prose.
- Preserve exact identifiers verbatim: full URLs, file paths, endpoint paths, HTTP methods, \
parameter names, request payloads that worked, error strings, and finding IDs (for example \
vuln-0003). Never paraphrase or truncate these.
- Important Details holds constraints, decisions, and why each was made.
- Findings / Artifacts is where discovered endpoints, working payloads, parameter names, and \
finding identifiers are pinned so they survive compaction.
- Tested / Ruled Out records every check already performed and its outcome - endpoints, \
parameters, payloads, CVEs, versions, and auth flows probed - INCLUDING negative results \
(not vulnerable / not present / not reachable) so the same work is never repeated. Carry \
these forward across every summary.
- Do not continue or answer the conversation; you are only summarizing it.
- Never mention that the context was summarized or compacted.
- Reply in the same language the conversation uses."""

SUMMARIZER_USER_TEMPLATE = (
    "Summarize the conversation history below into the required structure.\n\n"
    "<history>\n{history}\n</history>"
)


def summary_message(summary_text: str) -> dict[str, str]:
    return {"role": "user", "content": f"{SUMMARY_MARKER}\n{summary_text}"}
