"""Per-agent compaction state: summary, anchor, force flag, calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

from strix.core.compaction.tokens import Calibrator


@dataclass
class AgentCompactionState:
    summary: str | None = None
    anchor: int = 0
    force: bool = False
    last_input_tokens: int = 0
    pending_chars: int = 0
    calibrator: Calibrator = field(default_factory=Calibrator)


class CompactionStore:
    def __init__(self) -> None:
        self._agents: dict[str, AgentCompactionState] = {}
        self._latest_input_tokens = 0

    def state(self, agent_id: str) -> AgentCompactionState:
        return self._agents.setdefault(agent_id, AgentCompactionState())

    def request_compaction(self, agent_id: str) -> None:
        self.state(agent_id).force = True

    def consume_force(self, agent_id: str) -> bool:
        st = self.state(agent_id)
        was = st.force
        st.force = False
        return was

    def set_summary(self, agent_id: str, summary: str, anchor: int) -> None:
        st = self.state(agent_id)
        st.summary = summary
        st.anchor = anchor

    def record_pending_chars(self, agent_id: str, chars: int) -> None:
        self.state(agent_id).pending_chars = chars

    def record_real_usage(self, agent_id: str, input_tokens: int) -> None:
        st = self.state(agent_id)
        st.last_input_tokens = input_tokens
        if st.pending_chars > 0:
            st.calibrator.update(real_tokens=input_tokens, chars=st.pending_chars)
        self._latest_input_tokens = input_tokens

    def ratio(self, agent_id: str) -> float | None:
        return self.state(agent_id).calibrator.ratio

    def agent_input_tokens(self, agent_id: str) -> int:
        return self.state(agent_id).last_input_tokens

    def latest_input_tokens(self) -> int:
        return self._latest_input_tokens


_STORE = CompactionStore()


def get_store() -> CompactionStore:
    return _STORE
