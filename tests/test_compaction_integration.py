"""Integration tests: filter registers on RunConfig, hooks feed the store."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agents import RunConfig
from agents.run_config import CallModelData, ModelInputData

from strix.config.settings import Settings
from strix.core.compaction.config import CompactionConfig
from strix.core.compaction.filter import build_compaction_filter
from strix.core.compaction.store import CompactionStore, get_store
from strix.core.hooks import ReportUsageHooks


@pytest.mark.asyncio
async def test_filter_registers_on_run_config_and_returns_model_input_data() -> None:
    settings = Settings(compaction={"reserved_output": 1000})
    cfg = CompactionConfig.from_env(session_model="openai/gpt-4o", settings=settings)
    store = CompactionStore()
    run_config = RunConfig(call_model_input_filter=build_compaction_filter(cfg, store))
    assert run_config.call_model_input_filter is not None

    model_data = ModelInputData(input=[{"role": "user", "content": "hi"}], instructions="SYS")
    agent = type("A", (), {"name": "strix"})()
    payload: CallModelData[Any] = CallModelData(
        model_data=model_data, agent=agent, context={"agent_id": "root"}
    )
    result = await run_config.call_model_input_filter(payload)
    assert isinstance(result, ModelInputData)


@pytest.mark.asyncio
async def test_hooks_record_input_tokens_into_store() -> None:
    hooks = ReportUsageHooks(model="openai/gpt-4o", max_budget_usd=None)
    state = MagicMock()
    state.record_sdk_usage = MagicMock()
    state.get_total_llm_cost.return_value = 0.0
    ctx = MagicMock()
    ctx.context = {"agent_id": "agent-xyz"}
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=4242))
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(ctx, MagicMock(), response)
    assert get_store().agent_input_tokens("agent-xyz") == 4242
