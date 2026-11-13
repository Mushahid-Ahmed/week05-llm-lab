from dataclasses import dataclass, field

import pytest

from llm_lab.client import LLMResult
from llm_lab.pipeline import ControlledFailure, analyze_claim

VALID = """
{
  "category": "property",
  "summary": "A pipe burst damaged the kitchen ceiling.",
  "missing_information": ["repair estimate"],
  "needs_human_review": true,
  "rationale": "Household water damage."
}
"""

INVALID = '{"category": "fire", "summary": "Unsupported category."}'


@dataclass
class FakeClient:
    responses: list[str]
    prompts: list[str] = field(default_factory=list)
    latency: float = 0.01
    json_modes: list[bool] = field(default_factory=list)

    def chat(self, prompt, temperature=0.0, json_mode=False, system=None):
        self.prompts.append(prompt)
        self.json_modes.append(json_mode)
        return LLMResult(text=self.responses.pop(0), latency_seconds=self.latency)


def test_valid_first_response_is_returned() -> None:
    client = FakeClient([VALID])

    result = analyze_claim("A pipe burst.", client)

    assert result.analysis.category == "property"
    assert result.attempts == 1
    assert result.repair_feedback == []
    assert all(client.json_modes)


def test_invalid_response_is_repaired_on_retry() -> None:
    client = FakeClient([INVALID, VALID])

    result = analyze_claim("A pipe burst.", client)

    assert result.attempts == 2
    assert len(result.repair_feedback) == 1
    assert "previous attempt failed" in client.prompts[1].lower()


def test_budget_exhaustion_raises_controlled_failure() -> None:
    client = FakeClient([INVALID, INVALID, INVALID])

    with pytest.raises(ControlledFailure, match="after 3 attempts"):
        analyze_claim("A pipe burst.", client, max_attempts=3)


def test_empty_claim_is_rejected_before_any_call() -> None:
    client = FakeClient([VALID])

    with pytest.raises(ValueError, match="cannot be empty"):
        analyze_claim("   ", client)

    assert client.prompts == []


def test_result_reports_total_latency() -> None:
    client = FakeClient([INVALID, VALID], latency=0.5)

    result = analyze_claim("A pipe burst.", client)

    assert result.total_latency_seconds == pytest.approx(1.0)
