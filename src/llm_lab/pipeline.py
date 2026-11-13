from dataclasses import dataclass, field

from llm_lab.client import OllamaClient
from llm_lab.config import MAX_ATTEMPTS
from llm_lab.prompts import analysis_prompt
from llm_lab.schemas import ClaimAnalysis, parse_analysis


class ControlledFailure(RuntimeError):
    """Raised when the retry budget is exhausted without a valid result."""


@dataclass
class PipelineResult:
    analysis: ClaimAnalysis
    attempts: int
    total_latency_seconds: float
    repair_feedback: list[str] = field(default_factory=list)


def analyze_claim(
    claim: str,
    client: OllamaClient,
    max_attempts: int = MAX_ATTEMPTS,
) -> PipelineResult:
    """Analyze a claim with schema validation and bounded repair."""

    if not claim.strip():
        raise ValueError("Claim cannot be empty.")

    feedback = ""
    repairs: list[str] = []
    total_latency = 0.0

    for attempt in range(1, max_attempts + 1):
        result = client.chat(
            analysis_prompt(claim, feedback),
            json_mode=True,
        )
        total_latency += result.latency_seconds

        try:
            analysis = parse_analysis(result.text)
        except ValueError as exc:
            feedback = str(exc)
            repairs.append(feedback)
            print(f"Attempt {attempt}: rejected -> {feedback}")
            continue

        print(f"Attempt {attempt}: accepted")
        return PipelineResult(
            analysis=analysis,
            attempts=attempt,
            total_latency_seconds=total_latency,
            repair_feedback=repairs,
        )

    raise ControlledFailure(
        f"No schema-valid result after {max_attempts} attempts. "
        f"Last error: {feedback}"
    )
