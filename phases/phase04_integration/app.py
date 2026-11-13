"""Phase 04 - Integrated, validated claim-analysis application."""

import argparse
import json
import sys
from pathlib import Path

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.pipeline import ControlledFailure, analyze_claim
from llm_lab.schemas import ClaimAnalysis

DATASET = json.loads(
    Path("data/claims.json").read_text(encoding="utf-8")
)


def render(claim: str, analysis: ClaimAnalysis) -> str:
    return json.dumps(
        {
            "claim": claim,
            "analysis": analysis.model_dump(),
        },
        indent=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preliminary claim triage with a validated LLM pipeline."
    )
    parser.add_argument("claim", nargs="?", help="Claim text to analyze")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the first three claims from data/claims.json",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Override the repair budget",
    )
    args = parser.parse_args()

    if not args.claim and not args.demo:
        parser.error("provide a claim or use --demo")

    claims = (
        [item["claim"] for item in DATASET[:3]]
        if args.demo
        else [args.claim]
    )

    client = OllamaClient()
    failures = 0

    for claim in claims:
        print("=" * 70)
        print("CLAIM:", claim)

        try:
            kwargs = {}
            if args.max_attempts is not None:
                kwargs["max_attempts"] = args.max_attempts

            result = analyze_claim(claim, client, **kwargs)

        except ValueError as exc:
            print(f"INPUT REJECTED: {exc}")
            failures += 1
            continue

        except ControlledFailure as exc:
            print(f"CONTROLLED FAILURE: {exc}")
            failures += 1
            continue

        print(render(claim, result.analysis))
        print(
            f"Attempts: {result.attempts} | "
            f"Model latency: {result.total_latency_seconds:.2f}s"
        )

        if result.repair_feedback:
            print("Repair feedback used:")
            for note in result.repair_feedback:
                print(f"  - {note}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
