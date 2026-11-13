"""Phase 03 - Evaluator-optimizer loop with a JSON evaluator."""

import json

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

MAX_ATTEMPTS = 3

CLAIM = """
A pipe burst and damaged the kitchen ceiling.
Photographs are available. A repair estimate is missing.
The incident occurred yesterday.
""".strip()


def generate_draft(client: OllamaClient, feedback: str = "") -> str:
    result = client.chat(
        f"""
Summarize the claim in 40 to 80 words.

Requirements:
- Use only stated facts.
- Mention missing evidence.
- Do not infer coverage or liability.
- Do not approve or deny the claim.

Claim:
{CLAIM}

Feedback from previous evaluation:
{feedback}

Return only the summary.
""".strip(),
        temperature=0,
        max_tokens=220,
    )

    return result.text


def evaluate_draft(client: OllamaClient, draft: str) -> dict:
    result = client.chat(
        f"""
Evaluate the draft against the original claim.

Original claim:
{CLAIM}

Draft:
{draft}

Check:
1. Word count is between 40 and 80 inclusive.
2. The draft does not introduce unsupported facts.
3. Missing evidence is identified.
4. No approval or denial decision is made.

Return JSON with:
- passed: true or false
- word_count: integer
- unsupported_claims: list of strings
- missing_evidence_identified: true or false
- feedback: string
""".strip(),
        temperature=0,
        json_mode=True,
        max_tokens=220,
    )

    try:
        evaluation = json.loads(result.text)

        required = {
            "passed",
            "word_count",
            "unsupported_claims",
            "missing_evidence_identified",
            "feedback",
        }

        if not required.issubset(evaluation):
            raise ValueError("Evaluator omitted required fields.")

        if not isinstance(evaluation["passed"], bool):
            raise ValueError("'passed' must be a boolean.")

        if not isinstance(evaluation["unsupported_claims"], list):
            raise ValueError("'unsupported_claims' must be a list.")

        if not isinstance(evaluation["missing_evidence_identified"], bool):
            raise ValueError(
                "'missing_evidence_identified' must be a boolean."
            )

        return evaluation

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "passed": False,
            "word_count": len(draft.split()),
            "unsupported_claims": [],
            "missing_evidence_identified": False,
            "feedback": f"Invalid evaluator response: {exc}",
        }


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    feedback = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")

        draft = generate_draft(client, feedback)
        evaluation = evaluate_draft(client, draft)

        print("\nDraft:\n", draft)
        print("\nEvaluation:\n", json.dumps(evaluation, indent=2))

        word_count = len(draft.split())
        mechanically_valid = 40 <= word_count <= 80

        if evaluation["passed"] and mechanically_valid:
            print("\nEvaluator accepted the draft.")
            print("Note: acceptance is not proof of factual correctness.")
            return

        feedback = evaluation["feedback"]

        if not mechanically_valid:
            feedback += (
                f" The draft has {word_count} words; "
                "the required range is 40 to 80."
            )

    print("\nStopped: attempt budget exhausted.")


if __name__ == "__main__":
    main()
