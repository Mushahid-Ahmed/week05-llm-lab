"""Phase 03 - Bounded validation-retry loop."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

MIN_WORDS = 40
MAX_ATTEMPTS = 3

CLAIM = """
A pipe burst in a house and damaged the kitchen ceiling.
Photographs are available, but a repair estimate has not
been obtained. The damage occurred yesterday.
""".strip()


def generate_summary(client: OllamaClient, feedback: str = "") -> tuple[str, float]:
    result = client.chat(
        f"""
Write a summary of the insurance claim below.

Requirements:
- At least {MIN_WORDS} words.
- Use only supplied facts.
- Identify missing information.
- Do not infer coverage or liability.
- Do not approve or deny the claim.

Claim:
{CLAIM}

{feedback}

Return only the summary.
""".strip(),
        temperature=0,
        max_tokens=140,
    )

    return result.text, result.latency_seconds


def validate(text: str) -> tuple[bool, str]:
    word_count = len(text.split())

    if word_count < MIN_WORDS:
        return False, (
            f"Output has {word_count} words; minimum is {MIN_WORDS}."
        )

    return True, f"Word count passed: {word_count}."


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    feedback = ""
    total_latency = 0.0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")

        summary, latency = generate_summary(client, feedback)
        total_latency += latency

        valid, reason = validate(summary)

        print(summary)
        print("Validation:", reason)

        if valid:
            print("SUCCESS: returning validated output.")
            print(f"Total latency: {total_latency:.2f}s")
            return

        feedback = (
            f"Your previous response failed validation: {reason} "
            "Revise it to satisfy the requirement without inventing facts."
        )

    print(
        f"\nFAILED: maximum of {MAX_ATTEMPTS} attempts reached. "
        "The last output must not be treated as validated."
    )
    print(f"Total latency: {total_latency:.2f}s")


if __name__ == "__main__":
    main()
