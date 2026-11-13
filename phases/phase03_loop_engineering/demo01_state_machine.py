"""Phase 03 - Explicit state machine for one generate-validate cycle."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

MIN_WORDS = 40

CLAIM = """
A pipe burst in a house and damaged the kitchen ceiling.
Photographs are available, but a repair estimate has not
been obtained. The damage occurred yesterday.
""".strip()


def generate_summary(client: OllamaClient) -> str:
    result = client.chat(
        f"""
Summarize this claim in at least {MIN_WORDS} words.
Use only the supplied facts. Do not invent details.

Claim:
{CLAIM}
""".strip(),
        temperature=0,
        max_tokens=140,
    )

    return result.text


def validate(summary: str) -> bool:
    return len(summary.split()) >= MIN_WORDS


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    state = "START"
    summary = ""

    while state != "DONE":
        print(f"\nCurrent state: {state}")

        if state == "START":
            state = "GENERATE"

        elif state == "GENERATE":
            summary = generate_summary(client)
            print("Generated summary:", summary)
            state = "VALIDATE"

        elif state == "VALIDATE":
            state = "SUCCESS" if validate(summary) else "FAILURE"

        elif state == "SUCCESS":
            print(f"Validation passed ({len(summary.split())} words).")
            state = "DONE"

        elif state == "FAILURE":
            print(
                f"Validation failed ({len(summary.split())} words). "
                "No retry is configured."
            )
            state = "DONE"

    print("\nFinal state:", state)


if __name__ == "__main__":
    main()
