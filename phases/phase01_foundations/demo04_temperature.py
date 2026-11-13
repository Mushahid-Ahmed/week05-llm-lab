"""Phase 01 - Observe how temperature changes output."""

import re

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

PROMPT = """
Explain why software systems validate LLM output.
Use exactly three bullet points.
""".strip()


def count_bullets(text: str) -> int:
    return len(
        [
            line for line in text.splitlines()
            if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
        ]
    )


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    for temperature in (0, 0.7, 1.0):
        print("=" * 70)
        print(f"Temperature: {temperature}")

        for run in range(1, 3):
            result = client.chat(
                PROMPT, temperature=temperature, max_tokens=200
            )
            bullets = count_bullets(result.text)

            print(f"\nRun {run} ({result.latency_seconds:.2f}s, "
                  f"{bullets} bullet(s)):")
            print(result.text)

    print(
        "\nObservation: temperature changes sampling, not correctness.\n"
        "Check whether every run obeyed the three-bullet requirement at\n"
        "each temperature."
    )


if __name__ == "__main__":
    main()
