"""Phase 01 - Separate plausible statements from supported evidence."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

SOURCE = """
The customer has provided photographs.
A repair estimate has not been supplied.
""".strip()

DRAFT = """
The customer has submitted all required documents.
The claim should be approved.
""".strip()


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")
    print("SOURCE INFORMATION")
    print(SOURCE)
    print("\nDRAFT UNDER REVIEW")
    print(DRAFT)

    prompt = f"""
You are auditing a draft against source information.

SOURCE INFORMATION:
{SOURCE}

DRAFT:
{DRAFT}

Evaluate the draft against these criteria:
1. Does each factual statement follow from the source?
2. Does the draft invent missing information?
3. Does it make an unauthorized final claim decision?
4. What should be corrected?

For every statement in the draft, classify it as:
- Supported by the input
- Reasonable but not established
- Unsupported

Return a concise audit.
""".strip()

    result = client.chat(prompt, temperature=0, max_tokens=200)

    print("\nMODEL AUDIT")
    print("=" * 70)
    print(result.text)
    print(f"\nLatency: {result.latency_seconds:.2f}s")
    print(
        "\nObservation: the draft asserts that all documents were supplied\n"
        "and recommends approval. The source states that a repair estimate\n"
        "is missing. A fluent draft can still be wrong."
    )


if __name__ == "__main__":
    main()
