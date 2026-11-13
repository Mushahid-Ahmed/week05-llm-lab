"""Phase 03 - Generate, critique, then revise."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

CLAIM = """
A pipe burst upstairs and damaged the kitchen ceiling.
The homeowner has photographs but no repair estimate.
The incident occurred yesterday.
""".strip()


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    draft = client.chat(
        f"""
Write a concise preliminary summary of this claim.

Use only stated facts. Identify missing information.
Do not infer coverage or liability.

Claim:
{CLAIM}
""".strip(),
        temperature=0,
    )

    print("\n--- DRAFT ---\n")
    print(draft.text)
    print(f"(latency {draft.latency_seconds:.2f}s)")

    critique = client.chat(
        f"""
Review the draft against the original claim.

Original claim:
{CLAIM}

Draft:
{draft.text}

Identify:
- Unsupported factual assertions.
- Important omitted facts.
- Missing information not identified.
- Any claim of coverage, liability, approval, or denial.

Do not rewrite the draft. Return a concise critique.
""".strip(),
        temperature=0,
    )

    print("\n--- CRITIQUE ---\n")
    print(critique.text)
    print(f"(latency {critique.latency_seconds:.2f}s)")

    revised = client.chat(
        f"""
Revise the draft using the critique.

Original claim:
{CLAIM}

Draft:
{draft.text}

Critique:
{critique.text}

Rules:
- Preserve supported facts.
- Remove unsupported assertions.
- Identify missing information.
- Do not infer coverage or liability.
- Do not approve or deny the claim.
- If the critique is incorrect, do not follow it blindly.

Return only the revised summary.
""".strip(),
        temperature=0,
    )

    print("\n--- REVISED SUMMARY ---\n")
    print(revised.text)
    print(f"(latency {revised.latency_seconds:.2f}s)")

    print(
        "\nObservation: the critique is itself model output. It can be\n"
        "wrong, so the revision must be checked against the claim."
    )


if __name__ == "__main__":
    main()
