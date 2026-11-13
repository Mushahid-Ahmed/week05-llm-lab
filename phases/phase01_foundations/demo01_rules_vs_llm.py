"""Phase 01 - Compare a fixed rule classifier with an LLM."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL


def rule_classifier(text: str) -> str:
    text = text.lower()

    if "accident" in text or "collision" in text:
        return "vehicle accident"

    if "water leak" in text or "burst pipe" in text:
        return "property water damage"

    return "unknown"


EXAMPLES = [
    "A delivery van struck my parked car.",
    "Water from a broken pipe damaged my kitchen.",
    "My car was damaged after another driver hit it.",
]


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    for example in EXAMPLES:
        print("INPUT:", example)
        print("RULE:", rule_classifier(example))

        result = client.chat(
            "Classify this insurance claim in one short phrase. "
            "Do not invent details.\n\n"
            f"Claim: {example}",
            temperature=0,
            max_tokens=32,
        )

        print("LLM: ", result.text.replace("\n", " "))
        print(f"Latency: {result.latency_seconds:.2f}s")
        print("-" * 70)

    print(
        "\nObservation: the rules return 'unknown' whenever the wording\n"
        "does not contain a listed keyword, while the LLM interprets all\n"
        "three descriptions. The rule output is reproducible and testable;\n"
        "the LLM output is plausible but must still be checked."
    )


if __name__ == "__main__":
    main()
