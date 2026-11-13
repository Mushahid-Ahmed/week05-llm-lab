"""Phase 02 - Compare zero-shot and few-shot classification."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.prompts import (
    few_shot_label_prompt,
    normalize_label,
    zero_shot_label_prompt,
)

TEST_CLAIMS = [
    {
        "claim": "My car was hit by another driver in a parking lot.",
        "reference": "vehicle",
    },
    {
        "claim": "A pipe burst and flooded my kitchen.",
        "reference": "property",
    },
    {
        "claim": "My suitcase was lost during a flight.",
        "reference": "travel",
    },
    {
        "claim": "My laptop was damaged while I was using it at home.",
        "reference": "context-dependent",
    },
    {
        "claim": "I needed medical treatment after a road accident.",
        "reference": "context-dependent",
    },
]


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    rows = []

    for item in TEST_CLAIMS:
        zero = client.chat(
            zero_shot_label_prompt(item["claim"]),
            temperature=0,
            max_tokens=24,
        )
        few = client.chat(
            few_shot_label_prompt(item["claim"]),
            temperature=0,
            max_tokens=24,
        )

        zero_label = normalize_label(zero.text)
        few_label = normalize_label(few.text)
        rows.append((item, zero_label, few_label))

        print("Test input:", item["claim"])
        print(f"  Zero-shot: {zero_label:18} ({zero.latency_seconds:.2f}s)")
        print(f"  Few-shot:  {few_label:18} ({few.latency_seconds:.2f}s)")
        print(f"  Reference: {item['reference']}")
        print("-" * 70)

    zero_correct = sum(
        zero == item["reference"] for item, zero, _ in rows
    )
    few_correct = sum(
        few == item["reference"] for item, _, few in rows
    )

    print("\nSUMMARY")
    print(f"Zero-shot matches: {zero_correct}/{len(rows)}")
    print(f"Few-shot matches:  {few_correct}/{len(rows)}")
    print(
        "\nObservation: examples can improve consistency, but they also\n"
        "bias the model toward the categories they demonstrate.\n"
        "Ambiguous inputs expose that bias."
    )


if __name__ == "__main__":
    main()
