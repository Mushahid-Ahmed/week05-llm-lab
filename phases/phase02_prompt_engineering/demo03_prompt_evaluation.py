"""Phase 02 - Evaluate baseline and enhanced prompts on the same claims."""

import json
from pathlib import Path

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.prompts import analysis_prompt
from llm_lab.schemas import ClaimAnalysis

DATASET = json.loads(
    Path("data/claims.json").read_text(encoding="utf-8")
)

OUTPUT = Path("evidence/prompt_evaluation.jsonl")


def baseline_prompt(claim: str) -> str:
    return f"""
Analyze this insurance claim.

Claim: {claim}
""".strip()


def parse(raw: str) -> ClaimAnalysis | None:
    try:
        return ClaimAnalysis.model_validate_json(raw)
    except ValueError:
        return None


def missing_recall(found: list[str], required: list[str]) -> float:
    joined = " ".join(found).lower()

    if not required:
        return 1.0

    hits = sum(
        1 for item in required
        if item.lower() in joined
        or all(word in joined for word in item.lower().split())
    )

    return hits / len(required)


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    rows = []

    for item in DATASET:
        for version, prompt, json_mode in [
            ("baseline", baseline_prompt(item["claim"]), False),
            ("enhanced", analysis_prompt(item["claim"]), True),
        ]:
            result = client.chat(
                prompt,
                temperature=0,
                json_mode=json_mode,
                max_tokens=200,
            )
            analysis = parse(result.text)

            category = analysis.category if analysis else "invalid-output"
            recall = (
                missing_recall(
                    analysis.missing_information,
                    item["required_missing"],
                )
                if analysis
                else 0.0
            )

            row = {
                "id": item["id"],
                "version": version,
                "claim": item["claim"],
                "reference": item["reference"],
                "raw_output": result.text,
                "valid_json": analysis is not None,
                "category": category,
                "correct": category == item["reference"],
                "missing_recall": round(recall, 2),
                "latency_seconds": round(result.latency_seconds, 2),
            }
            rows.append(row)

            print(
                f"Claim {item['id']} | {version:8} | "
                f"valid={row['valid_json']!s:5} | "
                f"category={category:18} | "
                f"missing recall={recall:.0%} | "
                f"{row['latency_seconds']:.2f}s"
            )

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("=" * 70)

    for version in ("baseline", "enhanced"):
        group = [row for row in rows if row["version"] == version]

        print(
            f"{version:8} | valid output rate: "
            f"{sum(r['valid_json'] for r in group) / len(group):.0%} | "
            f"category accuracy: "
            f"{sum(r['correct'] for r in group) / len(group):.0%} | "
            f"missing-info recall: "
            f"{sum(r['missing_recall'] for r in group) / len(group):.0%} | "
            f"mean latency: "
            f"{sum(r['latency_seconds'] for r in group) / len(group):.2f}s"
        )

    print(f"\nDetailed rows saved to {OUTPUT}")
    print(
        "\nObservation: quality changes must be weighed against latency.\n"
        "A five-claim sample is a signal, not proof; a larger unseen set\n"
        "is needed before treating the difference as meaningful."
    )


if __name__ == "__main__":
    main()
