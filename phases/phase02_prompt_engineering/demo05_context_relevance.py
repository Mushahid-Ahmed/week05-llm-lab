"""Phase 02 - Relevant versus irrelevant context."""

import csv
import time
from collections import Counter
from pathlib import Path

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.prompts import CATEGORIES, normalize_label

REPEATS = 3
OUTPUT = Path("evidence/context_results.csv")

RELEVANT_CONTEXT = """
Claim category definitions:
- vehicle: Damage to or collision involving a vehicle.
- property: Damage to a home, building, or household property.
- travel: Travel-related loss, such as lost luggage.
- health: Medical treatment or expenses.
- other: The claim does not fit the listed categories.
- context-dependent: The information is insufficient to assign
  a reliable category.
""".strip()

IRRELEVANT_CONTEXT = RELEVANT_CONTEXT + """

Background paragraph 1 - Insurance history:
Insurance has a long history of transferring financial risk.
Different societies have developed many approaches to sharing
losses. Historical insurance practices changed over time as
commercial activity and legal systems developed.

Background paragraph 2 - Vehicle maintenance:
Vehicles may require oil changes, tire inspections, brake
maintenance, battery checks, and periodic servicing. Maintenance
schedules depend on vehicle type, age, usage, and manufacturer
recommendations.

Background paragraph 3 - Unrelated customer complaints:
Some customers complain about long telephone waiting times.
Others ask about website navigation, office opening hours,
billing notifications, or how to update their contact details.
These complaints are not necessarily related to an insurance claim.

Background paragraph 4 - General insurance administration:
Insurance organizations may use forms, customer portals,
document-management systems, and telephone support teams.
Administrative workflows vary between organizations.
""".strip()

TEST_CLAIMS = [
    {
        "id": 1,
        "claim": "My car collided with another vehicle at a junction.",
        "reference": "vehicle",
    },
    {
        "id": 2,
        "claim": "A pipe burst and damaged the walls of my house.",
        "reference": "property",
    },
    {
        "id": 3,
        "claim": "My suitcase was lost during an international flight.",
        "reference": "travel",
    },
    {
        "id": 4,
        "claim": "I received medical treatment after an accident.",
        "reference": "health",
    },
    {
        "id": 5,
        "claim": "Something was damaged, but I have not explained what it was.",
        "reference": "context-dependent",
    },
]


def build_prompt(context: str, claim: str) -> str:
    return f"""
TASK:
Classify the claim using the supplied category definitions.

CONTEXT:
{context}

CLAIM:
<claim>
{claim}
</claim>

RULES:
- Use only the categories defined in the context.
- Do not invent facts.
- If the information is insufficient, use context-dependent.
- Return exactly one category label and nothing else.
""".strip()


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    rows = []

    for item in TEST_CLAIMS:
        for context_name, context in [
            ("A_relevant", RELEVANT_CONTEXT),
            ("B_extra_irrelevant", IRRELEVANT_CONTEXT),
        ]:
            for repeat in range(1, REPEATS + 1):
                start = time.perf_counter()
                result = client.chat(
                    build_prompt(context, item["claim"]),
                    temperature=0,
                    max_tokens=24,
                )
                elapsed = time.perf_counter() - start
                label = normalize_label(result.text)

                rows.append({
                    "claim_id": item["id"],
                    "claim": item["claim"],
                    "context": context_name,
                    "repeat": repeat,
                    "reference": item["reference"],
                    "label": label,
                    "latency_seconds": round(elapsed, 3),
                    "output_chars": len(result.text),
                    "raw_output": result.text,
                })

                print(
                    f"Claim {item['id']} | {context_name:19} | "
                    f"run {repeat} | {label:18} | {elapsed:.2f}s"
                )

    OUTPUT.parent.mkdir(exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY")
    print("=" * 80)

    for item in TEST_CLAIMS:
        for context_name in ("A_relevant", "B_extra_irrelevant"):
            group = [
                row for row in rows
                if row["claim_id"] == item["id"]
                and row["context"] == context_name
            ]

            labels = [row["label"] for row in group]
            most_common, count = Counter(labels).most_common(1)[0]
            accuracy = sum(
                row["label"] == row["reference"] for row in group
            ) / len(group)
            mean_latency = sum(
                row["latency_seconds"] for row in group
            ) / len(group)
            mean_chars = sum(
                row["output_chars"] for row in group
            ) / len(group)

            print(
                f"Claim {item['id']} | {context_name:19} | "
                f"accuracy {accuracy:.0%} | "
                f"consistency {count / len(group):.0%} | "
                f"mean latency {mean_latency:.2f}s | "
                f"mean chars {mean_chars:.0f}"
            )

    print(f"\nDetailed results saved to {OUTPUT}")
    print(f"Allowed labels: {CATEGORIES}")


if __name__ == "__main__":
    main()
