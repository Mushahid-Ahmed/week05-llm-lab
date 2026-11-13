"""Phase 01 - Compare concise and noisy context for the same claim."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.prompts import normalize_label

CONCISE_PROMPT = """
Classify this insurance claim as vehicle, property, health,
travel, or other.

Claim: A tree fell on my parked car.
""".strip()

NOISY_PROMPT = """
You are an assistant. Be detailed. Think carefully.
Insurance is important. Claims may involve many situations.
Vehicles can be cars, buses, and trucks. Property can mean
buildings or belongings. Health claims may involve treatment.
Travel claims may involve luggage or cancellation.
Many things can happen to customers. Please classify the claim
carefully and explain everything relevant.

Claim: A tree fell on my parked car.
""".strip()

REPEATS = 3


def run(client: OllamaClient, label: str, prompt: str, repeat: int) -> dict:
    result = client.chat(prompt, temperature=0, max_tokens=80)
    category = normalize_label(result.text)

    print(f"{label} | run {repeat} | {category} | "
          f"{result.latency_seconds:.2f}s | "
          f"{len(result.text)} chars")

    return {
        "label": label,
        "category": category,
        "latency": result.latency_seconds,
        "chars": len(result.text),
    }


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    results = []

    for repeat in range(1, REPEATS + 1):
        results.append(run(client, "Concise", CONCISE_PROMPT, repeat))
        results.append(run(client, "Noisy  ", NOISY_PROMPT, repeat))

    print("\nSUMMARY")
    print("=" * 70)

    for label in ("Concise", "Noisy  "):
        group = [row for row in results if row["label"] == label]
        mean_latency = sum(row["latency"] for row in group) / len(group)
        mean_chars = sum(row["chars"] for row in group) / len(group)
        categories = {row["category"] for row in group}

        print(f"{label} | categories: {sorted(categories)} | "
              f"mean latency: {mean_latency:.2f}s | "
              f"mean output chars: {mean_chars:.0f}")

    print(
        "\nObservation: the noisy prompt adds no evidence about the claim.\n"
        "Compare whether it changed the label, added latency, or invited\n"
        "unsupported explanation."
    )


if __name__ == "__main__":
    main()
