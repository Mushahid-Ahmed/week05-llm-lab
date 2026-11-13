"""Phase 02 - Build a reusable prompt from components."""

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL
from llm_lab.prompts import triage_prompt

TEST_CLAIMS = [
    "A pipe burst in my house, but I have no repair estimate.",
    "My car was hit in a parking lot, and the other driver left.",
    "My laptop was stolen from my office, but I have no police report.",
]


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    print("\nFIXED PARTS: role, objective, allowed categories, rules, output")
    print("VARIABLE PART: the claim inside <claim> tags")

    for index, claim in enumerate(TEST_CLAIMS, start=1):
        print("\n" + "=" * 70)
        print(f"TEST CLAIM {index}: {claim}")
        print("=" * 70)

        result = client.chat(triage_prompt(claim), temperature=0)

        print(result.text)
        print(f"\nLatency: {result.latency_seconds:.2f}s")


if __name__ == "__main__":
    main()
