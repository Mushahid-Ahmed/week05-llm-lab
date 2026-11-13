"""Phase 02 - Zero-shot sentiment classification of the supplied samples."""

from pathlib import Path

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

TEXT_FILE = Path("data/sentiment.txt")
ALLOWED = {"positive", "negative", "neutral"}


def normalize(text: str) -> str:
    lowered = text.strip().lower()

    for label in ALLOWED:
        if label in lowered:
            return label

    return "invalid-output"


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    sentences = [
        line.strip()
        for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for sentence in sentences:
        result = client.chat(
            f"""
Classify the sentiment of the text as exactly one of:
positive, negative, neutral.

Return only the label.

Text: {sentence}
Label:
""".strip(),
            temperature=0,
            max_tokens=16,
        )

        print(f"{sentence}")
        print(f"  -> {normalize(result.text)} "
              f"({result.latency_seconds:.2f}s)")

    print(
        "\nObservation: mixed sentiment (for example, a working product\n"
        "delivered late) exposes how category boundaries are defined.\n"
        "The model may also add explanation despite the instruction."
    )


if __name__ == "__main__":
    main()
