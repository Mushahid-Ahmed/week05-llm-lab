"""Phase 01 - Observe token-by-token generation."""

import time

from ollama import Client

from llm_lab.config import MODEL, OLLAMA_HOST

PROMPTS = [
    "Complete this sentence with one word:\nThe capital of France is",
    "Complete this sentence with one word:\nThe capital of France is not",
    "Complete this sentence with one word:\nIn a fictional world, the capital of France is",
]


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    for index, prompt in enumerate(PROMPTS, start=1):
        print("=" * 70)
        print(f"Prompt {index}:")
        print(prompt)
        print("\nStreamed continuation: ", end="", flush=True)

        chunks = 0
        start = time.perf_counter()

        for chunk in client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0, "num_predict": 12},
        ):
            piece = chunk.message.content
            chunks += 1
            print(piece, end="", flush=True)

        elapsed = time.perf_counter() - start
        print(f"\n\nChunks received: {chunks}")
        print(f"Latency: {elapsed:.2f}s")
        print()


if __name__ == "__main__":
    main()
