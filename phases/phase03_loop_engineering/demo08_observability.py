"""Phase 03 - Loop observability with a JSONL event log."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

MIN_WORDS = 40
MAX_ATTEMPTS = 3

CLAIM = """
A pipe burst in a house and damaged the kitchen ceiling.
Photographs are available, but a repair estimate has not
been obtained. The incident occurred yesterday.
""".strip()

LOG_FILE = Path("evidence/demo08_events.jsonl")


def log_event(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}\n")

    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")

    run_start = time.perf_counter()
    feedback = ""
    final_status = "failed"
    total_latency = 0.0
    last_output = ""
    calls = 0

    log_event({
        "event": "run_started",
        "model": MODEL,
        "max_attempts": MAX_ATTEMPTS,
    })

    for attempt in range(1, MAX_ATTEMPTS + 1):
        calls += 1

        try:
            result = client.chat(
                f"""
Write a preliminary claim summary containing at least {MIN_WORDS} words.

Claim:
{CLAIM}

Requirements:
- Use only facts stated in the claim.
- Identify missing information.
- Do not infer coverage or liability.
- Do not approve or deny the claim.

Previous validation feedback:
{feedback}

Return only the summary.
""".strip(),
                temperature=0,
                max_tokens=140,
            )
        except Exception as exc:
            # Log the error type, not potentially sensitive details.
            log_event({
                "event": "model_call_error",
                "attempt": attempt,
                "error_type": type(exc).__name__,
            })

            print("Model call failed:", type(exc).__name__)
            final_status = "error"
            break

        total_latency += result.latency_seconds
        last_output = result.text

        word_count = len(result.text.split())
        valid = word_count >= MIN_WORDS

        log_event({
            "event": "model_call_completed",
            "attempt": attempt,
            "latency_seconds": round(result.latency_seconds, 4),
            "output_word_count": word_count,
            "validation_passed": valid,
        })

        print(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")
        print("Word count:", word_count)
        print("Latency:", f"{result.latency_seconds:.3f}s")
        print("Validation passed:", valid)

        if valid:
            final_status = "success"
            break

        feedback = (
            f"The output contained {word_count} words. "
            f"At least {MIN_WORDS} words are required. "
            "Revise without inventing facts."
        )

        log_event({
            "event": "validation_failed",
            "attempt": attempt,
            "reason": feedback,
        })

    elapsed = time.perf_counter() - run_start

    report = {
        "event": "run_finished",
        "status": final_status,
        "model": MODEL,
        "calls": calls,
        "total_model_latency_seconds": round(total_latency, 4),
        "total_elapsed_seconds": round(elapsed, 4),
        "final_word_count": len(last_output.split()),
    }

    log_event(report)

    print("\n--- FINAL REPORT ---")
    print(json.dumps(report, indent=2))
    print(f"\nEvent log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
