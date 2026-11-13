"""Phase 03 - Plan-execute-observe loop (ReAct-style action selection)."""

import json

from llm_lab.client import OllamaClient
from llm_lab.config import MODEL

MAX_STEPS = 4

CLAIM = """
A pipe burst in my house. The kitchen ceiling is damaged.
I have photographs, but no repair estimate.
""".strip()

TOOLS = {
    "extract_facts": "Extract explicitly stated facts from the claim.",
    "identify_missing_information": "List information needed for further assessment.",
    "classify_claim": "Choose a claim category using the available facts.",
    "finish": "Finish the workflow.",
}


def call_model(client: OllamaClient, prompt: str) -> str:
    return client.chat(prompt, temperature=0).text


def choose_action(client: OllamaClient, state: dict) -> dict:
    raw = client.chat(
        f"""
You are controlling a small claim-analysis workflow.

Available actions:
{json.dumps(TOOLS, indent=2)}

Current state:
{json.dumps(state, indent=2)}

Choose exactly one action from the available actions.

Return JSON:
{{
  "action": "one allowed action",
  "reason": "short explanation"
}}

Do not include code or other actions.
""".strip(),
        temperature=0,
        json_mode=True,
        max_tokens=64,
    ).text

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid action JSON: {raw[:200]}") from exc

    if decision.get("action") not in TOOLS:
        raise ValueError("Model selected an unknown action.")

    return decision


def execute_action(client: OllamaClient, action: str, state: dict) -> dict:
    if action == "extract_facts":
        state["facts"] = call_model(
            client,
            f"""
Extract only explicitly stated facts from this claim.
Do not infer missing details.

Claim:
{CLAIM}
""".strip(),
        )

    elif action == "identify_missing_information":
        state["missing_information"] = call_model(
            client,
            f"""
Identify missing information from the claim and extracted facts.

Claim:
{CLAIM}

Extracted facts:
{state.get("facts", "Not available")}

Return only a list of missing information.
""".strip(),
        )

    elif action == "classify_claim":
        state["classification"] = call_model(
            client,
            f"""
Classify this claim as vehicle, property, travel, health,
other, or context-dependent.

Claim:
{CLAIM}

Extracted facts:
{state.get("facts", "Not available")}

Missing information:
{state.get("missing_information", "Not available")}

Return one category and a short explanation.
Do not decide coverage or claim approval.
""".strip(),
        )

    elif action == "finish":
        state["finished"] = True

    return state


def main() -> None:
    client = OllamaClient()
    print(f"Model: {MODEL}")

    state = {
        "facts": "",
        "missing_information": "",
        "classification": "",
        "finished": False,
    }

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- STEP {step} ---")

        decision = choose_action(client, state)
        action = decision["action"]

        print("Chosen action:", action)
        print("Reason:", decision.get("reason", ""))

        if action == "finish":
            state["finished"] = True
            break

        state = execute_action(client, action, state)
        print("Updated state:", json.dumps(state, indent=2))

        completed = {
            "extract_facts": bool(state["facts"]),
            "identify_missing_information": bool(state["missing_information"]),
            "classify_claim": bool(state["classification"]),
        }

        if all(completed.values()):
            state["finished"] = True
            break

    print("\nFINAL STATE")
    print(json.dumps(state, indent=2))

    if not state["finished"]:
        print("Workflow stopped because the step budget was exhausted.")


if __name__ == "__main__":
    main()
