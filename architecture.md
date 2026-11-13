# Week 05 LLM Engineering - Architecture

## 1. Purpose

This lab moves from a single prompt call to a validated LLM application. The
demonstrations are grouped into four phases that build on each other:

| Phase | Topic                    | Question answered                                     |
| ----- | ------------------------ | ----------------------------------------------------- |
| 01    | Generative AI foundations | What does a model actually do with a prompt?         |
| 02    | Prompt engineering        | How do prompts, examples and context change results?  |
| 03    | Loop engineering          | How does the application control repeated calls?     |
| 04    | Integration               | How do the pieces become one validated pipeline?      |

## 2. Component map

```
Input layer            argparse CLI / demo constants
      |
      v
Prompt layer           src/llm_lab/prompts.py
      |
      v
LLM client             src/llm_lab/client.py  ->  Ollama / qwen2.5:7b
      |
      v
Validation and control src/llm_lab/schemas.py, pipeline.py
      |
      v
Application output     validated ClaimAnalysis or ControlledFailure
```

### Input layer

- `phases/phase04_integration/app.py` accepts a claim argument or `--demo`.
- Empty input is rejected before any model call.
- Claim text is passed to the prompt layer as data, never as instructions.

### Prompt layer (`src/llm_lab/prompts.py`)

- Category definitions and the JSON output contract.
- `analysis_prompt()` adds repair feedback on a retry.
- `triage_prompt()` provides the sectioned format used in Phase 02.
- Zero-shot and few-shot label prompts for comparison.

### LLM client (`src/llm_lab/client.py`)

- `OllamaClient.chat()` wraps `ollama.Client`.
- Supports temperature, JSON mode and timing.
- Returns `LLMResult(text, latency_seconds)`.
- The interface is small so tests substitute a fake client.

### Validation and control (`src/llm_lab/schemas.py`, `pipeline.py`)

- `ClaimAnalysis` (Pydantic) enforces:
  - category inside the allowed list,
  - non-empty summary and rationale,
  - `missing_information` as a list of non-blank strings,
  - no extra fields.
- `analyze_claim()` performs bounded repair: invalid output is fed back
  with the validation error, at most `MAX_ATTEMPTS` times.
- Exhausting the budget raises `ControlledFailure`; the last invalid output
  is never returned as if it were valid.

### Output

- Validated JSON printed to stdout.
- Attempt count, total model latency and repair notes.
- Exit code 1 for rejected input or controlled failure.

## 3. Failure handling

| Failure                          | Handling                                      |
| -------------------------------- | --------------------------------------------- |
| Empty claim                      | Rejected before the model is called           |
| Model returns prose, not JSON    | Schema parse fails, bounded repair            |
| Category outside allowed list    | Schema validation fails, bounded repair       |
| Missing field / wrong type       | Schema validation fails, bounded repair       |
| Budget exhausted                 | `ControlledFailure`, non-zero exit            |
| Temporary service error          | Backoff demo (Phase 03) with bounded retries  |
| Schema-valid but incorrect facts | Not solved by retries; requires evaluation    |

## 4. Loop engineering

Phase 03 demonstrates the control patterns behind the pipeline:

```
State  ->  Action  ->  Observation  ->  Validation  ->  Termination
```

- `demo01_state_machine.py`: explicit state transitions, no retry.
- `demo02_validation_retry.py`: repair with feedback, fixed attempt budget.
- `demo03_critique_revise.py`: model critique as an extra stage.
- `demo04_evaluator_optimizer.py`: JSON evaluator plus an independent
  word-count check in code.
- `demo05_react_loop.py`: action selection from a fixed tool list, step budget.
- `demo06_budgeted_loop.py`: call and wall-clock budgets.
- `demo07_backoff.py`: exponential backoff with jitter for a simulated
  transient failure (no model call).
- `demo08_observability.py`: JSONL events for each attempt and termination.

## 5. Observability and privacy

- Event logs record attempt counts, latency, validation result and error
  type. They do not record credentials.
- The application uses no API key: Ollama runs locally.
- `.env` is optional and git-ignored; `.env.example` documents the keys.

## 6. From LLM application to agent

The Phase 04 pipeline follows a fixed request path: validate input, build
prompt, call the model, validate output, stop. The Phase 03 ReAct demo adds a
bounded decision loop over a fixed tool set. A full agent would extend that
loop with dynamic tool selection, persistent state and authorization checks,
which is the subject of later weeks.
