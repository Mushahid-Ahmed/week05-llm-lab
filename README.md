# Week 05 - LLM Engineering Lab

A runnable lab on generative AI, prompt engineering, loop engineering and
integration, using a **local** model through Ollama. Every demonstration in
the Week 05 slides is implemented as a small script with measured output.

No API key is used. Nothing leaves the machine.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- Model: `qwen2.5:7b` (`ollama pull qwen2.5:7b`)

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama list

pytest -v
bash run_all.sh                   # runs all demos, writes evidence/*.txt
```

Environment overrides (optional, see `.env.example`):

| Variable       | Default                  |
| -------------- | ------------------------ |
| `OLLAMA_HOST`  | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen2.5:7b`             |
| `MAX_ATTEMPTS` | `3`                      |

Scripts are modules, so run them from the project root with `src` on the path:

```bash
PYTHONPATH=src python -m phases.phase01_foundations.demo01_rules_vs_llm
```

## Project structure

```
Week05/
├── src/
│   └── llm_lab/
│       ├── config.py              # environment settings
│       ├── client.py              # Ollama adapter + latency
│       ├── prompts.py             # prompt templates and label normalizer
│       ├── schemas.py             # Pydantic output contract
│       └── pipeline.py            # bounded repair loop
├── phases/
│   ├── phase01_foundations/       # 5 demos
│   ├── phase02_prompt_engineering/ # 7 demos
│   ├── phase03_loop_engineering/  # 8 demos
│   └── phase04_integration/       # validated CLI application
├── tests/                         # schema and pipeline tests (no model needed)
├── data/                          # claims dataset and sentiment samples
├── evidence/                      # captured output, screenshots, CSV/JSONL
├── run_all.sh
├── requirements.txt
├── pyproject.toml
├── architecture.md
└── README.md
```

## Phase 01 - Generative AI foundations

| Script                            | Demonstrates                                   |
| --------------------------------- | ---------------------------------------------- |
| `demo01_rules_vs_llm.py`          | Keyword rules versus an LLM on the same claims |
| `demo02_token_continuation.py`    | Streamed token-by-token completion             |
| `demo03_context_noise.py`         | Concise versus noisy context, 3 repeats        |
| `demo04_temperature.py`           | Temperature 0 / 0.7 / 1.0, 2 runs each         |
| `demo05_plausibility_audit.py`    | Auditing a fluent but unsupported draft        |

## Phase 02 - Prompt engineering

| Script                            | Demonstrates                                      |
| --------------------------------- | ------------------------------------------------- |
| `demo00_sentiment_zero_shot.py`   | Zero-shot sentiment labels on `data/sentiment.txt`|
| `demo01_vague_vs_structured.py`   | "Analyze this claim" versus a full template       |
| `demo02_prompt_components.py`     | Fixed template, variable claim, 3 claims          |
| `demo03_prompt_evaluation.py`     | Baseline versus enhanced prompt, 5 claims         |
| `demo04_zero_shot_vs_few_shot.py` | Label accuracy with and without examples          |
| `demo05_context_relevance.py`     | Relevant versus irrelevant context, 3 repeats     |
| `demo06_chaining.py`              | One prompt versus a two-stage chain               |

`demo03_prompt_evaluation.py` measures valid-output rate, category accuracy,
missing-information recall and mean latency, and writes
`evidence/prompt_evaluation.jsonl`.

## Phase 03 - Loop engineering

| Script                            | Loop pattern                                  |
| --------------------------------- | --------------------------------------------- |
| `demo01_state_machine.py`         | Explicit states, no retry                     |
| `demo02_validation_retry.py`      | Repair with feedback, bounded attempts        |
| `demo03_critique_revise.py`       | Generate, critique, revise                    |
| `demo04_evaluator_optimizer.py`   | JSON evaluator plus a mechanical word check   |
| `demo05_react_loop.py`            | Action selection over a fixed tool list       |
| `demo06_budgeted_loop.py`         | Call and wall-clock budgets                   |
| `demo07_backoff.py`               | Exponential backoff with jitter (simulated)   |
| `demo08_observability.py`         | JSONL events for every attempt and outcome    |

## Phase 04 - Integration and evaluation

`phases/phase04_integration/app.py` combines every layer:

```
input validation -> prompt -> Ollama -> Pydantic schema -> bounded repair -> JSON
```

```bash
PYTHONPATH=src python -m phases.phase04_integration.app \
  "A storm damaged the roof of my house yesterday."

PYTHONPATH=src python -m phases.phase04_integration.app --demo
```

Valid output is printed as JSON with attempt count and latency. Invalid output
is rejected, fed back to the model, and retried up to the attempt budget. If
the budget runs out, the application exits with a controlled failure instead
of returning unvalidated text.

## Tests

```bash
pytest -v
```

The 12 tests cover:

- schema acceptance and rejection (category, fields, types, blank strings),
- valid output accepted on the first call,
- invalid output repaired on retry,
- retry budget exhaustion,
- empty input rejected before any model call,
- latency aggregation.

The tests use a fake client, so they do not need Ollama.

## Evidence

`evidence/` contains one text log per demonstration plus:

- `prompt_evaluation.jsonl` - per-claim baseline/enhanced results
- `context_results.csv` - 30 runs for the context experiment
- `chaining_results.txt` - single versus chained output
- `demo08_events.jsonl` - observability event log
- screenshots of the pytest run and key demo output

The model was run locally with `temperature=0` unless a demo explicitly
changes it. Timings are machine-specific and are reported for comparison
within a run, not as absolute performance claims.

## Note on evaluation

A five-claim sample is enough to demonstrate a pipeline, not enough to prove
one prompt is better than another. Results should be read as observations
from this run; larger and unseen test sets are required before drawing
general conclusions.
