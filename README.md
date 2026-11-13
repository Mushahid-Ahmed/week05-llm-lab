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

`evidence/` contains the captured output of a full run on this machine.

| File                              | Contents                                       |
| --------------------------------- | ---------------------------------------------- |
| `phase01_demo01..05.txt`          | Foundations demos with real model responses    |
| `phase02_demo00..06.txt`          | Prompt engineering demos                       |
| `phase03_demo01..08.txt`          | Loop engineering demos                         |
| `phase04_demo_app.txt`            | Integrated CLI on three claims                 |
| `pytest.txt` / `pytest.png`       | Test run, 12 passed                            |
| `rules_vs_llm.png`                | Rendered Phase 01 rules-versus-LLM log         |
| `prompt_evaluation.png`           | Rendered baseline-versus-enhanced comparison   |
| `validation_retry.png`            | Rendered retry exhaustion log                  |
| `integrated_app.png`              | Rendered Phase 04 CLI output                   |
| `prompt_evaluation.jsonl`         | Per-claim baseline/enhanced rows               |
| `context_results.csv`             | 30 runs for the context experiment             |
| `chaining_results.txt`            | Single versus chained output                   |
| `demo08_events.jsonl`             | Observability event log                        |

The model was run locally with `temperature=0` unless a demo explicitly
changes it. Timings are machine-specific and are reported for comparison
within a run, not as absolute performance claims.

## Measured results from this run

Hardware was CPU-only, so latencies are long: expect roughly 2-5 tokens per
second and 5-50 s per call depending on output length.

| Experiment                        | Observation                                                     |
| --------------------------------- | --------------------------------------------------------------- |
| Rules vs LLM (Phase 01)           | Rules returned `unknown` on all three paraphrases; the LLM labelled all three plausibly |
| Concise vs noisy context          | Both prompts classified every claim correctly; the noisy prompt cost more tokens without adding accuracy |
| Temperature 0 / 0.7 / 1.0         | All six runs obeyed the three-bullet requirement; higher temperature changed wording and emphasis, not structure or factual content |
| Baseline vs enhanced prompt       | Baseline 0% valid output; enhanced 100% valid, 80% category accuracy, 60% missing-info recall, and lower mean latency (25.3 s vs 41.7 s) because the baseline rambled until it hit the token cap |
| Zero-shot vs few-shot labels      | Both 3/5 against the reference labels; the two ambiguous claims (laptop at home, medical treatment after an accident) were labelled confidently by both |
| Context relevance (30 runs)       | 100% label accuracy and consistency in both contexts; the extra paragraphs changed nothing except token count |
| Validation-retry loop             | All three attempts produced 38-39 words against a 40-word minimum; the loop exited with a controlled failure rather than returning invalid text |
| Evaluator-optimizer loop          | The model's self-reported `word_count: 40` was false (draft was ~21 words); the independent mechanical check correctly rejected the argument |
| ReAct tool loop                   | Ended after 3 steps with `missing_information` still empty; the loop control, not the model, is the limit here |
| Phase 04 CLI                      | All three claims accepted on the first attempt with schema-valid JSON |

Several logs are failures on purpose. The retry loop, the self-evaluating
model and the early-finishing ReAct loop show where loop engineering has to
carry guarantees that prompting cannot provide.

## Note on evaluation

A five-claim sample is enough to demonstrate a pipeline, not enough to prove
one prompt is better than another. Results should be read as observations
from this run; larger and unseen test sets are required before drawing
general conclusions.
