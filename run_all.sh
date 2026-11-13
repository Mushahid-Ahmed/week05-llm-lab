#!/usr/bin/env bash
# Run every Week 05 demonstration and save the output under evidence/.
# Usage (from the project root, virtualenv active):
#   bash run_all.sh

set -u

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
mkdir -p evidence

run() {
    local name="$1"
    shift
    echo "==> $name"
    "$@" 2>&1 | tee "evidence/${name}.txt"
}

run phase01_demo01_rules_vs_llm \
    python -m phases.phase01_foundations.demo01_rules_vs_llm
run phase01_demo02_token_continuation \
    python -m phases.phase01_foundations.demo02_token_continuation
run phase01_demo03_context_noise \
    python -m phases.phase01_foundations.demo03_context_noise
run phase01_demo04_temperature \
    python -m phases.phase01_foundations.demo04_temperature
run phase01_demo05_plausibility_audit \
    python -m phases.phase01_foundations.demo05_plausibility_audit

run phase02_demo00_sentiment_zero_shot \
    python -m phases.phase02_prompt_engineering.demo00_sentiment_zero_shot
run phase02_demo01_vague_vs_structured \
    python -m phases.phase02_prompt_engineering.demo01_vague_vs_structured
run phase02_demo02_prompt_components \
    python -m phases.phase02_prompt_engineering.demo02_prompt_components
run phase02_demo03_prompt_evaluation \
    python -m phases.phase02_prompt_engineering.demo03_prompt_evaluation
run phase02_demo04_zero_shot_vs_few_shot \
    python -m phases.phase02_prompt_engineering.demo04_zero_shot_vs_few_shot
run phase02_demo05_context_relevance \
    python -m phases.phase02_prompt_engineering.demo05_context_relevance
run phase02_demo06_chaining \
    python -m phases.phase02_prompt_engineering.demo06_chaining

run phase03_demo01_state_machine \
    python -m phases.phase03_loop_engineering.demo01_state_machine
run phase03_demo02_validation_retry \
    python -m phases.phase03_loop_engineering.demo02_validation_retry
run phase03_demo03_critique_revise \
    python -m phases.phase03_loop_engineering.demo03_critique_revise
run phase03_demo04_evaluator_optimizer \
    python -m phases.phase03_loop_engineering.demo04_evaluator_optimizer
run phase03_demo05_react_loop \
    python -m phases.phase03_loop_engineering.demo05_react_loop
run phase03_demo06_budgeted_loop \
    python -m phases.phase03_loop_engineering.demo06_budgeted_loop
run phase03_demo07_backoff \
    python -m phases.phase03_loop_engineering.demo07_backoff
run phase03_demo08_observability \
    python -m phases.phase03_loop_engineering.demo08_observability

run phase04_demo_app \
    python -m phases.phase04_integration.app --demo

run pytest \
    python -m pytest -v

echo "All demonstrations finished. Outputs are in evidence/."
