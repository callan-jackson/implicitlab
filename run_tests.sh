#!/usr/bin/env bash
# Both suites. The Python tests cover design, scoring, quality and the LLM
# guardrail; the Node suite covers the browser trial engine, which is the one
# component Python cannot reach.
set -e
PY=${PY:-./.venv/bin/python}
echo "── Python ──────────────────────────────────────────"
"$PY" -m pytest -q
echo
echo "── Trial engine (Node) ─────────────────────────────"
node --test tests/engine.test.mjs
