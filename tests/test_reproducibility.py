"""Cross-process reproducibility.

A seeded design is only reproducible if nothing in the generation path consumes
randomness in an order that varies between runs. The obvious trap is iterating a
``set`` of strings: Python randomises string hashing per process by default, so
the loop order changes on every restart. Every same-process test still passes,
which is what makes it dangerous — the bug only appears when someone tries to
reproduce a session tomorrow.

These tests run the generator in a *subprocess* with an explicit
``PYTHONHASHSEED`` and compare digests, which is the only way to catch it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

SNIPPET = """
import hashlib, json, sys
sys.path.insert(0, {root!r})
from app.design import STUDIES, build_session
s = build_session(STUDIES[{study!r}], preset={preset!r}, seed=12345)
print(json.dumps({{
    "digest": hashlib.sha256(json.dumps(s["trials"], sort_keys=True).encode()).hexdigest(),
    "counterbalance": s["counterbalance"],
    "n": s["n_trials"],
}}))
"""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate(hashseed: str, study: str = "gin-premium", preset: str = "standard") -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(root=ROOT, study=study, preset=preset)],
        capture_output=True, text=True, env=env, check=True, cwd=ROOT,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_design_is_identical_across_process_hash_seeds():
    results = [_generate(str(h)) for h in (0, 1, 7, 12345)]
    digests = {r["digest"] for r in results}
    assert len(digests) == 1, (
        "The same seed produced different trial sequences in different processes. "
        "Something in the generation path is iterating an unordered collection "
        "while consuming the RNG."
    )
    assert len({json.dumps(r["counterbalance"], sort_keys=True) for r in results}) == 1


def test_every_study_and_preset_is_hash_stable():
    for study in ("gin-premium", "aurelia-sciat"):
        for preset in ("express", "full"):
            a = _generate("1", study, preset)
            b = _generate("999", study, preset)
            assert a["digest"] == b["digest"], f"{study}/{preset} is not reproducible"
            assert a["n"] == b["n"]
