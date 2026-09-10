# ImplicitLab

**A browser-based implicit association testing platform.** Millisecond
response-latency capture, the Greenwald (2003) *D*-score with its data-quality
exclusions, distribution-free resampling inference, and an LLM reporting layer
kept strictly downstream of the statistics.

**[Live demo →](https://implicitlab-callan.azurewebsites.net)** · **[Method →](https://implicitlab-callan.azurewebsites.net/method)** · **[API docs →](https://implicitlab-callan.azurewebsites.net/api/docs)**

---

## What it does

An implicit association test measures how quickly someone sorts items into
categories when two categories share a response key. If the categories are
already associated in memory, sorting is easy and fast; if they aren't, the
participant hesitates by 40–200 ms. That gap, standardised, is the *D*-score.

ImplicitLab runs the real seven-block instrument end to end:

| | |
|---|---|
| **Instruments** | Seven-block IAT (Greenwald, Nosek & Banaji 2003) and single-category SC-IAT (Karpinski & Steinman 2006) |
| **Timing** | Stimulus onset from nested `requestAnimationFrame`; response from `event.timeStamp`; measured display refresh rate; per-trial event-loop delay |
| **Scoring** | Improved *D* algorithm with the inclusive SD, forced error correction, and participant-level exclusions — plus two published alternative procedures reported side by side |
| **Inference** | Percentile bootstrap CI and a permutation test, both distribution-free, plus a split-half stability diagnostic |
| **Reporting** | Azure OpenAI writes the executive summary from computed statistics only, and every number it emits is verified against them before publication |
| **Data** | Trial-level CSV and full session JSON export; every session reproducible from a `(study, seed)` pair |

## Why the details matter

Most portfolio IATs get one of four things wrong. This one is tested against
all four:

1. **The denominator.** *D* divides by the *inclusive* SD — every retained
   trial from both pairings concatenated, errors included, one SD ignoring
   block membership. It is deliberately not Cohen's pooled-within SD, and
   `test_denominator_is_inclusive_sd_not_cohens_pooled_within` asserts the two
   differ on real data.
2. **Which mean goes in the numerator.** The mean of *correct* latencies exists
   only to build the error replacement; the numerator uses the mean over *all*
   scored trials. Conflating them shifts *D* toward zero.
3. **Blocks resolved by pairing, not position.** Block order is counterbalanced
   from the session seed, so subtracting by presentation order flips the sign
   for half the sample. `test_block_order_does_not_change_d` pins this.
4. **The 10% / 300 ms rule excludes a participant, not a trial.** Applying it as
   a filter keeps the participant in and deletes the evidence they should have
   been dropped.

And the limitation that usually goes unmentioned: the IAT's test–retest
reliability is around *r* = .50, and Cummins & Hussey (2026) report that a *D*
of exactly zero carries a 95% CI of roughly ±0.38. So this reports an interval,
never a verdict, and refuses to produce a number at all when the exclusion
criteria fire.

## The LLM boundary

> The model never sees raw trial data and is never the source of a number.

FastAPI computes *D*, the component *D*s, the SDs, the exclusions, the CI and
the reliability diagnostics deterministically. Those values — and nothing else —
are serialised into a fixed JSON block that is the model's entire input. Even
the trivially derivable judgements (does the interval exclude zero, is the
result significant) are computed in Python and handed over as a verdict, because
a wrong claim about a correct number is the one failure a numeric check cannot
catch.

What comes back is then checked: every numeral in the generated text is
extracted and matched against the payload. If any figure cannot be traced, the
draft is rejected, a deterministic rules-based summary is published instead, and
the rejected draft is shown alongside the reason. The full prompt is displayed
on every report — if a model writes part of a research report, the instructions
it was given are part of the method.

If no model is configured, the rules-based engine produces the same report from
the same numbers and the response says which engine wrote it. The product does
not break when the model is unavailable.

## Architecture

```
Browser                         FastAPI                        Azure OpenAI
───────                         ───────                        ────────────
timing.js    ── rAF onset ──▶
engine.js    ── trials ─────▶   design.py    seeded, counterbalanced
                                scoring/     D, SC-IAT, quality, resampling
                                analysis.py  orchestration
                                             │
                                             ├──▶ stats block only ──▶ gpt-5-mini
                                             │                            │
                                             │◀── prose ──────────────────┘
                                             │
                                             └──▶ llm/verify.py  every number checked
dashboard.js ◀── report ─────   storage.py   SQLite: sessions + trials + designs
```

```
app/
├── design.py          stimulus sets, block structure, counterbalancing, seeding
├── scoring/
│   ├── models.py      Trial, ScoringConfig, published parameter presets
│   ├── dscore.py      Greenwald 2003 improved algorithm
│   ├── sciat.py       Karpinski & Steinman 2006 single-category scoring
│   ├── quality.py     engagement / instrumentation / design checks
│   └── stats.py       bootstrap, permutation test, split-half, KDE
├── llm/
│   ├── prompts.py     system prompt, payload assembly, deterministic fallback
│   ├── verify.py      numeric guardrail
│   └── client.py      Azure OpenAI with a never-fails fallback path
├── analysis.py        stats first, model second, always
├── simulate.py        ex-Gaussian synthetic respondents
├── storage.py         SQLite
└── api.py / main.py   HTTP surface + static hosting

static/
├── js/timing.js       onset measurement, refresh calibration, stall watchdog
├── js/engine.js       trial runner with forced error correction
└── js/dashboard.js    Chart.js report
```

## Running it

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. The insight layer falls back to the deterministic
engine with no configuration. To enable the model path, copy `.env.example` to
`.env` and fill in your Azure OpenAI endpoint, key and deployment name.

### Tests

```bash
./.venv/bin/python -m pytest -q
```

70 tests. The ones worth reading:

- `test_dscore.py::test_d_matches_hand_computation` — reproduces a *D* worked
  out by hand, to ten decimal places.
- `test_dscore.py::test_denominator_is_inclusive_sd_not_cohens_pooled_within`
  — asserts the correct denominator differs from the common wrong one.
- `test_reproducibility.py` — runs the design generator in subprocesses under
  different `PYTHONHASHSEED` values and compares digests. This caught a real
  bug: a `set()` iteration in the generator consumed the RNG in a
  hash-order-dependent sequence, so the same seed produced different sessions
  across processes while every same-process test passed.
- `test_llm_guardrail.py::test_catches_a_fabricated_statistic` — fluent prose
  containing an invented figure must be rejected. This one also caught a real
  bug: the number-extraction regex skipped numbers at the end of a sentence,
  which is exactly where a hallucinated figure sits.
- `test_pipeline.py::test_recovers_a_planted_effect_on_average` — simulated
  respondents with a known *D* must be recovered by the scorer.

## Deployment

Azure App Service (Linux, Python 3.13), with Azure OpenAI for the insight
layer. Configuration is entirely environment-driven; no secret is in the repo.

```bash
az webapp deploy -g <rg> -n <app> --src-path implicitlab.zip --type zip
```

## Limitations

Stated on the [method page](https://implicitlab-callan.azurewebsites.net/method)
as well, because a research tool that hides them is not a research tool:

- Word stimuli only. Production pack testing uses images, which changes onset
  timing and needs preloading plus per-image onset verification.
- Desktop keyboard only. Touch changes the motor component substantially.
- Individual-level reporting, with all the reliability caveats above. Group
  aggregation, power analysis and mixed-effects modelling across a panel are the
  obvious next step and are not built.
- The comparison distribution is a convenience sample of whoever opened the
  link. It is not a norm.
- The speeded attribute-association format that commercial platforms mostly
  sell — one stimulus, many attributes, ~2 minutes — is not implemented. For
  brand-image mapping under a 15-minute session budget it is arguably a better
  fit than either instrument here.

## References

- Greenwald, Nosek & Banaji (2003). An improved scoring algorithm.
  *JPSP, 85*(2), 197–216.
- Greenwald, Brendl, Cai et al. (2022). Best research practices for using the
  IAT. *Behavior Research Methods, 54*, 1161–1180.
- Karpinski & Steinman (2006). The Single Category IAT. *JPSP, 91*(1), 16–32.
- Richetin, Costantini, Perugini & Schönbrodt (2015). *PLOS ONE, 10*(6):e0129601.
- Cummins & Hussey (2026). Individual-level confidence intervals for IAT scores.
  *Behavior Research Methods, 58*:21.
- Bridges, Pitiot, MacAskill & Peirce (2020). The timing mega-study.
  *PeerJ, 8*:e9414.

---

Built by [Callan Jackson](https://callan-jackson.github.io/). Demonstration
software — not a clinical or diagnostic instrument, and not affiliated with any
brand named in a study. The brands in the bundled studies are invented.
