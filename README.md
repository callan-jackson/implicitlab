# ImplicitLab

**A browser-based implicit association testing platform, from single trial to
client deck.** Millisecond response-latency capture, the Greenwald (2003)
*D*-score with its data-quality exclusions, distribution-free resampling
inference, and an LLM reporting layer kept strictly downstream of the
statistics — plus a **cohort Studio** that ingests a raw panel dump, screens
participants, segments the audience, tests every result two ways with
false-discovery-rate control, and exports a formatted Excel workbook and a
PowerPoint deck.

**[Live demo →](https://implicitlab.onrender.com)** · **[Studio →](https://implicitlab.onrender.com/studio)** · **[Method →](https://implicitlab.onrender.com/method)** · **[API docs →](https://implicitlab.onrender.com/api/docs)**

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
| **Cohorts** | Batch CSV/JSON ingestion, participant-level screening with a live exclusion funnel, segmentation, parametric *and* permutation tests with Benjamini–Hochberg correction |
| **Deliverables** | Five-tab Excel workbook (pandas + openpyxl) and a nine-slide PowerPoint deck (python-pptx), built from the same function as the dashboard |

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

## Cohort analysis — the Studio

A single D-score is one person on one occasion, and the instrument is not built
to be read that way. A commercial study recruits a panel, runs everyone through
a battery of brand-attribute tests, and asks which associations hold across the
audience and between segments of it. `/studio` is that workflow:

```
raw panel dump ─▶ ingest ─▶ screen participants ─▶ score every task ─▶ group inference ─▶ dashboard
  (CSV / JSON)     report      exclusion funnel      vectorised D       bootstrap CIs       workbook
                   of every    (live counts)         (pinned to the     t on log-RT +       deck
                   rejected                           reference)         permutation, FDR
                   row
```

**Ingestion.** One row per trial, participant variables repeated on each row —
the shape Gorilla, Inquisit and most panel platforms export — or JSON with a
separate participants table. Column aliases are mapped (`rt`, `RT_ms`,
`Respondent ID`…), booleans in any common spelling are parsed, and every
rejected row is counted with its reason. Any non-trial column that is constant
within participants becomes a segmentation variable; continuous ones (age in
years) are cut into tertiles; columns that vary within participants are
recognised as trial-level and left alone.

**Screening is per participant, and ordered.** Incomplete data → latency screen
(> 10% of trials under 300 ms) → accuracy screen (< 75%) → trials remaining
after trimming. The funnel shows how many fell at each stage, live, as the
thresholds move. The latency screen runs *before* any lower trim, so dragging
the trim slider cannot hide a fast responder by deleting the evidence against
them. Participants are dropped whole, so every attribute is analysed on the
same people.

**Scoring 1,500 IATs in 6 ms.** Re-scoring a panel with the reference
implementation takes seconds — too slow to re-run on every slider move — so
`app/cohort/scoring.py` computes D for every (participant, attribute) task at
once with `np.bincount` group-bys. Two implementations of one algorithm are a
liability unless something pins them together:
`test_vectorised_d_matches_reference_scorer` scores every task both ways and
requires agreement to 1e-9 (it is 1.6 × 10⁻¹⁵ in practice).

**Two families of test, always both.** *Parametric*: t-tests on each
participant's difference in mean **log** RT — raw latencies are right-skewed,
their logs are nearly symmetric — with Welch's test between segments.
*Non-parametric*: sign-flip and label-shuffle permutation tests on D itself.
The dashboard switches between them instantly and counts the significance calls
that change. CIs are percentile bootstraps **over participants**, the unit of
sampling at group level.

**Multiple comparisons are corrected.** An attribute × segment grid is a lot of
tests, and an uncorrected grid will put a false positive in a client deck.
Every p-value carries a Benjamini–Hochberg q-value within its family, and the
significance flags are set on q. Cells under 30 respondents are marked low
base, as agencies do.

**Deliverables come from the same function as the screen.** The Excel workbook
(Executive Summary · Segment Breakdown · Participant D-scores · Raw Trial Log ·
Method & Parameters) and the PowerPoint deck (takeaways, sample funnel,
heatmap, forest plot, the segment story, why the stats are non-parametric,
robustness, method appendix) both call `analyse_cohort` with the dashboard's
exact parameters, and write those parameters into the file. A deliverable that
cannot say which exclusion rules produced it cannot be defended.

**Takeaways are written by rules, not a model.** The single-session report
uses an LLM with a numeric verifier; a client deck goes out under the agency's
name, its claims come in a handful of fixed shapes, and those are what a
template does reliably. The rules: a result that fails q < .05 is never given a
direction; D is described as relative ("more associated with Premium than
Northvane"), never absolute; a significant-but-negligible effect is called
small.

**The demonstration panel has known answers.** 520 synthetic respondents,
three attributes (Premium, Eco-friendly, Trustworthy), four segment variables,
and deliberately planted effects: loyalty moves Premium, the campaign cell
moves Trust, 18–34s move Eco — and Device moves nothing. It also contains fast
responders, guessers, dropouts and people who walked away mid-trial at
online-panel rates. The test suite asserts that every planted respondent is
screened at the right stage, every planted effect is found under both methods,
and the null variable stays null. **Download sample CSV** in the Studio gives
you a raw dump in the upload format, so the ingestion path can be demonstrated
on a real file.

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
├── cohort/            panel analysis — see "Cohort analysis" above
│   ├── ingest.py      raw dump → validated trial table + ingest report
│   ├── scoring.py     participant screening + vectorised D
│   ├── stats.py       bootstrap, t on log-RT, permutation, Benjamini–Hochberg
│   ├── analysis.py    (batch, parameters) → the complete cohort report
│   ├── takeaways.py   rule-written executive bullets
│   ├── charts.py      matplotlib figures for the deliverables
│   ├── export_xlsx.py five-tab workbook
│   ├── export_pptx.py nine-slide deck
│   ├── simulate.py    synthetic panel with planted effects
│   └── registry.py    batch cache + persistence
├── analysis.py        stats first, model second, always
├── simulate.py        ex-Gaussian synthetic respondents
├── storage.py         SQLite
└── api.py / main.py   HTTP surface + static hosting

static/
├── js/timing.js       onset measurement, refresh calibration, stall watchdog
├── js/engine.js       trial runner with forced error correction
├── js/dashboard.js    Chart.js report
├── js/studio.js       cohort Studio: live parameters, funnel, heatmap, forest
└── assets/            brand lockup, favicons, social card — all generated

tools/
└── make_logo_assets.py   derives every brand asset from one source lockup
```

### Brand assets

`static/assets/logo-source.png` is the single source of truth. Everything else
in that directory is generated:

```bash
./.venv/bin/python tools/make_logo_assets.py static/assets/logo-source.png
```

The one non-obvious step is the dark-background variant. The wordmark is drawn
in two materials — "Implicit" in near-black neutral ink and "Lab" in a
blue-to-purple gradient — and on a dark UI the neutral half disappears. Splitting
them by lightness would fail, because the deep end of the purple gradient is as
dark as the ink; they are separated by **chroma** instead, which is near zero for
neutral ink and large everywhere along the gradient, including through the
antialiased edges. That lifts the ink without leaving a pale halo around the
gradient glyphs.

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
./run_tests.sh
```

115 tests — Python covering design, scoring, quality, the LLM
guardrail, the cohort pipeline and the exports, plus 10 in Node covering the
browser trial engine, which is the one component Python cannot reach. The ones
worth reading:

- `test_cohort.py::test_vectorised_d_matches_reference_scorer` — the fast
  cohort scorer against the reference implementation, every task in a panel,
  with and without trimming, to 1e-9.
- `test_cohort.py::test_planted_segment_effects_are_recovered` and
  `test_null_variable_stays_null` — both statistical methods find the effects
  built into the simulated panel, and neither finds one in the variable that
  has none.
- `test_cohort.py::test_lower_trim_cannot_rescue_a_fast_responder` — the
  latency screen is evaluated before trimming.

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
- `engine.test.mjs` — the trial engine against a small hand-written DOM stub.
  This caught a third real bug: aborting mid-trial left the engine awaiting a
  keypress that would never arrive, so its global `keydown` listener was never
  removed and a second session in the same page would have registered every key
  twice.

## Deployment

A Docker image (`Dockerfile`) on Render's free tier, described by
`render.yaml`. Deployment is continuous: a push to `main` runs the test
workflow, and only when it passes does `deploy.yml` call Render's deploy hook
for that exact commit. The
520-person demonstration panel is baked into the image at build time, and the
default Studio views are computed in the background at boot, so a cold start
does not mean a slow first page. Configuration is environment-driven and no
secret is in the repository.

```bash
docker build -t implicitlab . && docker run -p 7860:7860 implicitlab
```

The free tier sleeps after fifteen minutes idle (the first request then takes
under a minute) and runs on a fraction of a CPU, so the Studio's live sliders
are noticeably quicker on a laptop, where a full re-analysis of the panel takes
about 150 ms. Analyses are memoised, so any view visited once is instant. The
filesystem is ephemeral: uploaded batches and collected sessions last until the
instance restarts. A production deployment would put batches in object storage
behind authentication; storage sits behind one module (`storage.py`) for that
reason.

The app also runs unchanged on Azure App Service (`startup.sh`), where it was
first deployed, with Azure OpenAI behind the insight layer. Without model
credentials the deterministic report engine is used, and the report says so.

## Limitations

Stated on the [method page](https://implicitlab.onrender.com/method)
as well, because a research tool that hides them is not a research tool:

- Word stimuli only. Production pack testing uses images, which changes onset
  timing and needs preloading plus per-image onset verification.
- Desktop keyboard only. Touch changes the motor component substantially.
- Cohort inference is on per-participant summaries (D, mean log-RT
  difference). That is the standard applied analysis and it is what the
  deliverables report, but a trial-level mixed-effects model (random
  intercepts for participants and stimuli) would use the data more fully and
  generalise over stimuli as well as people. Power analysis for planning
  sample sizes is not built.
- The cohort scorer implements the built-in error penalty only (forced
  correction, time-to-correct scored). Uploads from tasks that did not force
  correction are flagged at ingestion rather than silently re-scored.
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
