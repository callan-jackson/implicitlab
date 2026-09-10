"""Single-Category IAT scoring — Karpinski & Steinman (2006).

Reference
---------
Karpinski, A., & Steinman, R. B. (2006). The Single Category Implicit
Association Test as a measure of implicit social cognition. *Journal of
Personality and Social Psychology, 91*(1), 16-32.
https://doi.org/10.1037/0022-3514.91.1.16

Provenance note, stated rather than glossed
-------------------------------------------
The primary paper is paywalled. The parameters below were taken from
consistent secondary sources — the ``implicitMeasures`` R package source,
SoSci Survey's SC-IAT implementation, and applied papers that report their
settings. Millisecond's Inquisit SC-IAT differs on several of them. This is a
field where reference implementations disagree, and the honest position is to
name the one being followed rather than to imply consensus.

Why this exists alongside the classic IAT
-----------------------------------------
The IAT is intrinsically **relative**. It cannot distinguish "Brand A is fast"
from "Brand B is slow", and the comparator is chosen by the researcher, which
means choosing it can decide the answer. That relativity is exactly right for a
comparative question — pack A versus pack B, our brand versus the category
leader — and exactly wrong for "how do people feel about us, in absolute
terms". The SC-IAT drops the second target category to answer the second kind
of question.

It is a different algorithm, not a flag on the same one:

===========================  ============================  ==========================
Step                         IAT (Greenwald 2003)          SC-IAT (Karpinski 2006)
===========================  ============================  ==========================
Lower latency cut-off        none (fast trials are signal) 350 ms, deleted
Upper latency cut-off        10,000 ms                     10,000 ms
Practice blocks              scored (B3/B6)                discarded entirely
Error penalty                block mean + 600 ms           block mean + 400 ms
Denominator                  SD of *all* retained trials   SD of *correct* trials
Number of quotients          two, averaged                 one
Response window              none                          1,500 ms, with a prompt
Participant error limit      ~30% (judgement)              20%
===========================  ============================  ==========================

The response window is the substantive difference. The SC-IAT pushes
participants to respond fast enough that deliberate control is hard, and
prompts them when they exceed it. That reshapes the latency distribution, so
reusing the IAT's cut-offs on SC-IAT data would be wrong.

The 7:7:10 stimulus ratio (target : attribute-A : attribute-B, reversed across
the two blocks) is the designed correction for response-key bias in the absence
of a contrast category. It is not cosmetic — without it, one key is correct
more often than the other and participants exploit the base rate.
"""

from __future__ import annotations

import numpy as np

from .dscore import _magnitude, _mean, _pooled_sd, _project, _summarise
from .models import (
    BlockSummary,
    DScoreResult,
    ErrorPenalty,
    Instrument,
    KARPINSKI_2006_SCIAT,
    ScoringConfig,
    Trial,
)

#: Karpinski & Steinman invalidate a participant above this error rate. It is
#: stricter than the IAT convention because the response window already pushes
#: participants toward errors, so a high rate means the window was beating them
#: rather than that one pairing was harder.
SCIAT_ERROR_LIMIT = 0.20


def compute_sciat_d(
    trials: list[Trial],
    config: ScoringConfig | None = None,
    *,
    target_label: str = "the brand",
    pole_a_label: str = "positive",
) -> DScoreResult:
    """Score a single-category IAT session.

    ``pairing`` on each trial means:

    * ``"congruent"``   — the target shared a key with attribute pole A
    * ``"incongruent"`` — the target shared a key with attribute pole B

    ``D > 0`` means faster when the target shared a key with pole A.

    Only blocks whose ``block_role`` is ``"sciat"`` are scored. The practice
    blocks that precede each critical block carry ``block_role = None`` and are
    therefore discarded here by construction, which is what the published
    procedure requires.
    """
    config = config or KARPINSKI_2006_SCIAT
    warnings: list[str] = []
    exclusions: list[str] = []

    relevant = [t for t in trials if t.block_role == "sciat" and t.pairing]
    if not relevant:
        return DScoreResult(
            d=None, d_practice=None, d_test=None, sd_practice=None, sd_test=None,
            instrument=Instrument.SC_IAT.value,
            interpretation="No scorable trials were submitted.",
            direction="none", magnitude="none",
            config=config.to_dict(), excluded=True,
            exclusion_reasons=["No SC-IAT critical-block trials present."],
        )

    n_presented = len(relevant)
    relevant = _project(relevant, config)

    def keep(t: Trial) -> bool:
        if t.timed_out:
            return False
        if t.latency_ms > config.upper_cutoff_ms:
            return False
        if config.lower_cutoff_ms is not None and t.latency_ms < config.lower_cutoff_ms:
            return False
        return True

    retained = [t for t in relevant if keep(t)]
    n_dropped = n_presented - len(retained)
    if n_dropped:
        lo = config.lower_cutoff_ms or 0
        warnings.append(
            f"{n_dropped} of {n_presented} critical-block trials removed by the "
            f"{lo:.0f}-{config.upper_cutoff_ms:.0f} ms cut-offs or by timing out. "
            f"The SC-IAT deletes fast trials outright, unlike the IAT."
        )

    timed_out = sum(1 for t in relevant if t.timed_out)
    if timed_out:
        warnings.append(
            f"{timed_out} trials exceeded the 1,500 ms response window and were "
            f"recorded as non-responses."
        )

    cells = {
        "congruent": [t for t in retained if t.pairing == "congruent"],
        "incongruent": [t for t in retained if t.pairing == "incongruent"],
    }

    for pairing, cell in cells.items():
        if len(cell) < config.min_trials_per_block:
            exclusions.append(
                f"The {pairing} block retained only {len(cell)} trials "
                f"(minimum {config.min_trials_per_block})."
            )

    if retained:
        too_fast = sum(1 for t in retained if t.latency_ms < config.fast_trial_threshold_ms)
        if too_fast / len(retained) > config.fast_trial_proportion_limit:
            exclusions.append(
                f"{too_fast / len(retained):.0%} of trials were faster than "
                f"{config.fast_trial_threshold_ms:.0f} ms."
            )
        errors = sum(1 for t in retained if not t.correct)
        if errors / len(retained) > SCIAT_ERROR_LIMIT:
            exclusions.append(
                f"Error rate was {errors / len(retained):.0%}, above the {SCIAT_ERROR_LIMIT:.0%} "
                f"limit Karpinski & Steinman apply to the SC-IAT. The response window "
                f"was beating the participant rather than one pairing being harder."
            )
    else:
        exclusions.append("No trials survived the cut-offs.")

    # --- Error handling and block means -----------------------------------
    summaries: list[BlockSummary] = []
    means: dict[str, float | None] = {}
    penalised: dict[str, list[Trial]] = {}

    for pairing, cell in cells.items():
        correct = [t.latency_ms for t in cell if t.correct]
        mean_correct = _mean(correct)

        if config.error_penalty is ErrorPenalty.BUILT_IN:
            scored = list(cell)
        elif config.error_penalty is ErrorPenalty.DELETE or mean_correct is None:
            scored = [t for t in cell if t.correct]
        else:
            bump = {
                ErrorPenalty.MEAN_PLUS_400: 400.0,
                ErrorPenalty.MEAN_PLUS_600: 600.0,
            }.get(config.error_penalty)
            if bump is None:  # MEAN_PLUS_2SD
                sd_c = float(np.std(np.asarray(correct, dtype=float), ddof=1)) if len(correct) > 1 else 0.0
                bump = 2.0 * sd_c
            scored = [t if t.correct else t.with_latency(mean_correct + bump) for t in cell]

        penalised[pairing] = scored
        means[pairing] = _mean([t.latency_ms for t in scored])
        summaries.append(_summarise(
            cell[0].block_index if cell else -1, "sciat", pairing, cell, scored, mean_correct
        ))

    summaries.sort(key=lambda s: s.pairing)

    # --- Denominator: one SD, correct responses only, both blocks pooled ---
    if config.sd_before_penalty:
        pool = cells["congruent"] + cells["incongruent"]
    else:
        pool = penalised["congruent"] + penalised["incongruent"]
    sd_source = [t.latency_ms for t in pool] if config.inclusive_sd else [
        t.latency_ms for t in pool if t.correct
    ]
    sd = _pooled_sd(sd_source, config.sd_ddof)

    d = None
    if means["congruent"] is not None and means["incongruent"] is not None and sd:
        d = (means["incongruent"] - means["congruent"]) / sd

    excluded = bool(exclusions)
    if excluded:
        d = None

    if d is None:
        interpretation = (
            "No D-score could be computed. The session did not meet the "
            "data-quality criteria."
        )
        direction = "none"
    else:
        mag = _magnitude(d)
        if mag == "little to no":
            direction = "neutral"
            interpretation = (
                f"D = {d:+.3f}. {target_label} showed little to no automatic "
                f"leaning: pairing it with '{pole_a_label}' was no easier than the "
                f"alternative."
            )
        elif d > 0:
            direction = "positive"
            interpretation = (
                f"D = {d:+.3f}. A {mag} automatic association between {target_label} "
                f"and '{pole_a_label}': responses were faster when they shared a key."
            )
        else:
            direction = "negative"
            interpretation = (
                f"D = {d:+.3f}. A {mag} automatic association running against "
                f"'{pole_a_label}' for {target_label}: responses were slower when "
                f"they shared a key."
            )

    return DScoreResult(
        d=d,
        d_practice=None,
        d_test=d,
        sd_practice=None,
        sd_test=sd,
        instrument=Instrument.SC_IAT.value,
        interpretation=interpretation,
        direction=direction,
        magnitude=_magnitude(d) if d is not None else "none",
        blocks=summaries,
        config=config.to_dict(),
        excluded=excluded,
        exclusion_reasons=exclusions,
        warnings=warnings,
        n_trials_scored=sum(len(v) for v in penalised.values()),
        n_trials_dropped=n_dropped,
    )
