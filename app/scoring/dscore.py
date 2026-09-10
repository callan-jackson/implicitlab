"""The IAT *D* measure — Greenwald, Nosek & Banaji (2003) improved algorithm.

References
----------
Greenwald, A. G., Nosek, B. A., & Banaji, M. R. (2003). Understanding and using
the Implicit Association Test: I. An improved scoring algorithm. *Journal of
Personality and Social Psychology, 85*(2), 197-216.
https://doi.org/10.1037/0022-3514.85.2.197

Greenwald, A. G., Brendl, M., Cai, H., et al. (2022). Best research practices
for using the Implicit Association Test. *Behavior Research Methods, 54*,
1161-1180. https://doi.org/10.3758/s13428-021-01624-3

Richetin, J., Costantini, G., Perugini, M., & Schönbrodt, F. (2015). Should we
stop looking for a better scoring algorithm for handling Implicit Association
Test data? Test of the role of errors, extreme latencies treatment, scoring
formula, and practice trials on reliability and validity.
*PLOS ONE, 10*(6): e0129601. https://doi.org/10.1371/journal.pone.0129601

The recommended procedure (Greenwald et al., 2003, Table 4):

    1. Use data from B3, B4, B6 and B7.
    2. Eliminate trials with latencies > 10,000 ms.
    3. Eliminate subjects for whom more than 10% of trials have latency
       less than 300 ms.
    4. Compute the mean of correct latencies for each block.
    5. Compute one pooled SD for all trials in B3 and B6, and another for all
       trials in B4 and B7.
    6. Replace each error latency with the block mean (from Step 4) + 600 ms.
    7. Average the resulting values for each of the four blocks.
    8. Compute two differences: B6 - B3 and B7 - B4.
    9. Divide each difference by its associated pooled-trials SD from Step 5.
    10. Average the two quotients.

Four details in that sequence are load-bearing, and they are the ones most
often got wrong in re-implementations:

*   **The denominator is the inclusive SD.** Concatenate every retained trial
    from *both* pairings of a block pair and take one standard deviation,
    ignoring which block each trial came from. It is deliberately **not**
    Cohen's pooled-within-condition SD, **not** the SD of difference scores,
    **not** the mean of the two blocks' SDs, and **not** one block's SD.
    Because the two blocks have different means, the between-block variance is
    part of the denominator on purpose — that is what makes D an
    individual-level effect size rather than a raw latency difference. The
    paper's own Table 4 says "pooled", which is exactly what trips people up;
    the body text is the unambiguous version: the SD is computed "ignoring the
    condition membership of each score".

*   **Two different means are in play.** The mean of *correct* latencies
    (Step 4) exists only to build the error replacement. The mean that goes
    into the numerator (Step 7) is taken over *all* scored trials in the cell,
    errors included at their replaced values. Conflating them is a silent bug
    that shifts D toward zero.

*   **Blocks are identified by pairing, not position.** Block order is
    counterbalanced, so "B6 − B3" means "incongruent practice − congruent
    practice". Subtracting by presentation order flips the sign for half the
    sample, which does not add noise — it adds bias.

*   **The 10% / 300 ms rule excludes a participant, not a trial.** It is a
    judgement that someone was not doing the task. Applying it as a trial
    filter quietly keeps the participant in and removes exactly the evidence
    that they should have been dropped.

A documented ambiguity, stated rather than hidden
-------------------------------------------------
Greenwald et al. (2003) is internally inconsistent on the fast-responder rule:
p. 208 says "10% or more", p. 212 says "more than 10%". Implementations also
differ on the denominator — ``iatgen::cleanIAT`` uses the raw four-block trial
count *before* the 10,000 ms drop, while the 2022 best-practice paper says
"more than 10% of *remaining* trials". This module uses a strict ``>`` against
the post-cutoff trial count, and says so here rather than pretending the
literature is unanimous.

Sign convention
---------------
``D > 0`` means the participant was *faster* in the congruent pairing
(target A + attribute pole A) than in the incongruent pairing.
"""

from __future__ import annotations

import numpy as np

from .models import (
    BlockSummary,
    DScoreResult,
    ErrorPenalty,
    Instrument,
    LatencySource,
    ScoringConfig,
    Trial,
)

# Conventional labels. These are the reporting bands **Project Implicit** uses
# when it tells a participant what their score means. They are loosely
# motivated by effect-size conventions but they are NOT Cohen's d cut-offs:
# Cohen's are .20/.50/.80, and the rough d-equivalents of these IAT bands are
# nearer .3/.7/1.3. Quoting them as "small/medium/large in Cohen's sense" is a
# common and wrong shorthand.
SLIGHT = 0.15
MODERATE = 0.35
STRONG = 0.65

THRESHOLD_NOTE = (
    "Bands of 0.15 / 0.35 / 0.65 are the Project Implicit reporting convention, "
    "not Cohen's d cut-offs (which are 0.20 / 0.50 / 0.80). They describe the "
    "size of a latency difference, not its reliability."
)


def _magnitude(d: float) -> str:
    a = abs(d)
    if a < SLIGHT:
        return "little to no"
    if a < MODERATE:
        return "slight"
    if a < STRONG:
        return "moderate"
    return "strong"


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _pooled_sd(latencies: list[float], ddof: int) -> float | None:
    """SD of the concatenated trials from both pairings of a block pair."""
    if len(latencies) <= ddof:
        return None
    sd = float(np.std(np.asarray(latencies, dtype=float), ddof=ddof))
    return sd if sd > 0 else None


def _scored_latency(t: Trial, config: ScoringConfig) -> float | None:
    """Pick the clock this configuration scores.

    With the built-in penalty the trial is scored on time-to-correct-response,
    which for a correct trial is identical to the first-response latency. A
    trial with no recorded correction (the participant abandoned it) has no
    time-to-correct and is unusable under that configuration.
    """
    if config.latency_source is LatencySource.TO_CORRECT:
        if t.correct:
            return t.latency_ms
        return t.latency_to_correct_ms
    return t.latency_ms


def _project(trials: list[Trial], config: ScoringConfig) -> list[Trial]:
    """Rewrite each trial's ``latency_ms`` to the clock being scored."""
    out: list[Trial] = []
    for t in trials:
        lat = _scored_latency(t, config)
        if lat is None:
            continue
        out.append(t if lat == t.latency_ms else t.with_latency(lat))
    return out


def _apply_penalty(
    trials: list[Trial], config: ScoringConfig
) -> tuple[list[Trial], float | None]:
    """Substitute error latencies within a single block (Steps 4 and 6).

    Returns the block's trials with error latencies replaced, plus the mean of
    the block's *correct* latencies, which is reported and is also what the
    substitution is built from.
    """
    penalty = config.error_penalty
    correct = [t.latency_ms for t in trials if t.correct]
    mean_correct = _mean(correct)

    if penalty is ErrorPenalty.BUILT_IN:
        # Nothing to substitute: the task already charged the participant for
        # the error by making them correct it, and that time is in the latency.
        return list(trials), mean_correct

    if penalty is ErrorPenalty.DELETE:
        return [t for t in trials if t.correct], mean_correct

    if mean_correct is None:
        # Every trial in the block was an error, so there is nothing to anchor
        # the replacement to and the block is unusable.
        return [], None

    if penalty is ErrorPenalty.MEAN_PLUS_600:
        replacement = mean_correct + 600.0
    elif penalty is ErrorPenalty.MEAN_PLUS_400:
        replacement = mean_correct + 400.0
    elif penalty is ErrorPenalty.MEAN_PLUS_2SD:
        sd_c = float(np.std(np.asarray(correct, dtype=float), ddof=1)) if len(correct) > 1 else 0.0
        replacement = mean_correct + 2.0 * sd_c
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unhandled penalty {penalty!r}")

    return [t if t.correct else t.with_latency(replacement) for t in trials], mean_correct


def _summarise(
    block_index: int,
    role: str,
    pairing: str,
    retained: list[Trial],
    scored: list[Trial],
    mean_correct: float | None,
) -> BlockSummary:
    errors = sum(1 for t in retained if not t.correct)
    lat = [t.latency_ms for t in retained]
    return BlockSummary(
        block_index=block_index,
        block_role=role,
        pairing=pairing,
        n_presented=len(retained),
        n_retained=len(retained),
        n_errors=errors,
        error_rate=(errors / len(retained)) if retained else 0.0,
        mean_correct_ms=mean_correct,
        mean_scored_ms=_mean([t.latency_ms for t in scored]),
        median_ms=float(np.median(lat)) if lat else None,
        sd_ms=float(np.std(np.asarray(lat, dtype=float), ddof=1)) if len(lat) > 1 else None,
    )


def compute_d(
    trials: list[Trial],
    config: ScoringConfig | None = None,
    *,
    target_a_label: str = "Brand A",
    target_b_label: str = "Brand B",
    pole_a_label: str = "positive",
    instrument: Instrument = Instrument.IAT,
) -> DScoreResult:
    """Score a two-target IAT session."""
    config = config or ScoringConfig()
    warnings: list[str] = []
    exclusions: list[str] = []

    # --- Step 1: keep only the four combined blocks -----------------------
    relevant = [t for t in trials if t.block_role in ("practice", "test") and t.pairing]
    if not relevant:
        return _empty(config, instrument, "No combined-block trials present.")

    n_presented_total = len(relevant)
    relevant = _project(relevant, config)
    if len(relevant) < n_presented_total:
        warnings.append(
            f"{n_presented_total - len(relevant)} trials had no recorded "
            f"time-to-correct-response and could not be scored under the "
            f"built-in error penalty."
        )

    # --- Step 2: latency treatment ----------------------------------------
    if config.winsorize:
        # Richetin et al. (2015): pull extreme latencies to a fixed boundary
        # rather than deleting them. Deleting an extreme trial throws away the
        # fact that it happened; winsorizing keeps the trial and caps its
        # leverage. They found this outperformed trimming on most datasets.
        retained = [
            t if config.winsor_low_ms <= t.latency_ms <= config.winsor_high_ms
            else t.with_latency(min(max(t.latency_ms, config.winsor_low_ms), config.winsor_high_ms))
            for t in relevant
            if not t.timed_out
        ]
        n_dropped = sum(1 for t in relevant if t.timed_out)
    else:
        def keep(t: Trial) -> bool:
            if t.timed_out:
                return False
            if t.latency_ms > config.upper_cutoff_ms:
                return False
            if config.lower_cutoff_ms is not None and t.latency_ms < config.lower_cutoff_ms:
                return False
            return True

        retained = [t for t in relevant if keep(t)]
        n_dropped = len(relevant) - len(retained)

    if n_dropped:
        warnings.append(
            f"{n_dropped} of {n_presented_total} combined-block trials removed by "
            f"the latency cut-offs."
        )

    if config.drop_first_n_per_block:
        trimmed: list[Trial] = []
        for b in sorted({t.block_index for t in retained}):
            block = sorted((t for t in retained if t.block_index == b), key=lambda t: t.index)
            trimmed.extend(block[config.drop_first_n_per_block:])
        retained = trimmed

    # --- Step 3: participant-level exclusion ------------------------------
    if retained:
        too_fast = sum(1 for t in retained if t.latency_ms < config.fast_trial_threshold_ms)
        fast_prop = too_fast / len(retained)
        if fast_prop > config.fast_trial_proportion_limit:
            exclusions.append(
                f"{fast_prop:.0%} of trials were faster than "
                f"{config.fast_trial_threshold_ms:.0f} ms (limit "
                f"{config.fast_trial_proportion_limit:.0%}). The participant was "
                f"responding faster than the stimuli could be read."
            )
    else:
        exclusions.append("No trials survived the latency cut-offs.")

    # Group by (role, pairing). Position is irrelevant; pairing is what the
    # algorithm subtracts.
    if config.pool_practice_and_test:
        # Richetin et al. (2015): one D over the pooled practice+test trials
        # rather than the average of two quotients.
        roles = ["pooled"]
        cells = {
            ("pooled", "congruent"): [t for t in retained if t.pairing == "congruent"],
            ("pooled", "incongruent"): [t for t in retained if t.pairing == "incongruent"],
        }
    else:
        roles = ["practice", "test"]
        cells = {
            (r, p): [t for t in retained if t.block_role == r and t.pairing == p]
            for r in roles for p in ("congruent", "incongruent")
        }

    for (role, pairing), cell in cells.items():
        if len(cell) < config.min_trials_per_block:
            exclusions.append(
                f"The {pairing} {role} block retained only {len(cell)} trials "
                f"(minimum {config.min_trials_per_block})."
            )

    # --- Steps 4 and 6: error handling per block --------------------------
    penalised: dict[tuple[str, str], list[Trial]] = {}
    mean_correct_by_cell: dict[tuple[str, str], float | None] = {}
    for key, cell in cells.items():
        scored, mean_correct = _apply_penalty(cell, config)
        penalised[key] = scored
        mean_correct_by_cell[key] = mean_correct

    # --- Step 5: the inclusive pooled SD, one per block pair --------------
    def sd_for(role: str) -> float | None:
        if config.sd_before_penalty:
            pool = cells[(role, "congruent")] + cells[(role, "incongruent")]
        else:
            pool = penalised[(role, "congruent")] + penalised[(role, "incongruent")]
        lat = [t.latency_ms for t in pool] if config.inclusive_sd else [
            t.latency_ms for t in pool if t.correct
        ]
        return _pooled_sd(lat, config.sd_ddof)

    sds = {role: sd_for(role) for role in roles}

    # --- Step 7: block means over ALL scored trials -----------------------
    summaries: list[BlockSummary] = []
    scored_means: dict[tuple[str, str], float | None] = {}
    n_scored = 0
    for key, scored in penalised.items():
        role, pairing = key
        n_scored += len(scored)
        scored_means[key] = _mean([t.latency_ms for t in scored])
        cell = cells[key]
        summaries.append(_summarise(
            cell[0].block_index if cell else -1, role, pairing, cell, scored,
            mean_correct_by_cell[key],
        ))
    summaries.sort(key=lambda s: (s.block_role != "practice", s.pairing))

    # --- Steps 8, 9: difference / SD --------------------------------------
    def quotient(role: str) -> float | None:
        m_con = scored_means.get((role, "congruent"))
        m_inc = scored_means.get((role, "incongruent"))
        sd = sds.get(role)
        if m_con is None or m_inc is None or sd is None:
            return None
        return (m_inc - m_con) / sd

    quotients = {role: quotient(role) for role in roles}
    parts = [q for q in quotients.values() if q is not None]

    # --- Step 10: average the quotients -----------------------------------
    d = float(np.mean(parts)) if parts else None

    if d is not None and len(roles) == 2 and len(parts) == 1:
        warnings.append(
            "Only one of the practice/test block pairs was scorable, so D is "
            "based on half the usual data and is correspondingly noisier."
        )

    overall_errors = sum(s.n_errors for s in summaries)
    overall_trials = sum(s.n_retained for s in summaries)
    error_rate = overall_errors / overall_trials if overall_trials else 0.0
    if error_rate > 0.30:
        warnings.append(
            f"Overall error rate was {error_rate:.0%}. Above roughly 30% the "
            f"speed-accuracy trade-off dominates and D should not be read as a "
            f"clean association measure."
        )

    excluded = bool(exclusions)
    if excluded:
        d = None
        quotients = {r: None for r in roles}

    direction, interpretation = _interpret(
        d, target_a_label, target_b_label, pole_a_label
    )

    return DScoreResult(
        d=d,
        d_practice=quotients.get("practice") if not config.pool_practice_and_test else None,
        d_test=quotients.get("test") if not config.pool_practice_and_test else quotients.get("pooled"),
        sd_practice=sds.get("practice"),
        sd_test=sds.get("test") if not config.pool_practice_and_test else sds.get("pooled"),
        instrument=instrument.value,
        interpretation=interpretation,
        direction=direction,
        magnitude=_magnitude(d) if d is not None else "none",
        blocks=summaries,
        config=config.to_dict(),
        excluded=excluded,
        exclusion_reasons=exclusions,
        warnings=warnings,
        n_trials_scored=n_scored,
        n_trials_dropped=n_dropped,
    )


def _empty(config: ScoringConfig, instrument: Instrument, reason: str) -> DScoreResult:
    return DScoreResult(
        d=None, d_practice=None, d_test=None, sd_practice=None, sd_test=None,
        instrument=instrument.value,
        interpretation="No scorable trials were submitted.",
        direction="none", magnitude="none",
        config=config.to_dict(), excluded=True, exclusion_reasons=[reason],
    )


def _interpret(
    d: float | None, a: str, b: str, pole_a: str
) -> tuple[str, str]:
    if d is None:
        return "none", (
            "No D-score could be computed. The session did not meet the data-quality "
            "criteria, so no association estimate is reported."
        )
    mag = _magnitude(d)
    if mag == "little to no":
        return "neutral", (
            f"D = {d:+.3f}. Little to no difference in association strength between "
            f"{a} and {b}. Sorting was about as easy under either pairing."
        )
    favoured, other = (a, b) if d > 0 else (b, a)
    return ("a" if d > 0 else "b"), (
        f"D = {d:+.3f}. A {mag} relative association favouring {favoured} over "
        f"{other}: sorting was faster when {favoured} shared a response key with "
        f"'{pole_a}'."
    )
