"""Tests for the D-score.

The important ones here are not "does it run". They are the four places where a
plausible-looking implementation is silently wrong:

* the denominator (inclusive SD, not Cohen's pooled-within SD),
* which mean goes into the numerator,
* blocks resolved by pairing rather than by presentation order,
* the fast-responder rule excluding a *participant* rather than filtering trials.

Each of those has a test that would fail if the code took the wrong branch, and
the expected values are computed here by explicit arithmetic rather than by
calling the module under test.
"""

from __future__ import annotations

import math

import pytest

from app.scoring.dscore import compute_d
from app.scoring.models import (
    GREENWALD_2003_D,
    GREENWALD_2003_D_600,
    ErrorPenalty,
    LatencySource,
    ScoringConfig,
    Trial,
)

BUILT_IN = GREENWALD_2003_D


def make(
    latencies_by_cell: dict[tuple[str, str], list[float]],
    *,
    correct_by_cell: dict[tuple[str, str], list[bool]] | None = None,
    block_index_by_cell: dict[tuple[str, str], int] | None = None,
) -> list[Trial]:
    """Build a trial list from a {(role, pairing): [latencies]} mapping."""
    default_blocks = {
        ("practice", "congruent"): 3,
        ("test", "congruent"): 4,
        ("practice", "incongruent"): 6,
        ("test", "incongruent"): 7,
    }
    blocks = {**default_blocks, **(block_index_by_cell or {})}
    trials: list[Trial] = []
    idx = 0
    for (role, pairing), lats in latencies_by_cell.items():
        flags = (correct_by_cell or {}).get((role, pairing), [True] * len(lats))
        for lat, ok in zip(lats, flags):
            trials.append(Trial(
                index=idx, block_index=blocks[(role, pairing)], latency_ms=lat,
                correct=ok, stimulus="X", stimulus_category="target_a",
                response_key="E", block_role=role, pairing=pairing,
                latency_to_correct_ms=lat,
            ))
            idx += 1
    return trials


# --------------------------------------------------------------------------
# The headline check: a fully hand-worked D
# --------------------------------------------------------------------------


def test_d_matches_hand_computation():
    """Reproduce a D computed by hand, step by step, to 10 decimal places.

    Practice congruent   [500, 600, 700,  800]   mean 650
    Practice incongruent [700, 800, 900, 1000]   mean 850
    Inclusive SD over all eight practice trials  = sqrt(180000 / 7)
    D_practice = (850 - 650) / SD

    Test congruent       [600, 700, 800,  900]   mean 750
    Test incongruent     [800, 900, 1000, 1100]  mean 950
    Same spread, so the same SD, and D_test = D_practice.
    """
    trials = make({
        ("practice", "congruent"): [500, 600, 700, 800],
        ("practice", "incongruent"): [700, 800, 900, 1000],
        ("test", "congruent"): [600, 700, 800, 900],
        ("test", "incongruent"): [800, 900, 1000, 1100],
    })

    sd = math.sqrt(180000 / 7)          # sample SD, ddof = 1
    expected_practice = (850 - 650) / sd
    expected_test = (950 - 750) / sd
    expected_d = (expected_practice + expected_test) / 2

    result = compute_d(trials, BUILT_IN)

    assert result.excluded is False
    assert result.d == pytest.approx(expected_d, abs=1e-10)
    assert result.d_practice == pytest.approx(expected_practice, abs=1e-10)
    assert result.d_test == pytest.approx(expected_test, abs=1e-10)
    assert result.sd_practice == pytest.approx(sd, abs=1e-10)
    # Sanity: this dataset is engineered to give a large positive D.
    assert result.d == pytest.approx(1.2472191289, abs=1e-8)
    assert result.magnitude == "strong"


# --------------------------------------------------------------------------
# The denominator
# --------------------------------------------------------------------------


def test_denominator_is_inclusive_sd_not_cohens_pooled_within():
    """The inclusive SD must differ from Cohen's pooled-within-condition SD.

    Cohen's pooled-within SD removes the between-block mean difference before
    pooling. The IAT's inclusive SD deliberately keeps it. On any dataset where
    the two conditions differ in mean, the two quantities are not equal — and a
    D computed with Cohen's SD is systematically too large.
    """
    con = [500.0, 600.0, 700.0, 800.0]
    inc = [700.0, 800.0, 900.0, 1000.0]
    trials = make({
        ("practice", "congruent"): con,
        ("practice", "incongruent"): inc,
        ("test", "congruent"): con,
        ("test", "incongruent"): inc,
    })

    def sample_sd(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    inclusive = sample_sd(con + inc)
    # Cohen's pooled-within: sqrt of the mean of the two within-condition
    # variances (equal n, so a simple average).
    cohens = math.sqrt((sample_sd(con) ** 2 + sample_sd(inc) ** 2) / 2)

    assert inclusive != pytest.approx(cohens)
    assert inclusive > cohens  # between-block variance is included on purpose

    result = compute_d(trials, BUILT_IN)
    assert result.sd_practice == pytest.approx(inclusive, abs=1e-10)
    assert result.sd_practice != pytest.approx(cohens, abs=1e-6)

    # And confirm the D that a Cohen's-SD implementation would have produced is
    # meaningfully different, so this is not a distinction without a difference.
    d_wrong = (sum(inc) / len(inc) - sum(con) / len(con)) / cohens
    assert abs(d_wrong - result.d) > 0.3


def test_denominator_is_not_sd_of_difference_scores():
    trials = make({
        ("practice", "congruent"): [500, 600, 700, 800],
        ("practice", "incongruent"): [700, 800, 900, 1000],
        ("test", "congruent"): [500, 600, 700, 800],
        ("test", "incongruent"): [700, 800, 900, 1000],
    })
    result = compute_d(trials, BUILT_IN)
    # SD of the four paired differences would be 0 here (every pair differs by
    # exactly 200), which would make D infinite. It must not be.
    assert result.d is not None
    assert math.isfinite(result.d)


def test_denominator_concatenates_rather_than_averaging_block_sds():
    """One SD over the concatenated trials, not the mean of two block SDs."""
    con = [400.0, 400.0, 400.0, 400.0]     # zero within-block variance
    inc = [800.0, 800.0, 800.0, 800.0]     # zero within-block variance
    trials = make({
        ("practice", "congruent"): con,
        ("practice", "incongruent"): inc,
        ("test", "congruent"): con,
        ("test", "incongruent"): inc,
    })
    result = compute_d(trials, BUILT_IN)
    # Averaging the two block SDs would give 0 and a division by zero. The
    # inclusive SD is driven entirely by the between-block difference here.
    assert result.sd_practice is not None and result.sd_practice > 0
    assert result.d is not None and math.isfinite(result.d)


# --------------------------------------------------------------------------
# Which mean goes in the numerator
# --------------------------------------------------------------------------


def test_numerator_uses_all_scored_trials_not_correct_only():
    """With a substitution penalty, errors must be *in* the numerator mean.

    Block: three correct at 500/600/700 (mean 600) and one error. Under the
    +600 ms rule the error becomes 1200, so the numerator mean for that cell is
    (500 + 600 + 700 + 1200) / 4 = 750, not 600.
    """
    trials = make(
        {
            ("practice", "congruent"): [500, 600, 700, 999],
            ("practice", "incongruent"): [500, 600, 700, 800],
            ("test", "congruent"): [500, 600, 700, 800],
            ("test", "incongruent"): [500, 600, 700, 800],
        },
        correct_by_cell={("practice", "congruent"): [True, True, True, False]},
    )
    result = compute_d(trials, GREENWALD_2003_D_600)
    cell = next(
        b for b in result.blocks
        if b.block_role == "practice" and b.pairing == "congruent"
    )
    assert cell.mean_correct_ms == pytest.approx(600.0)   # Step 4 mean
    assert cell.mean_scored_ms == pytest.approx(750.0)    # Step 7 mean
    assert cell.n_errors == 1


def test_error_trials_are_not_deleted_by_default():
    trials = make(
        {
            ("practice", "congruent"): [500, 600, 700, 800],
            ("practice", "incongruent"): [500, 600, 700, 800],
            ("test", "congruent"): [500, 600, 700, 800],
            ("test", "incongruent"): [500, 600, 700, 800],
        },
        correct_by_cell={("practice", "congruent"): [True, True, False, False]},
    )
    result = compute_d(trials, GREENWALD_2003_D_600)
    cell = next(
        b for b in result.blocks
        if b.block_role == "practice" and b.pairing == "congruent"
    )
    assert cell.n_retained == 4          # nothing deleted
    assert cell.n_errors == 2


# --------------------------------------------------------------------------
# Counterbalancing
# --------------------------------------------------------------------------


def test_block_order_does_not_change_d():
    """Scoring must key on pairing, not on presentation order.

    The same latencies presented with the incongruent pairing first must give
    the same D. An implementation that subtracts "second block minus first
    block" flips the sign for half the sample.
    """
    cells = {
        ("practice", "congruent"): [500.0, 600.0, 700.0, 800.0],
        ("practice", "incongruent"): [700.0, 800.0, 900.0, 1000.0],
        ("test", "congruent"): [600.0, 700.0, 800.0, 900.0],
        ("test", "incongruent"): [800.0, 900.0, 1000.0, 1100.0],
    }
    congruent_first = compute_d(make(cells), BUILT_IN)
    incongruent_first = compute_d(
        make(cells, block_index_by_cell={
            ("practice", "incongruent"): 3,
            ("test", "incongruent"): 4,
            ("practice", "congruent"): 6,
            ("test", "congruent"): 7,
        }),
        BUILT_IN,
    )
    assert congruent_first.d == pytest.approx(incongruent_first.d, abs=1e-12)
    assert congruent_first.d > 0


def test_sign_convention():
    """D > 0 when the congruent pairing was faster."""
    faster_congruent = compute_d(make({
        ("practice", "congruent"): [400, 400, 400, 400],
        ("practice", "incongruent"): [900, 900, 900, 900],
        ("test", "congruent"): [400, 400, 400, 400],
        ("test", "incongruent"): [900, 900, 900, 900],
    }), BUILT_IN)
    assert faster_congruent.d > 0

    faster_incongruent = compute_d(make({
        ("practice", "congruent"): [900, 900, 900, 900],
        ("practice", "incongruent"): [400, 400, 400, 400],
        ("test", "congruent"): [900, 900, 900, 900],
        ("test", "incongruent"): [400, 400, 400, 400],
    }), BUILT_IN)
    assert faster_incongruent.d < 0
    assert faster_congruent.d == pytest.approx(-faster_incongruent.d, abs=1e-12)


# --------------------------------------------------------------------------
# Data-quality rules
# --------------------------------------------------------------------------


def test_fast_responder_rule_excludes_the_participant_not_the_trials():
    """More than 10% of trials under 300 ms excludes the whole session.

    The tell for a wrong implementation is a returned D with the fast trials
    quietly filtered out. Here 3 of 20 trials (15%) are fast, so D must be None
    and a reason must be recorded.
    """
    trials = make({
        ("practice", "congruent"): [250, 250, 250, 600, 600],
        ("practice", "incongruent"): [700, 700, 700, 700, 700],
        ("test", "congruent"): [600, 600, 600, 600, 600],
        ("test", "incongruent"): [700, 700, 700, 700, 700],
    })
    result = compute_d(trials, BUILT_IN)
    assert result.excluded is True
    assert result.d is None
    assert any("300 ms" in r for r in result.exclusion_reasons)

    # And at 5% (1 of 20) it must NOT exclude.
    ok = compute_d(make({
        ("practice", "congruent"): [250, 600, 600, 600, 600],
        ("practice", "incongruent"): [700, 700, 700, 700, 700],
        ("test", "congruent"): [600, 600, 600, 600, 600],
        ("test", "incongruent"): [700, 700, 700, 700, 700],
    }), BUILT_IN)
    assert ok.excluded is False
    assert ok.d is not None


def test_slow_trials_over_10s_are_dropped():
    trials = make({
        ("practice", "congruent"): [500, 600, 700, 800, 15000],
        ("practice", "incongruent"): [700, 800, 900, 1000, 1100],
        ("test", "congruent"): [600, 700, 800, 900, 1000],
        ("test", "incongruent"): [800, 900, 1000, 1100, 1200],
    })
    result = compute_d(trials, BUILT_IN)
    assert result.n_trials_dropped == 1
    cell = next(b for b in result.blocks if b.block_role == "practice" and b.pairing == "congruent")
    assert cell.n_retained == 4


def test_no_lower_trim_by_default():
    """The improved IAT algorithm does not trim fast trials. Fast is signal."""
    assert GREENWALD_2003_D.lower_cutoff_ms is None
    trials = make({
        ("practice", "congruent"): [310, 600, 700, 800],
        ("practice", "incongruent"): [700, 800, 900, 1000],
        ("test", "congruent"): [600, 700, 800, 900],
        ("test", "incongruent"): [800, 900, 1000, 1100],
    })
    result = compute_d(trials, BUILT_IN)
    cell = next(b for b in result.blocks if b.block_role == "practice" and b.pairing == "congruent")
    assert cell.n_retained == 4  # the 310 ms trial is kept


def test_insufficient_trials_blocks_a_score():
    trials = make({
        ("practice", "congruent"): [500, 600],
        ("practice", "incongruent"): [700, 800],
        ("test", "congruent"): [600, 700],
        ("test", "incongruent"): [800, 900],
    })
    result = compute_d(trials, BUILT_IN)
    assert result.excluded is True
    assert result.d is None


def test_no_trials_returns_a_reason_not_a_crash():
    result = compute_d([], BUILT_IN)
    assert result.d is None
    assert result.excluded is True
    assert result.exclusion_reasons


# --------------------------------------------------------------------------
# Error-penalty variants
# --------------------------------------------------------------------------


def test_built_in_penalty_scores_time_to_correct():
    """With forced correction, the scored latency is time-to-correct."""
    trials = [
        Trial(index=0, block_index=3, latency_ms=400.0, correct=False, stimulus="X",
              stimulus_category="target_a", response_key="I", block_role="practice",
              pairing="congruent", latency_to_correct_ms=1400.0, n_corrections=1),
    ]
    trials += make({
        ("practice", "congruent"): [500, 600, 700],
        ("practice", "incongruent"): [600, 700, 800, 900],
        ("test", "congruent"): [500, 600, 700, 800],
        ("test", "incongruent"): [600, 700, 800, 900],
    })
    result = compute_d(trials, BUILT_IN)
    cell = next(b for b in result.blocks if b.block_role == "practice" and b.pairing == "congruent")
    # 1400 (the corrected latency), not 400 (the first keypress).
    assert cell.mean_scored_ms == pytest.approx((1400 + 500 + 600 + 700) / 4)


def test_penalty_variants_give_different_numbers():
    """The choice of penalty is a real analytic decision, not a formality."""
    cells = {
        ("practice", "congruent"): [500.0, 600.0, 700.0, 800.0],
        ("practice", "incongruent"): [700.0, 800.0, 900.0, 1000.0],
        ("test", "congruent"): [600.0, 700.0, 800.0, 900.0],
        ("test", "incongruent"): [800.0, 900.0, 1000.0, 1100.0],
    }
    errs = {("practice", "congruent"): [True, True, True, False]}
    trials = make(cells, correct_by_cell=errs)

    d600 = compute_d(trials, GREENWALD_2003_D_600).d
    d2sd = compute_d(trials, ScoringConfig(
        error_penalty=ErrorPenalty.MEAN_PLUS_2SD,
        latency_source=LatencySource.FIRST_RESPONSE,
    )).d
    ddel = compute_d(trials, ScoringConfig(
        error_penalty=ErrorPenalty.DELETE,
        latency_source=LatencySource.FIRST_RESPONSE,
    )).d
    assert d600 != pytest.approx(d2sd)
    assert d600 != pytest.approx(ddel)


def test_sd_before_and_after_penalty_are_both_available_and_differ():
    """The two canonical sources disagree on this step, so both must work."""
    trials = make(
        {
            ("practice", "congruent"): [500.0, 600.0, 700.0, 800.0],
            ("practice", "incongruent"): [700.0, 800.0, 900.0, 1000.0],
            ("test", "congruent"): [600.0, 700.0, 800.0, 900.0],
            ("test", "incongruent"): [800.0, 900.0, 1000.0, 1100.0],
        },
        correct_by_cell={("practice", "congruent"): [True, True, True, False]},
    )
    before = compute_d(trials, ScoringConfig(
        error_penalty=ErrorPenalty.MEAN_PLUS_600,
        latency_source=LatencySource.FIRST_RESPONSE, sd_before_penalty=True))
    after = compute_d(trials, ScoringConfig(
        error_penalty=ErrorPenalty.MEAN_PLUS_600,
        latency_source=LatencySource.FIRST_RESPONSE, sd_before_penalty=False))
    assert before.sd_practice != pytest.approx(after.sd_practice)
    assert before.d != pytest.approx(after.d)


def test_interpretation_thresholds():
    def d_for(gap: float) -> float:
        trials = make({
            ("practice", "congruent"): [600.0] * 8,
            ("practice", "incongruent"): [600.0 + gap] * 8,
            ("test", "congruent"): [600.0] * 8,
            ("test", "incongruent"): [600.0 + gap] * 8,
        })
        return compute_d(trials, BUILT_IN).d

    assert compute_d(make({
        ("practice", "congruent"): [600.0, 610.0, 620.0, 630.0],
        ("practice", "incongruent"): [601.0, 611.0, 621.0, 631.0],
        ("test", "congruent"): [600.0, 610.0, 620.0, 630.0],
        ("test", "incongruent"): [601.0, 611.0, 621.0, 631.0],
    }), BUILT_IN).magnitude == "little to no"
    assert d_for(400) > 0.65
