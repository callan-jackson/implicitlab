"""Core data structures for implicit-measure scoring.

Everything downstream of the browser is expressed in terms of a :class:`Trial`.
A trial is deliberately dumb: it records what was shown, what was pressed, how
long it took, and which experimental condition it belonged to. All of the
interpretation lives in the scoring modules so that the raw data stays
re-analysable under a different algorithm later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

#: In a two-target IAT, every combined block pairs the targets with the
#: attributes in one of two ways. We label them by *hypothesis*, not by
#: position, because block order is counterbalanced across participants.
#:
#:   "congruent"   -> target_a + positive   /  target_b + negative
#:   "incongruent" -> target_a + negative   /  target_b + positive
#:
#: "Congruent" here is a naming convention only. It carries no assumption
#: about which pairing is actually easier; that is what D measures.
Pairing = Literal["congruent", "incongruent"]

#: Blocks used by the scoring algorithm. Greenwald, Nosek & Banaji (2003) use
#: the two combined *practice* blocks (B3/B6) and the two combined *test*
#: blocks (B4/B7). The single-category blocks are only used by the SC-IAT.
BlockRole = Literal["practice", "test", "sciat"]


class ErrorPenalty(str, Enum):
    """How to handle trials on which the participant pressed the wrong key.

    The choice matters and is one of the things that separates the D variants
    reported in Greenwald, Nosek & Banaji (2003, Table 3).
    """

    #: Delete error trials entirely. Only defensible when the procedure forces
    #: the participant to correct their error before continuing, because the
    #: correction time is then already "built in" to the recorded latency.
    DELETE = "delete"

    #: Replace the error latency with the block mean of *correct* responses
    #: plus a fixed 600 ms. This is the penalty in the recommended procedure
    #: for designs *without* a built-in correction requirement.
    MEAN_PLUS_600 = "mean_plus_600"

    #: Replace the error latency with the block mean of correct responses plus
    #: two standard deviations of that block's correct responses. Scales the
    #: penalty to the participant's own variability.
    MEAN_PLUS_2SD = "mean_plus_2sd"

    #: SC-IAT convention (Karpinski & Steinman, 2006): a 400 ms penalty.
    MEAN_PLUS_400 = "mean_plus_400"

    #: No substitution at all, because the *task* already penalised the error:
    #: the participant had to press the correct key before the trial advanced,
    #: so the recorded latency is time-to-correct-response and the correction
    #: time is built in. Greenwald, Nosek & Banaji (2003) treat this as an
    #: equivalent procedure and Greenwald, Brendl et al. (2022, Appendix B)
    #: label it the preferred one. It is the default here because it makes the
    #: 600 ms vs 2SD argument disappear: there is nothing arbitrary to choose.
    BUILT_IN = "built_in"


class LatencySource(str, Enum):
    """Which of the two clocks each trial recorded should be scored.

    The front end records both: time to the *first* keypress whatever it was,
    and time to the *correct* keypress after any forced correction. Storing
    both means either scoring procedure can be run against the same raw file
    rather than the choice being baked into data collection.
    """

    FIRST_RESPONSE = "first_response"
    TO_CORRECT = "to_correct"


class ScoringVariant(str, Enum):
    """Which published set of analytic choices to apply."""

    #: Greenwald, Nosek & Banaji (2003) improved algorithm.
    GREENWALD_2003 = "greenwald_2003"

    #: Richetin, Costantini, Perugini & Schonbrodt (2015, PLOS ONE 10(6):
    #: e0129601) tested the algorithm's choices empirically rather than by
    #: convention and recommend departing from the canon on two points:
    #: winsorize extreme latencies to a fixed value instead of deleting them,
    #: and pool the practice and test trials into a single D rather than
    #: averaging two. They report improvements on 76-83% of the datasets tested.
    RICHETIN_2015 = "richetin_2015"


class Instrument(str, Enum):
    """Which implicit measure a session ran."""

    IAT = "iat"
    SC_IAT = "sciat"


# --------------------------------------------------------------------------
# Trial
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trial:
    """One stimulus presentation and one keypress.

    Attributes
    ----------
    index:
        Position of the trial within the whole session (0-based). Kept so the
        raw file can be re-ordered without losing sequence information.
    block_index:
        1-based block number as it was actually presented (B1..B7).
    block_role:
        Whether this block feeds the practice term, the test term, or is an
        SC-IAT block. Blocks that the algorithm ignores are role ``None``.
    pairing:
        Which of the two pairings the block used. ``None`` for the
        single-discrimination blocks (B1, B2, B5) that scoring ignores.
    latency_ms:
        Response latency in milliseconds, measured in the browser from the
        first animation frame on which the stimulus was actually painted to
        the ``keydown`` event timestamp. Float, sub-millisecond resolution
        permitting (see ``docs/TIMING.md``).
    correct:
        Whether the keypress matched the required response.
    stimulus:
        The word or image label that was shown.
    stimulus_category:
        Which category the stimulus belonged to (e.g. ``"target_a"``,
        ``"positive"``).
    response_key:
        The physical key pressed, normalised to upper case.
    onset_uncertainty_ms:
        Half the frame interval on which the stimulus was painted. This is the
        irreducible quantisation error on stimulus onset for a display-based
        task and is carried through so it can be reported rather than hidden.
    timed_out:
        True if the response window elapsed with no keypress (SC-IAT only).
    """

    index: int
    block_index: int
    latency_ms: float
    correct: bool
    stimulus: str
    stimulus_category: str
    response_key: str
    block_role: BlockRole | None = None
    pairing: Pairing | None = None
    onset_uncertainty_ms: float = 0.0
    timed_out: bool = False
    latency_to_correct_ms: float | None = None
    n_corrections: int = 0
    dispatch_delay_ms: float | None = None
    focus_lost: bool = False

    def with_latency(self, latency_ms: float) -> "Trial":
        """Return a copy with a substituted latency (used by error penalties)."""
        return Trial(
            index=self.index,
            block_index=self.block_index,
            latency_ms=latency_ms,
            correct=self.correct,
            stimulus=self.stimulus,
            stimulus_category=self.stimulus_category,
            response_key=self.response_key,
            block_role=self.block_role,
            pairing=self.pairing,
            onset_uncertainty_ms=self.onset_uncertainty_ms,
            timed_out=self.timed_out,
            latency_to_correct_ms=self.latency_to_correct_ms,
            n_corrections=self.n_corrections,
            dispatch_delay_ms=self.dispatch_delay_ms,
            focus_lost=self.focus_lost,
        )


# --------------------------------------------------------------------------
# Scoring configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Every knob the scoring algorithm exposes, with the published defaults.

    The defaults reproduce the *recommended* improved algorithm of Greenwald,
    Nosek & Banaji (2003, Table 4) — the variant labelled D in most applied
    work. Deviating from the defaults is legitimate but should be a deliberate,
    recorded decision, which is why the config is serialised into every result.
    """

    #: Step 2: discard trials slower than this. Not a "cleaning" step so much
    #: as removing trials where the participant clearly disengaged.
    upper_cutoff_ms: float = 10_000.0

    #: Step 3: participant-level exclusion. If more than
    #: ``fast_trial_proportion_limit`` of a participant's scored trials are
    #: faster than this, the participant was not reading the stimuli.
    fast_trial_threshold_ms: float = 300.0
    fast_trial_proportion_limit: float = 0.10

    #: Step 6: error handling.
    error_penalty: ErrorPenalty = ErrorPenalty.MEAN_PLUS_600

    #: Step 5: the denominator. The "inclusive" SD pools *all* retained trials
    #: from both pairings of a block pair — correct and error alike — and is
    #: computed *before* the error penalty is applied. Setting this False gives
    #: the correct-trials-only denominator used by the SC-IAT.
    inclusive_sd: bool = True

    #: Sample (n-1) rather than population standard deviation.
    sd_ddof: int = 1

    #: Optional lower trim on individual trials. The improved IAT algorithm
    #: deliberately does *not* trim fast trials (fast responding is signal, not
    #: noise, and trimming it biases D downwards); the SC-IAT does, at 350 ms.
    lower_cutoff_ms: float | None = None

    #: SC-IAT drops the first N trials of each block as warm-up.
    drop_first_n_per_block: int = 0

    #: Minimum retained trials per scored block before the block is unusable.
    min_trials_per_block: int = 4

    #: Which recorded clock to score. With the built-in penalty the correct
    #: choice is ``TO_CORRECT``; with a substitution penalty it is
    #: ``FIRST_RESPONSE``, because the substitution is what stands in for the
    #: correction time.
    latency_source: LatencySource = LatencySource.TO_CORRECT

    #: Whether the inclusive SD is computed before or after error latencies are
    #: replaced.
    #:
    #: The two canonical sources disagree. Greenwald, Nosek & Banaji (2003,
    #: Table 4) order the steps so the SD is computed first, on raw latencies.
    #: Greenwald, Brendl et al. (2022, Appendix B, Step 8) say the opposite -
    #: "compute the two inclusive SDs using all trials (using the error trials
    #: with their replaced latencies)" - and both widely used reference
    #: implementations (iatgen's ``cleanIAT``, implicitMeasures' ``compute_iat``)
    #: follow the 2022 wording.
    #:
    #: Rather than pick a side silently, this is a flag, both paths are tested,
    #: and the default study uses the built-in penalty where no substitution
    #: happens and the question therefore cannot arise.
    sd_before_penalty: bool = True

    #: Winsorize rather than delete out-of-range latencies (Richetin variant).
    winsorize: bool = False
    winsor_low_ms: float = 400.0
    winsor_high_ms: float = 10_000.0

    #: Pool practice and test trials into one D instead of averaging two
    #: (Richetin variant).
    pool_practice_and_test: bool = False

    variant: ScoringVariant = ScoringVariant.GREENWALD_2003

    def to_dict(self) -> dict:
        return {
            "upper_cutoff_ms": self.upper_cutoff_ms,
            "fast_trial_threshold_ms": self.fast_trial_threshold_ms,
            "fast_trial_proportion_limit": self.fast_trial_proportion_limit,
            "error_penalty": self.error_penalty.value,
            "inclusive_sd": self.inclusive_sd,
            "sd_ddof": self.sd_ddof,
            "lower_cutoff_ms": self.lower_cutoff_ms,
            "drop_first_n_per_block": self.drop_first_n_per_block,
            "min_trials_per_block": self.min_trials_per_block,
            "latency_source": self.latency_source.value,
            "sd_before_penalty": self.sd_before_penalty,
            "winsorize": self.winsorize,
            "winsor_low_ms": self.winsor_low_ms,
            "winsor_high_ms": self.winsor_high_ms,
            "pool_practice_and_test": self.pool_practice_and_test,
            "variant": self.variant.value,
        }


#: The recommended IAT procedure as this tool ships it: forced error correction
#: in the task, so time-to-correct is scored and nothing is substituted.
GREENWALD_2003_D = ScoringConfig(
    error_penalty=ErrorPenalty.BUILT_IN,
    latency_source=LatencySource.TO_CORRECT,
)

#: The same algorithm scored the other way: first-response latencies with the
#: fixed 600 ms substitution. Kept so both can be reported side by side from
#: one raw file.
GREENWALD_2003_D_600 = ScoringConfig(
    error_penalty=ErrorPenalty.MEAN_PLUS_600,
    latency_source=LatencySource.FIRST_RESPONSE,
)

#: Richetin et al. (2015): winsorize instead of trim, and pool practice with
#: test rather than averaging two quotients.
RICHETIN_2015_D = ScoringConfig(
    error_penalty=ErrorPenalty.BUILT_IN,
    latency_source=LatencySource.TO_CORRECT,
    winsorize=True,
    pool_practice_and_test=True,
    variant=ScoringVariant.RICHETIN_2015,
)

#: D with the variance-scaled penalty instead of the fixed 600 ms.
GREENWALD_2003_D_2SD = ScoringConfig(
    error_penalty=ErrorPenalty.MEAN_PLUS_2SD,
    latency_source=LatencySource.FIRST_RESPONSE,
)

#: Karpinski & Steinman (2006) single-category scoring. Note that when the task
#: forces error correction the +400 ms substitution is redundant and the
#: built-in penalty is used instead; the parameter is kept so the published
#: procedure can be reproduced from the same data.
KARPINSKI_2006_SCIAT = ScoringConfig(
    upper_cutoff_ms=10_000.0,
    lower_cutoff_ms=350.0,
    error_penalty=ErrorPenalty.BUILT_IN,
    latency_source=LatencySource.TO_CORRECT,
    inclusive_sd=False,
    drop_first_n_per_block=0,
    fast_trial_threshold_ms=350.0,
    min_trials_per_block=8,
)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class BlockSummary:
    """Descriptives for one scored block, reported alongside D."""

    block_index: int
    block_role: str
    pairing: str
    n_presented: int
    n_retained: int
    n_errors: int
    error_rate: float
    mean_correct_ms: float | None
    mean_scored_ms: float | None
    median_ms: float | None
    sd_ms: float | None

    def to_dict(self) -> dict:
        return {
            "block_index": self.block_index,
            "block_role": self.block_role,
            "pairing": self.pairing,
            "n_presented": self.n_presented,
            "n_retained": self.n_retained,
            "n_errors": self.n_errors,
            "error_rate": self.error_rate,
            "mean_correct_ms": self.mean_correct_ms,
            "mean_scored_ms": self.mean_scored_ms,
            "median_ms": self.median_ms,
            "sd_ms": self.sd_ms,
        }


@dataclass(slots=True)
class DScoreResult:
    """The output of a scoring run.

    ``d`` is ``None`` when the data did not support a defensible score. That is
    a deliberate design decision: returning a number for unusable data is how
    demos mislead people.
    """

    d: float | None
    d_practice: float | None
    d_test: float | None
    sd_practice: float | None
    sd_test: float | None
    instrument: str
    interpretation: str
    direction: str
    magnitude: str
    blocks: list[BlockSummary] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    excluded: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_trials_scored: int = 0
    n_trials_dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "d": self.d,
            "d_practice": self.d_practice,
            "d_test": self.d_test,
            "sd_practice": self.sd_practice,
            "sd_test": self.sd_test,
            "instrument": self.instrument,
            "interpretation": self.interpretation,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "blocks": [b.to_dict() for b in self.blocks],
            "config": self.config,
            "excluded": self.excluded,
            "exclusion_reasons": self.exclusion_reasons,
            "warnings": self.warnings,
            "n_trials_scored": self.n_trials_scored,
            "n_trials_dropped": self.n_trials_dropped,
        }
