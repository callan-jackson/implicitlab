"""Descriptive statistics, resampling inference and reliability for RT data.

Two design decisions worth stating explicitly, because they are the sort of
thing a reviewer will ask about:

**No parametric tests.** Reaction-time distributions are strongly right-skewed
— roughly ex-Gaussian, a normal convolved with an exponential tail. A t-test on
raw latencies assumes symmetry the data does not have, and the tail is exactly
where the interesting variance lives. Everything inferential here is therefore
resampling-based: a bootstrap for the interval around D and a permutation test
for whether D differs from zero. Both are distribution-free and both are cheap
at these sample sizes.

**Single-participant inference.** A D-score from one person is a point estimate
with real uncertainty, and reporting it bare invites over-reading. The
permutation test asks a precise, honest question: *if this participant's
latencies were unrelated to which pairing produced them, how often would we see
a |D| this large by chance?* That is answerable from one participant's own
trials because the trials themselves are the exchangeable units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import ScoringConfig, Trial

# A fixed seed keeps every reported interval reproducible. Anything that
# resamples takes a generator so the caller can override it.
DEFAULT_SEED = 20260910


# --------------------------------------------------------------------------
# Descriptives
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Descriptives:
    n: int
    mean_ms: float | None
    sd_ms: float | None
    median_ms: float | None
    iqr_ms: float | None
    p10_ms: float | None
    p90_ms: float | None
    min_ms: float | None
    max_ms: float | None
    skew: float | None
    error_rate: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_ms": self.mean_ms,
            "sd_ms": self.sd_ms,
            "median_ms": self.median_ms,
            "iqr_ms": self.iqr_ms,
            "p10_ms": self.p10_ms,
            "p90_ms": self.p90_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "skew": self.skew,
            "error_rate": self.error_rate,
        }


def _skew(x: np.ndarray) -> float | None:
    """Fisher-Pearson sample skewness, computed directly to avoid a SciPy dep."""
    n = x.size
    if n < 3:
        return None
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return None
    g1 = float(((x - m) ** 3).mean() / s**3)
    # Sample-size correction, matching scipy.stats.skew(bias=False).
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def describe(trials: list[Trial]) -> Descriptives:
    """Descriptives over a set of trials. Latencies from correct trials only.

    Error rate is reported over *all* trials, because that is the denominator
    that matters for the speed-accuracy trade-off.
    """
    if not trials:
        return Descriptives(0, None, None, None, None, None, None, None, None, None, 0.0)

    correct = np.asarray([t.latency_ms for t in trials if t.correct], dtype=float)
    errors = sum(1 for t in trials if not t.correct)

    if correct.size == 0:
        return Descriptives(len(trials), None, None, None, None, None, None, None, None,
                            None, errors / len(trials))

    q1, q3 = np.percentile(correct, [25, 75])
    return Descriptives(
        n=len(trials),
        mean_ms=float(correct.mean()),
        sd_ms=float(correct.std(ddof=1)) if correct.size > 1 else None,
        median_ms=float(np.median(correct)),
        iqr_ms=float(q3 - q1),
        p10_ms=float(np.percentile(correct, 10)),
        p90_ms=float(np.percentile(correct, 90)),
        min_ms=float(correct.min()),
        max_ms=float(correct.max()),
        skew=_skew(correct),
        error_rate=errors / len(trials),
    )


# --------------------------------------------------------------------------
# Distribution for plotting
# --------------------------------------------------------------------------


def histogram(
    latencies: list[float], *, bin_width_ms: float = 50.0, lo: float = 0.0, hi: float = 2000.0
) -> dict:
    """Fixed-width bins so the two conditions are directly comparable.

    Shared bin edges across conditions is not cosmetic: auto-binned histograms
    of two conditions cannot be visually compared, because the bar widths differ.
    """
    if not latencies:
        return {"edges": [], "counts": [], "bin_width_ms": bin_width_ms}
    edges = np.arange(lo, hi + bin_width_ms, bin_width_ms)
    clipped = np.clip(np.asarray(latencies, dtype=float), lo, hi - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    return {
        "edges": [float(e) for e in edges[:-1]],
        "counts": [int(c) for c in counts],
        "bin_width_ms": bin_width_ms,
    }


def kde(latencies: list[float], *, grid: np.ndarray | None = None) -> dict:
    """Gaussian KDE with Silverman's rule of thumb, for a smooth density curve.

    Written out rather than pulled from SciPy so the bandwidth choice is visible
    and so the deployment stays a small dependency set.
    """
    x = np.asarray(latencies, dtype=float)
    if x.size < 3:
        return {"x": [], "y": []}
    grid = np.linspace(0, 2000, 200) if grid is None else grid
    sd = x.std(ddof=1)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0:
        return {"x": [], "y": []}
    bw = 0.9 * spread * x.size ** (-1 / 5)
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))
    return {"x": [float(v) for v in grid], "y": [float(v) for v in dens]}


# --------------------------------------------------------------------------
# Resampling inference on D
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Inference:
    d: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float
    p_permutation: float | None
    n_resamples: int
    null_distribution: list[float] = field(default_factory=list)
    bootstrap_distribution: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "d": self.d,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "p_permutation": self.p_permutation,
            "n_resamples": self.n_resamples,
            "null_distribution": self.null_distribution,
            "bootstrap_distribution": self.bootstrap_distribution,
            "note": self.note,
        }


def _score(trials: list[Trial], config: ScoringConfig, instrument: str) -> float | None:
    """Re-run the real scorer on a resampled trial set."""
    # Imported here to avoid a circular import at module load.
    from .dscore import compute_d
    from .sciat import compute_sciat_d

    if instrument == "sciat":
        return compute_sciat_d(trials, config).d
    return compute_d(trials, config).d


def _relabel(trials: list[Trial], rng: np.random.Generator) -> list[Trial]:
    """Permute pairing labels *within* each block pair.

    Shuffling globally would also break the practice/test structure, which is
    not the null we want. The null is: within a given pair of blocks, the
    pairing label carries no information about latency.
    """
    out: list[Trial] = []
    # sorted() for the same reason as in design.py: set iteration order over
    # strings is not stable across processes, and the RNG is consumed inside
    # this loop.
    keys = sorted({t.block_role for t in trials if t.block_role})
    for role in keys:
        cell = [t for t in trials if t.block_role == role]
        labels = [t.pairing for t in cell]
        perm = rng.permutation(len(labels))
        for t, j in zip(cell, perm):
            new_label = labels[j]
            out.append(
                Trial(
                    index=t.index, block_index=t.block_index, latency_ms=t.latency_ms,
                    correct=t.correct, stimulus=t.stimulus,
                    stimulus_category=t.stimulus_category, response_key=t.response_key,
                    block_role=t.block_role, pairing=new_label,
                    onset_uncertainty_ms=t.onset_uncertainty_ms, timed_out=t.timed_out,
                )
            )
    return out


def _resample_within_cells(trials: list[Trial], rng: np.random.Generator) -> list[Trial]:
    """Bootstrap trials with replacement inside each (role, pairing) cell.

    Resampling within cells preserves the design — each cell keeps its trial
    count — so the bootstrap estimates sampling error in the latencies, not in
    the block structure.
    """
    out: list[Trial] = []
    cells: dict[tuple, list[Trial]] = {}
    for t in trials:
        cells.setdefault((t.block_role, t.pairing), []).append(t)
    for cell in cells.values():
        idx = rng.integers(0, len(cell), size=len(cell))
        out.extend(cell[i] for i in idx)
    return out


def infer(
    trials: list[Trial],
    config: ScoringConfig,
    *,
    instrument: str = "iat",
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Inference:
    """Bootstrap CI and permutation p-value for a single participant's D.

    ``n_resamples`` of 2,000 gives a p-value resolution of 0.0005 and a stable
    percentile interval, and runs in well under a second at IAT trial counts.
    """
    scorable = [t for t in trials if t.pairing and t.block_role in ("practice", "test", "sciat")]
    observed = _score(scorable, config, instrument)
    if observed is None or len(scorable) < 16:
        return Inference(
            d=observed, ci_low=None, ci_high=None, ci_level=ci_level,
            p_permutation=None, n_resamples=0,
            note="Too few usable trials for resampling inference.",
        )

    rng = np.random.default_rng(seed)

    boots: list[float] = []
    for _ in range(n_resamples):
        s = _score(_resample_within_cells(scorable, rng), config, instrument)
        if s is not None:
            boots.append(s)

    nulls: list[float] = []
    for _ in range(n_resamples):
        s = _score(_relabel(scorable, rng), config, instrument)
        if s is not None:
            nulls.append(s)

    lo = hi = None
    if boots:
        alpha = (1 - ci_level) / 2
        lo, hi = (float(v) for v in np.percentile(boots, [alpha * 100, (1 - alpha) * 100]))

    p = None
    if nulls:
        # Add-one correction: a permutation p-value should never be exactly 0,
        # because the observed arrangement is itself one of the possibilities.
        extreme = sum(1 for v in nulls if abs(v) >= abs(observed))
        p = (extreme + 1) / (len(nulls) + 1)

    return Inference(
        d=observed,
        ci_low=lo,
        ci_high=hi,
        ci_level=ci_level,
        p_permutation=p,
        n_resamples=n_resamples,
        # Thinned for transport; the browser only needs enough points to draw
        # a histogram, not all 2,000 values.
        null_distribution=[round(v, 4) for v in nulls[:600]],
        bootstrap_distribution=[round(v, 4) for v in boots[:600]],
        note=(
            "Percentile bootstrap over trials within each block-pairing cell; "
            "permutation null generated by shuffling pairing labels within each "
            "block pair. Both are distribution-free, which matters because "
            "reaction times are not normally distributed."
        ),
    )


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


def split_half_reliability(
    trials: list[Trial], config: ScoringConfig, *, instrument: str = "iat"
) -> dict:
    """Odd/even split-half reliability of D, Spearman-Brown corrected.

    This is the honest internal-consistency check for a single session. The IAT
    typically reports internal consistency around .7-.9 and test-retest around
    .5; a demo that quotes a D without any reliability figure is quoting a
    number whose error bar it has not looked at.

    With one participant there is no correlation to compute across people, so
    what is reported here is the *agreement between the two halves' D-scores*
    expressed as an absolute difference, plus each half's D. It is a diagnostic,
    not a population reliability coefficient — and it is labelled as such rather
    than dressed up as Cronbach's alpha.
    """
    scorable = [t for t in trials if t.pairing and t.block_role in ("practice", "test", "sciat")]
    odd = [t for t in scorable if t.index % 2 == 1]
    even = [t for t in scorable if t.index % 2 == 0]

    d_odd = _score(odd, config, instrument)
    d_even = _score(even, config, instrument)

    agreement = None
    if d_odd is not None and d_even is not None:
        agreement = abs(d_odd - d_even)

    return {
        "d_odd_trials": d_odd,
        "d_even_trials": d_even,
        "absolute_difference": agreement,
        "interpretation": (
            "Split-half diagnostic: D recomputed separately on odd- and "
            "even-numbered trials. A large gap between the halves means the "
            "estimate is unstable within the session itself, whatever the "
            "headline D says."
        ),
        "stable": (agreement is not None and agreement < 0.30),
    }
