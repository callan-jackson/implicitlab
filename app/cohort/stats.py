"""Group-level inference for a panel of D-scores.

Two families of test are run on every analysis, and both are always computed
so the interface can switch between them without a round trip:

**Parametric** — a t-test on *log-transformed* reaction times. Each
participant contributes one number per attribute: the difference in mean log
latency between the incongruent and congruent pairings. Raw latencies are
right-skewed and a t-test on them is not defensible; their logs are close to
symmetric, and a mean of per-person log differences is closer still by the
central limit theorem. One-sample tests use ``ttest_1samp``; segment
comparisons use Welch's unequal-variance test, because segments of a panel
rarely have equal variances and nothing is lost by not assuming it.

**Non-parametric** — permutation tests on D itself. A one-sample test flips
the sign of each participant's D at random (under the null of no association,
D is symmetric about zero); a segment comparison shuffles segment labels.
Neither assumes a distributional shape.

The two ask slightly different questions of slightly different quantities, so
they are not expected to agree to the third decimal. They *are* expected to
agree on which results clear the bar, and the report counts the cases where
they do not.

**Multiple comparisons.** An attribute × segment grid is a lot of tests. At
α = .05, twenty true-null cells produce one "significant" result on average,
and a client deck built from an uncorrected grid will contain it. Every
p-value is therefore accompanied by a Benjamini–Hochberg q-value computed
within its family, and the significance flags are set on q.

Confidence intervals are percentile bootstraps over *participants*: at group
level the participant, not the trial, is the unit of sampling.

Every resampling routine is seeded from a stable hash of what it is testing,
so the same batch with the same parameters produces the same numbers on
screen, in the workbook and in the deck.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import stats as sps

N_BOOT = 2000
N_PERM = 5000
ALPHA = 0.05
LOW_BASE = 30

BASE_SEED = 20260923


def rng_for(*parts: object) -> np.random.Generator:
    """A generator seeded from a stable hash of ``parts``.

    ``hash()`` is salted per process, so it cannot be used here; CRC32 of the
    text is stable across processes, machines and Python versions.
    """
    key = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(BASE_SEED ^ zlib.crc32(key))


# --------------------------------------------------------------------------
# Resampling primitives. X is (n, k): n participants, k attributes. Sharing
# one set of resample indices across attributes is valid (each attribute's
# test is marginally exact) and makes a whole panel one matrix operation.
# --------------------------------------------------------------------------


def bootstrap_mean_ci(X: np.ndarray, rng: np.random.Generator, *, level: float = 0.95,
                      n_boot: int = N_BOOT) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 2:
        nan = np.full(X.shape[1], np.nan)
        return nan, nan
    idx = rng.integers(0, n, size=(n_boot, n))
    means = X[idx].mean(axis=1)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [a, 100 - a], axis=0)
    return lo, hi


def signflip_p(X: np.ndarray, rng: np.random.Generator, *, n_perm: int = N_PERM) -> np.ndarray:
    """Two-sided one-sample permutation p-value (sign-flip), add-one corrected."""
    n = X.shape[0]
    if n < 2:
        return np.full(X.shape[1], np.nan)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null = signs @ X / n
    obs = np.abs(X.mean(axis=0))
    extreme = (np.abs(null) >= obs - 1e-12).sum(axis=0)
    return (extreme + 1) / (n_perm + 1)


def perm_diff_p(Xa: np.ndarray, Xb: np.ndarray, rng: np.random.Generator,
                *, n_perm: int = N_PERM) -> np.ndarray:
    """Two-sided permutation p-value for a difference in means (label shuffle)."""
    na, nb = Xa.shape[0], Xb.shape[0]
    if na < 2 or nb < 2:
        return np.full(Xa.shape[1], np.nan)
    pooled = np.vstack([Xa, Xb])
    base = np.zeros(na + nb)
    base[:na] = 1.0
    masks = rng.permuted(np.tile(base, (n_perm, 1)), axis=1)
    sum_a = masks @ pooled
    total = pooled.sum(axis=0)
    null = sum_a / na - (total - sum_a) / nb
    obs = np.abs(Xa.mean(axis=0) - Xb.mean(axis=0))
    extreme = (np.abs(null) >= obs - 1e-12).sum(axis=0)
    return (extreme + 1) / (n_perm + 1)


def bootstrap_diff_ci(Xa: np.ndarray, Xb: np.ndarray, rng: np.random.Generator,
                      *, level: float = 0.95, n_boot: int = N_BOOT):
    na, nb = Xa.shape[0], Xb.shape[0]
    if na < 2 or nb < 2:
        nan = np.full(Xa.shape[1], np.nan)
        return nan, nan
    ia = rng.integers(0, na, size=(n_boot, na))
    ib = rng.integers(0, nb, size=(n_boot, nb))
    diffs = Xa[ia].mean(axis=1) - Xb[ib].mean(axis=1)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(diffs, [a, 100 - a], axis=0)
    return lo, hi


def hedges_g(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    na, nb = Xa.shape[0], Xb.shape[0]
    if na < 2 or nb < 2:
        return np.full(Xa.shape[1], np.nan)
    sp = np.sqrt(((na - 1) * Xa.var(axis=0, ddof=1) + (nb - 1) * Xb.var(axis=0, ddof=1))
                 / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)          # small-sample correction
    with np.errstate(invalid="ignore", divide="ignore"):
        return j * (Xa.mean(axis=0) - Xb.mean(axis=0)) / sp


def ttest_one(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = Y.shape[0]
    if n < 2:
        nan = np.full(Y.shape[1], np.nan)
        return nan, nan, 0
    r = sps.ttest_1samp(Y, 0.0, axis=0)
    return np.asarray(r.statistic), np.asarray(r.pvalue), n - 1


def welch(Ya: np.ndarray, Yb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    na, nb = Ya.shape[0], Yb.shape[0]
    if na < 2 or nb < 2:
        nan = np.full(Ya.shape[1], np.nan)
        return nan, nan, nan
    r = sps.ttest_ind(Ya, Yb, equal_var=False, axis=0)
    va, vb = Ya.var(axis=0, ddof=1) / na, Yb.var(axis=0, ddof=1) / nb
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return np.asarray(r.statistic), np.asarray(r.pvalue), df


def bh_fdr(p: list[float | None]) -> list[float | None]:
    """Benjamini–Hochberg q-values. ``None`` entries are passed through."""
    idx = [i for i, v in enumerate(p) if v is not None and np.isfinite(v)]
    out: list[float | None] = [None] * len(p)
    m = len(idx)
    if not m:
        return out
    vals = np.array([p[i] for i in idx], dtype=float)
    order = np.argsort(vals)
    ranked = vals[order] * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downwards.
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.minimum(q, 1.0)
    for rank, j in enumerate(order):
        out[idx[j]] = float(q[rank])
    return out
