"""Participant screening and vectorised D for a whole panel.

The reference implementation in :mod:`app.scoring.dscore` is written for
clarity: one participant, Python lists, every step spelled out. Scoring a
520-person, three-attribute panel with it means ~1,500 calls and a second or
two — fine once, far too slow to re-run every time a researcher drags a
trimming slider. This module computes the same quantity for every
(participant, attribute) task at once with ``np.bincount`` group-bys.

Two implementations of one algorithm is a liability unless they are pinned
together, so ``tests/test_cohort.py`` scores every task in a simulated panel
both ways and requires agreement to 1e-9. The reference stays the
specification; this is the fast path that is proven to match it.

Screening is at the **participant** level and deliberately separate from
trial-level trimming, in this order:

1.  *Complete data* — every attribute task present, and every scored
    (role × pairing) cell has at least ``min_trials`` trials.
2.  *Latency screen* — more than ``fast_limit`` of combined-block trials
    faster than ``fast_ms`` (Greenwald et al., 2003, step 3). Evaluated after
    the 10-second cut but **before** any lower trim, so moving the trim slider
    cannot hide a fast responder by deleting the evidence against them.
3.  *Accuracy screen* — overall accuracy on the combined blocks below
    ``min_accuracy``. The 2003 algorithm has no accuracy rule; commercial
    practice adds one, and 75% is a common threshold.
4.  *Sufficient trials after trimming* — only fires if the operator trims so
    aggressively that a cell falls below the minimum.

Participants are dropped whole (complete-case). An analysis that lets a
respondent into the Premium read but not the Trust read makes the two
attributes' samples differ, and then a difference between attributes could be
a difference between samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLES = ("practice", "test")
PAIRINGS = ("congruent", "incongruent")

STAGES = [
    ("incomplete", "Incomplete data",
     "Did not finish every attribute task, or a scored block has too few trials."),
    ("fast", "Latency screen",
     "More than {fast_limit:.0%} of combined-block trials faster than {fast_ms:.0f} ms — "
     "responding faster than the words can be read."),
    ("inaccurate", "Accuracy screen",
     "Accuracy on the combined blocks below {min_accuracy:.0%} — guessing rather than sorting."),
    ("insufficient", "Trimming",
     "A scored block fell below {min_trials} trials after the trial-level trim."),
]


@dataclass(frozen=True, slots=True)
class CohortParams:
    """Every analysis knob, with defaults matching the published procedure."""

    #: Trial-level trim. ``lower_ms = 0`` means no lower trim, which is the
    #: Greenwald (2003) recommendation: fast trials are signal, and the
    #: participant-level screen is what deals with fast responders.
    lower_ms: float = 0.0
    upper_ms: float = 10_000.0
    fast_ms: float = 300.0
    fast_limit: float = 0.10
    min_accuracy: float = 0.75
    min_trials: int = 4

    @classmethod
    def from_query(cls, q: dict) -> "CohortParams":
        def num(key: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(q.get(key, default))
            except (TypeError, ValueError):
                v = default
            return float(min(hi, max(lo, v)))
        lower = num("lower_ms", 0.0, 0.0, 1500.0)
        upper = num("upper_ms", 10_000.0, 1000.0, 30_000.0)
        return cls(
            lower_ms=lower,
            upper_ms=max(upper, lower + 100.0),
            fast_ms=num("fast_ms", 300.0, 100.0, 600.0),
            fast_limit=num("fast_limit", 0.10, 0.0, 1.0),
            min_accuracy=num("min_accuracy", 0.75, 0.0, 1.0),
            min_trials=int(num("min_trials", 4, 2, 40)),
        )

    def to_dict(self) -> dict:
        return {
            "lower_ms": self.lower_ms, "upper_ms": self.upper_ms,
            "fast_ms": self.fast_ms, "fast_limit": self.fast_limit,
            "min_accuracy": self.min_accuracy, "min_trials": self.min_trials,
        }

    def is_default(self) -> bool:
        return self == CohortParams()


@dataclass(slots=True)
class Prepared:
    """A batch reduced to the integer-coded arrays the scorer runs on.

    Built once per batch and cached; every re-analysis starts from here.
    """

    pids: np.ndarray
    attrs: list[str]
    segments: pd.DataFrame          # index: participant_id; one column per variable
    p: np.ndarray                   # participant code per combined-block trial
    a: np.ndarray                   # attribute code
    role: np.ndarray                # 0 practice, 1 test
    pair: np.ndarray                # 0 congruent, 1 incongruent
    lat: np.ndarray                 # scored latency (time-to-correct), NaN if unscorable
    first_lat: np.ndarray           # first-response latency
    correct: np.ndarray
    timed_out: np.ndarray
    n_rows_total: int

    @property
    def n_participants(self) -> int:
        return len(self.pids)


def prepare(trials: pd.DataFrame, segment_vars: list[str]) -> Prepared:
    pids = pd.unique(trials["participant_id"].astype(str))
    attrs = [str(a) for a in pd.unique(trials["attribute"].astype(str))]
    p_code = {pid: i for i, pid in enumerate(pids)}
    a_code = {a: i for i, a in enumerate(attrs)}

    scored = trials[trials["block_role"].isin(ROLES) & trials["pairing"].isin(PAIRINGS)]
    correct = scored["correct"].to_numpy(dtype=bool)
    first = scored["latency_ms"].to_numpy(dtype=float)
    to_correct = scored["latency_to_correct_ms"].to_numpy(dtype=float)
    # Built-in error penalty: forced correction means an error trial is scored
    # on its time-to-correct-response. A missing one is unscorable (NaN), which
    # is exactly what the reference implementation does.
    lat = np.where(correct, first, to_correct)

    seg = (
        trials.groupby("participant_id", sort=False)[segment_vars].first()
        if segment_vars else pd.DataFrame(index=pd.Index(pids, name="participant_id"))
    )
    seg.index = seg.index.astype(str)
    seg = seg.reindex(pids)

    return Prepared(
        pids=np.asarray(pids),
        attrs=attrs,
        segments=seg,
        p=scored["participant_id"].astype(str).map(p_code).to_numpy(dtype=np.int64),
        a=scored["attribute"].astype(str).map(a_code).to_numpy(dtype=np.int64),
        role=(scored["block_role"] == "test").to_numpy(dtype=np.int64),
        pair=(scored["pairing"] == "incongruent").to_numpy(dtype=np.int64),
        lat=lat,
        first_lat=first,
        correct=correct,
        timed_out=scored["timed_out"].to_numpy(dtype=bool),
        n_rows_total=len(trials),
    )


@dataclass(slots=True)
class Scored:
    params: CohortParams
    status: np.ndarray              # (P,) stage key or "included"
    included: np.ndarray            # (P,) bool
    fast_prop: np.ndarray           # (P,)
    accuracy: np.ndarray            # (P,)
    d: np.ndarray                   # (P, A) NaN unless included
    d_practice: np.ndarray
    d_test: np.ndarray
    dlog: np.ndarray                # (P, A) mean log RT incongruent − congruent
    mean_con: np.ndarray            # (P, A) ms
    mean_inc: np.ndarray
    n_trials_kept: np.ndarray       # (P, A)
    funnel: list[dict] = field(default_factory=list)
    trimming: dict = field(default_factory=dict)


def _group_mean(keys: np.ndarray, x: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.bincount(keys, minlength=size).astype(float)
    s = np.bincount(keys, weights=x, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n


def score(prep: Prepared, params: CohortParams) -> Scored:
    P, A = prep.n_participants, len(prep.attrs)
    lat = prep.lat
    presented = ~prep.timed_out
    valid = np.isfinite(lat) & presented
    with np.errstate(invalid="ignore"):
        base = valid & (lat <= params.upper_ms)
        fast = base & (lat < params.fast_ms)
        keep = base & (lat >= params.lower_ms) if params.lower_ms > 0 else base

    cell = ((prep.p * A + prep.a) * 2 + prep.role) * 2 + prep.pair
    ncell = P * A * 4

    # ---- participant screening --------------------------------------------
    n_base = np.bincount(prep.p, weights=base, minlength=P)
    n_fast = np.bincount(prep.p, weights=fast, minlength=P)
    n_pres = np.bincount(prep.p, weights=presented, minlength=P)
    n_corr = np.bincount(prep.p, weights=prep.correct & presented, minlength=P)
    with np.errstate(invalid="ignore", divide="ignore"):
        fast_prop = np.where(n_base > 0, n_fast / np.maximum(n_base, 1), np.nan)
        accuracy = np.where(n_pres > 0, n_corr / np.maximum(n_pres, 1), np.nan)

    raw_counts = np.bincount(cell, weights=presented, minlength=ncell).reshape(P, A * 4)
    complete = (raw_counts >= params.min_trials).all(axis=1)

    kept_counts = np.bincount(cell, weights=keep, minlength=ncell).reshape(P, A, 2, 2)
    sufficient = (kept_counts >= params.min_trials).all(axis=(1, 2, 3))

    status = np.full(P, "included", dtype=object)
    remaining = np.ones(P, dtype=bool)
    for key, mask in (
        ("incomplete", ~complete),
        ("fast", fast_prop > params.fast_limit),
        ("inaccurate", accuracy < params.min_accuracy),
        ("insufficient", ~sufficient),
    ):
        hit = remaining & mask
        status[hit] = key
        remaining &= ~hit
    included = remaining

    # ---- D for every task, then masked to the analytic sample --------------
    x = lat[keep]
    c = cell[keep]
    mean4, n4 = _group_mean(c, x, ncell)

    k2 = c // 2                                   # (participant, attribute, role)
    m2, n2 = _group_mean(k2, x, P * A * 2)
    ss2 = np.bincount(k2, weights=(x - m2[k2]) ** 2, minlength=P * A * 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd2 = np.sqrt(ss2 / (n2 - 1))
    sd2[(n2 <= 1) | (sd2 <= 0)] = np.nan

    mean4 = mean4.reshape(P, A, 2, 2)
    q = (mean4[..., 1] - mean4[..., 0]) / sd2.reshape(P, A, 2)
    d = q.mean(axis=2)

    # Log-RT difference per task, roles pooled: the input to the parametric
    # test. Logging makes a right-skewed latency distribution close to
    # symmetric, which is the assumption a t-test actually needs.
    k3 = (c // 4) * 2 + (c % 2)                   # (participant, attribute, pairing)
    mlog, _ = _group_mean(k3, np.log(x), P * A * 2)
    mms, n3 = _group_mean(k3, x, P * A * 2)
    mlog = mlog.reshape(P, A, 2)
    mms = mms.reshape(P, A, 2)

    mask = ~included[:, None]
    def masked(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(float).copy()
        out[np.broadcast_to(mask, out.shape)] = np.nan
        return out

    # ---- trial accounting for the analytic sample -------------------------
    inc_rows = included[prep.p]
    with np.errstate(invalid="ignore"):
        trimming = {
            "combined_trials": int((inc_rows & presented).sum()),
            "unscorable": int((inc_rows & presented & ~np.isfinite(lat)).sum()),
            "timed_out": int((inc_rows & prep.timed_out).sum()),
            "removed_upper": int((inc_rows & valid & (lat > params.upper_ms)).sum()),
            "removed_lower": int((inc_rows & base & ~keep).sum()),
            "analysed": int((inc_rows & keep).sum()),
        }
    trimming["removed_pct"] = (
        (trimming["removed_upper"] + trimming["removed_lower"]) / trimming["combined_trials"]
        if trimming["combined_trials"] else 0.0
    )

    return Scored(
        params=params,
        status=status,
        included=included,
        fast_prop=fast_prop,
        accuracy=accuracy,
        d=masked(d),
        d_practice=masked(q[..., 0]),
        d_test=masked(q[..., 1]),
        dlog=masked(mlog[..., 1] - mlog[..., 0]),
        mean_con=masked(mms[..., 0]),
        mean_inc=masked(mms[..., 1]),
        n_trials_kept=masked(kept_counts.sum(axis=(2, 3))),
        funnel=_funnel(status, params),
        trimming=trimming,
    )


def _funnel(status: np.ndarray, params: CohortParams) -> list[dict]:
    fmt = params.to_dict()
    n = len(status)
    out = [{"key": "recruited", "label": "Recruited", "n": n, "removed": 0,
            "reason": "Every participant in the upload."}]
    remaining = n
    for key, label, reason in STAGES:
        removed = int((status == key).sum())
        remaining -= removed
        out.append({"key": key, "label": label, "n": remaining, "removed": removed,
                    "reason": reason.format(**fmt)})
    out.append({"key": "final", "label": "Final analytic sample", "n": remaining,
                "removed": 0, "reason": "Included in every attribute's analysis."})
    return out
