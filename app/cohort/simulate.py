"""A synthetic respondent panel with known answers.

A cohort pipeline has more ways to be quietly wrong than a single-session one:
an exclusion rule that never fires, a segment comparison with the labels
swapped, a significance flag computed on the wrong family of tests. The only
way to check those is to run the pipeline on data where the truth is known. So
this module builds a panel the way a fieldwork agency would deliver one — a
flat trial-level dump with the demographic columns repeated on every row — but
from a generative model with every effect planted deliberately:

*   **Premium** — Aurelia is ahead overall, and further ahead among
    high-loyalty buyers.
*   **Eco-friendly** — Northvane is ahead (a negative D), more so among
    18–34s.
*   **Trustworthy** — flat among respondents who did not see the campaign;
    respondents randomised to the campaign cell lean clearly to Aurelia. This
    is the "did the ad work" read, and the overall number understates it.
*   **Device** — no effect on anything. A variable that *should* come back null
    is as useful a check as one that should not.

It also plants the respondents a real panel always contains, in roughly the
proportions online panels produce them:

*   fast responders who are clicking through without reading,
*   inaccurate responders who are guessing,
*   dropouts who did not finish the battery,
*   and a handful of genuine participants who walked away mid-trial, leaving
    single latencies above ten seconds that the trial-level cut-off should
    remove without costing anyone their place in the sample.

Latencies are ex-Gaussian (see :mod:`app.simulate`), because a pipeline tested
on normal data is a pipeline tested on data it will never see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..design import DIMENSIONS, STUDIES, Study, build_session
from ..simulate import MU_BASE, SIGMA_BASE, TAU_BASE

#: The three brand attributes the demo panel is tested on.
DEMO_ATTRIBUTES = ("premium", "eco", "trust")

#: Participant variables, with their levels in display order.
DEMO_SEGMENTS: dict[str, dict] = {
    "loyalty": {
        "label": "Brand loyalty",
        "levels": ["High", "Low"],
        "weights": [0.35, 0.65],
        "note": "Self-reported Aurelia purchase frequency in the screener.",
    },
    "exposure": {
        "label": "Campaign cell",
        "levels": ["Exposed", "Control"],
        "weights": [0.5, 0.5],
        "note": "Randomised: shown the new Aurelia film before the tests, or not.",
    },
    "age_band": {
        "label": "Age band",
        "levels": ["18-34", "35-54", "55+"],
        "weights": [0.34, 0.38, 0.28],
        "note": "From the screener.",
    },
    "device": {
        "label": "Device",
        "levels": ["Desktop", "Laptop"],
        "weights": [0.45, 0.55],
        "note": "Detected from the user agent. Planted with no effect.",
    },
}

#: Planted effects on the *true* D, per attribute. Keys are (variable, level).
PLANTED: dict[str, dict] = {
    "premium": {"base": 0.20, ("loyalty", "High"): 0.28, ("age_band", "55+"): 0.06},
    "eco": {"base": -0.12, ("age_band", "18-34"): -0.22},
    "trust": {"base": -0.12, ("exposure", "Exposed"): 0.26, ("loyalty", "High"): 0.06},
}

#: Between-person spread of the true association around the segment mean.
TRUE_D_SD = 0.24

#: Respondent-type mix. Everything not listed is a genuine respondent.
JUNK_RATES = {"fast": 0.055, "inaccurate": 0.05, "dropout": 0.03, "distracted": 0.02}

STUDY_NAME = "Aurelia vs Northvane — brand image study"


@dataclass(slots=True)
class SimulatedBatch:
    trials: pd.DataFrame
    meta: dict
    truth: pd.DataFrame


def _study_for(attribute: str) -> Study:
    base = STUDIES["gin-premium"]
    return Study(
        id=f"brand-image-{attribute}",
        name=f"Aurelia vs Northvane — {DIMENSIONS[attribute].name}",
        sector=base.sector,
        blurb=base.blurb,
        target_a=base.target_a,
        target_b=base.target_b,
        dimension=DIMENSIONS[attribute],
    )


def _exgauss(rng: np.random.Generator, n: int, mu, sigma, tau) -> np.ndarray:
    return rng.normal(mu, sigma, n) + rng.exponential(tau, n)


def simulate_batch(
    n_participants: int = 520,
    *,
    seed: int = 2026,
    attributes: tuple[str, ...] = DEMO_ATTRIBUTES,
    preset: str = "standard",
) -> SimulatedBatch:
    """Generate a complete trial-level panel dump.

    Deterministic in ``seed``: the same call always yields the same frame,
    which is what lets the test-suite assert on specific planted respondents.
    """
    rng = np.random.default_rng(seed)
    studies = {a: _study_for(a) for a in attributes}

    # ---- participants -------------------------------------------------------
    pids = [f"P{i + 1:04d}" for i in range(n_participants)]
    seg_values = {
        var: rng.choice(spec["levels"], size=n_participants, p=spec["weights"])
        for var, spec in DEMO_SEGMENTS.items()
    }
    kinds = rng.choice(
        ["fast", "inaccurate", "dropout", "distracted", "genuine"],
        size=n_participants,
        p=[*JUNK_RATES.values(), 1 - sum(JUNK_RATES.values())],
    )

    truth_rows = []
    frames: list[pd.DataFrame] = []

    for i, pid in enumerate(pids):
        kind = kinds[i]
        # Person-level speed and accuracy: real panels are heterogeneous, and a
        # simulator where everyone has the same RT distribution flatters the
        # inclusive-SD denominator.
        mu_p = rng.normal(MU_BASE, 55.0)
        sigma_p = SIGMA_BASE * float(rng.lognormal(0, 0.15))
        tau_p = TAU_BASE * float(rng.lognormal(0, 0.25))
        err_p = float(np.clip(rng.beta(2.2, 34.0), 0.005, 0.2))
        if kind == "inaccurate":
            err_p = float(rng.uniform(0.3, 0.45))

        n_tasks = len(attributes)
        if kind == "dropout":
            # Stopped somewhere in the battery: either a whole task is missing or
            # the last task was abandoned part-way through.
            n_tasks = int(rng.integers(1, len(attributes) + 1))

        for t_i, attr in enumerate(attributes[:n_tasks]):
            true_d = PLANTED[attr]["base"]
            for var in DEMO_SEGMENTS:
                true_d += PLANTED[attr].get((var, seg_values[var][i]), 0.0)
            true_d += rng.normal(0, TRUE_D_SD)
            truth_rows.append({"participant_id": pid, "attribute": attr,
                               "true_d": true_d, "kind": kind})

            design = build_session(
                studies[attr], preset=preset,
                seed=int(rng.integers(0, 2**31 - 1)), include_calibration=False,
            )
            tr = design["trials"]
            n = len(tr)
            pairing = np.array([t["pairing"] or "" for t in tr])
            role = np.array([t["block_role"] or "" for t in tr])

            gap = true_d * np.sqrt(sigma_p**2 + tau_p**2)
            mu = np.full(n, mu_p)
            mu[pairing == "incongruent"] += gap / 2
            mu[pairing == "congruent"] -= gap / 2
            mu[role == ""] -= 60.0  # single-discrimination blocks are easier
            lat = _exgauss(rng, n, mu, sigma_p, tau_p)

            p_err = np.where(pairing == "incongruent", err_p * 1.5, err_p)
            correct = rng.random(n) > p_err

            if kind == "fast":
                # Clicking through: most responses faster than anyone can read.
                fast = rng.random(n) < rng.uniform(0.35, 0.8)
                lat[fast] = rng.uniform(140, 290, fast.sum())
                correct[fast] = rng.random(fast.sum()) > 0.4
            if kind == "distracted":
                k = int(rng.integers(1, 4))
                idx = rng.choice(np.where(role != "")[0], size=k, replace=False)
                lat[idx] = rng.uniform(11_000, 40_000, k)

            lat = np.maximum(lat, 120.0)
            correction = _exgauss(rng, n, 420.0, 90.0, 120.0)
            to_correct = np.where(correct, lat, lat + correction)

            # A dropout always abandons the last task they start, before its
            # final block — so their data is incomplete however far they got.
            if kind == "dropout" and t_i == n_tasks - 1:
                keep = int(rng.integers(n // 3, n - 30))
            else:
                keep = n

            ck = np.array([t["correct_key"] for t in tr])
            frames.append(pd.DataFrame({
                "participant_id": pid,
                **{var: seg_values[var][i] for var in DEMO_SEGMENTS},
                "attribute": attr,
                "target_a": studies[attr].target_a.label,
                "target_b": studies[attr].target_b.label,
                "pole_a": studies[attr].dimension.pole_a.label,
                "pole_b": studies[attr].dimension.pole_b.label,
                "task_order": t_i + 1,
                "trial_index": np.arange(n),
                "block_index": [t["block_index"] for t in tr],
                "block_role": role,
                "pairing": pairing,
                "stimulus": [t["stimulus"] for t in tr],
                "stimulus_category": [t["stimulus_category"] for t in tr],
                "correct_key": ck,
                "response_key": np.where(correct, ck, np.where(ck == "E", "I", "E")),
                "latency_ms": np.round(lat, 2),
                "latency_to_correct_ms": np.round(to_correct, 2),
                "correct": correct,
                "timed_out": False,
            }).iloc[:keep])

    trials = pd.concat(frames, ignore_index=True)
    meta = {
        "name": STUDY_NAME,
        "source": "simulated",
        "seed": seed,
        "preset": preset,
        "n_participants": n_participants,
        "attributes": {
            a: {
                "label": DIMENSIONS[a].pole_a.label,
                "dimension": DIMENSIONS[a].name,
                "target_a": studies[a].target_a.label,
                "target_b": studies[a].target_b.label,
                "pole_a": DIMENSIONS[a].pole_a.label,
                "pole_b": DIMENSIONS[a].pole_b.label,
            }
            for a in attributes
        },
        "segments": {
            var: {"label": spec["label"], "levels": spec["levels"], "note": spec["note"]}
            for var, spec in DEMO_SEGMENTS.items()
        },
        "note": (
            "Synthetic panel. Latencies are ex-Gaussian with segment effects planted "
            "deliberately, plus fast, inaccurate, dropout and distracted respondents "
            "at realistic online-panel rates. Nobody took these tests."
        ),
    }
    return SimulatedBatch(trials=trials, meta=meta, truth=pd.DataFrame(truth_rows))
