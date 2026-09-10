"""Orchestration: raw trials in, complete report out.

This is the only module that knows about all the pieces at once. It exists so
that the API layer stays thin and so that the ordering — deterministic
statistics first, model second, always — is expressed in one place where it can
be read and checked.

The order is the point:

    trials -> quality assessment -> D (three ways) -> resampling inference
           -> reliability -> chart data -> THEN the language model

Nothing the model returns can change a number above it. If the model is
unavailable the report is complete without it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from .llm.client import InsightAgent
from .llm.prompts import build_payload
from .scoring import quality as quality_mod
from .scoring import stats as stats_mod
from .scoring.dscore import THRESHOLD_NOTE, compute_d
from .scoring.models import (
    GREENWALD_2003_D,
    GREENWALD_2003_D_600,
    KARPINSKI_2006_SCIAT,
    RICHETIN_2015_D,
    Instrument,
    ScoringConfig,
    Trial,
)
from .scoring.sciat import compute_sciat_d


def to_trials(records: list[dict]) -> list[Trial]:
    """Convert submitted JSON records into scoring trials.

    Calibration trials carry no ``block_role`` and are filtered out of scoring
    by every downstream function; they are kept in the list so the export and
    the covariate summary can use them.
    """
    out: list[Trial] = []
    for r in records:
        try:
            out.append(Trial(
                index=int(r["index"]),
                block_index=int(r["block_index"]),
                latency_ms=float(r["latency_ms"]),
                correct=bool(r["correct"]),
                stimulus=str(r.get("stimulus", "")),
                stimulus_category=str(r.get("stimulus_category", "")),
                response_key=str(r.get("response_key", "")),
                block_role=r.get("block_role") or None,
                pairing=r.get("pairing") or None,
                onset_uncertainty_ms=float(r.get("onset_uncertainty_ms") or 0.0),
                timed_out=bool(r.get("timed_out")),
                latency_to_correct_ms=(
                    None if r.get("latency_to_correct_ms") is None
                    else float(r["latency_to_correct_ms"])
                ),
                n_corrections=int(r.get("n_corrections") or 0),
                dispatch_delay_ms=(
                    None if r.get("dispatch_delay_ms") is None
                    else float(r["dispatch_delay_ms"])
                ),
                focus_lost=bool(r.get("focus_lost")),
            ))
        except (KeyError, TypeError, ValueError):
            # A malformed trial is dropped rather than allowed to abort the
            # whole session; the count of dropped rows is reported.
            continue
    return out


@dataclass(slots=True)
class Calibration:
    """Per-participant baselines derived from the two calibration blocks."""

    motor_median_ms: float | None
    motor_n: int
    motor_error_rate: float | None
    reading_slope_ms_per_word: float | None
    reading_intercept_ms: float | None
    reading_n: int
    note: str

    def to_dict(self) -> dict:
        return {
            "motor_median_ms": self.motor_median_ms,
            "motor_n": self.motor_n,
            "motor_error_rate": self.motor_error_rate,
            "reading_slope_ms_per_word": self.reading_slope_ms_per_word,
            "reading_intercept_ms": self.reading_intercept_ms,
            "reading_n": self.reading_n,
            "note": self.note,
        }


def calibrate(records: list[dict]) -> Calibration:
    """Summarise the motor and reading calibration blocks.

    These are reported as covariates and are deliberately **not** used to adjust
    D. D already standardises within the participant, so subtracting a motor
    baseline from it would be double-counting. What the baselines are good for
    is flagging a participant whose motor speed is an outlier — someone on a
    trackpad, or someone whose device is struggling — and for reading the raw
    latencies, where a 90 ms difference means something quite different for a
    participant whose baseline movement time is 220 ms than for one at 600 ms.
    """
    motor = [r for r in records if r.get("stimulus_category", "").startswith("motor")]
    reading = [r for r in records if r.get("stimulus_category") == "reading"]

    motor_ok = [float(r["latency_ms"]) for r in motor if r.get("correct")]
    motor_median = float(statistics.median(motor_ok)) if motor_ok else None
    motor_err = (sum(1 for r in motor if not r.get("correct")) / len(motor)) if motor else None

    slope = intercept = None
    good_reads = [
        (int(r.get("word_count") or 1), float(r["latency_ms"]))
        for r in reading
        if r.get("correct") and 100 < float(r["latency_ms"]) < 15000
    ]
    if len(good_reads) >= 3 and len({w for w, _ in good_reads}) >= 2:
        xs = np.array([w for w, _ in good_reads], dtype=float)
        ys = np.array([t for _, t in good_reads], dtype=float)
        slope_v, intercept_v = np.polyfit(xs, ys, 1)
        slope, intercept = float(slope_v), float(intercept_v)

    return Calibration(
        motor_median_ms=motor_median,
        motor_n=len(motor),
        motor_error_rate=motor_err,
        reading_slope_ms_per_word=slope,
        reading_intercept_ms=intercept,
        reading_n=len(reading),
        note=(
            "Baselines from the pre-task calibration blocks. Reported as "
            "covariates, not used to adjust D — D already standardises within "
            "the participant, so correcting it with these would double-count."
        ),
    )


def _condition_slices(trials: list[Trial]) -> dict[str, list[Trial]]:
    return {
        "congruent": [t for t in trials if t.pairing == "congruent" and t.block_role],
        "incongruent": [t for t in trials if t.pairing == "incongruent" and t.block_role],
    }


def _chart_data(trials: list[Trial]) -> dict:
    cells = _condition_slices(trials)
    out: dict = {"histograms": {}, "density": {}, "raw": {}}
    for name, cell in cells.items():
        correct = [t.latency_ms for t in cell if t.correct]
        out["histograms"][name] = stats_mod.histogram(correct)
        out["density"][name] = stats_mod.kde(correct)
        out["raw"][name] = [round(v, 1) for v in correct]
    # Trial-by-trial series, so practice effects and fatigue are visible rather
    # than averaged away.
    series = []
    for t in sorted((t for t in trials if t.block_role), key=lambda t: t.index):
        series.append({
            "index": t.index,
            "block": t.block_index,
            "pairing": t.pairing,
            "latency_ms": round(t.latency_ms, 1),
            "correct": t.correct,
        })
    out["series"] = series
    return out


def analyse(
    *,
    session: dict,
    records: list[dict],
    client_meta: dict,
    run_llm: bool = True,
    agent: InsightAgent | None = None,
    bootstrap_resamples: int = 2000,
) -> dict:
    """Produce the complete report for one session."""
    study = session.get("study", {})
    instrument = session.get("instrument", "iat")
    is_sciat = instrument == "sciat"

    trials = to_trials(records)
    n_malformed = len(records) - len(trials)

    a_label = (study.get("target_a") or {}).get("label", "Brand A")
    b_label = (study.get("target_b") or {}).get("label", "Brand B")
    pole_a = (study.get("positive") or {}).get("label", "positive")
    pole_b = (study.get("negative") or {}).get("label", "negative")

    primary_config = KARPINSKI_2006_SCIAT if is_sciat else GREENWALD_2003_D

    def score(cfg: ScoringConfig):
        if is_sciat:
            return compute_sciat_d(trials, cfg, target_label=a_label, pole_a_label=pole_a)
        return compute_d(
            trials, cfg, target_a_label=a_label, target_b_label=b_label,
            pole_a_label=pole_a, instrument=Instrument.IAT,
        )

    result = score(primary_config)

    # Two sensitivity analyses, reported alongside the headline number rather
    # than in a footnote. If the conclusion depends on which defensible
    # algorithm was chosen, that is a finding about the data, not a detail.
    sensitivity = []
    for label, cfg, why in (
        ("Greenwald 2003, +600 ms error penalty", GREENWALD_2003_D_600,
         "Scores the first keypress and substitutes block mean + 600 ms for errors, "
         "rather than scoring time-to-correct."),
        ("Richetin 2015 (winsorized, pooled)", RICHETIN_2015_D,
         "Winsorizes extreme latencies instead of deleting them and computes one D "
         "over pooled practice and test trials instead of averaging two."),
    ):
        if is_sciat:
            continue
        alt = score(cfg)
        sensitivity.append({
            "label": label,
            "why": why,
            "d": alt.d,
            "excluded": alt.excluded,
            "delta_from_primary": (
                None if (alt.d is None or result.d is None) else round(alt.d - result.d, 4)
            ),
        })

    quality = quality_mod.assess(trials, primary_config, client_meta=client_meta)
    if n_malformed:
        quality["checks"].append({
            "key": "malformed", "label": "Malformed rows", "severity": "warn",
            "detail": f"{n_malformed} submitted trial rows could not be parsed and were dropped.",
            "value": n_malformed,
        })

    inference = stats_mod.infer(
        trials, primary_config, instrument=instrument,
        n_resamples=bootstrap_resamples,
    ).to_dict()

    reliability = stats_mod.split_half_reliability(
        trials, primary_config, instrument=instrument
    )

    cells = _condition_slices(trials)
    descriptives = {
        "congruent": stats_mod.describe(cells["congruent"]).to_dict(),
        "incongruent": stats_mod.describe(cells["incongruent"]).to_dict(),
        "labels": {
            "congruent": f"{a_label} + {pole_a}",
            "incongruent": f"{a_label} + {pole_b}",
        },
    }

    calibration = calibrate(records)

    session_meta = {
        "preset_label": session.get("preset_label"),
        "preset_note": session.get("preset_note"),
        "n_trials": session.get("n_scored_trials") or session.get("n_trials"),
        "counterbalance": session.get("counterbalance"),
        "refresh_hz": client_meta.get("refresh_hz"),
        "onset_uncertainty_ms": client_meta.get("onset_uncertainty_ms"),
    }

    payload = build_payload(
        study=study, result=result.to_dict(), descriptives=descriptives,
        inference=inference, reliability=reliability, quality=quality,
        session_meta=session_meta,
    )

    insight = None
    if run_llm:
        insight = (agent or InsightAgent()).generate(payload).to_dict()

    return {
        "instrument": instrument,
        "study": study,
        "score": result.to_dict(),
        "sensitivity": sensitivity,
        "descriptives": descriptives,
        "inference": inference,
        "reliability": reliability,
        "quality": quality,
        "calibration": calibration.to_dict(),
        "charts": _chart_data(trials),
        "session": session_meta,
        "seed": session.get("seed"),
        "insight": insight,
        "llm_payload": payload,
        "interpretation_note": THRESHOLD_NOTE,
        "single_session_note": (
            "This is one participant on one occasion. The IAT's test-retest "
            "reliability is around r = .50, and Cummins & Hussey (2026, Behavior "
            "Research Methods) report that a D of exactly zero carries a 95% "
            "confidence interval of roughly -0.38 to +0.38, with only about half "
            "of participants scoring detectably differently from zero. The "
            "instrument is designed to be read at group level; a per-respondent "
            "verdict is not something it supports, which is why this report "
            "shows an interval rather than a label."
        ),
    }
