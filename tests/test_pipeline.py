"""End-to-end tests: simulated respondents through the full analysis.

The recovery test is the one that matters. Anything can produce a number; a
scoring implementation is only useful if the number it produces tracks the
effect that was actually in the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis import analyse, calibrate, to_trials
from app.design import STUDIES
from app.simulate import simulate_session

STUDY = STUDIES["gin-premium"]


def run(
    true_d: float, seed: int, preset: str = "full", *, resamples: int = 400, **kw
) -> dict:
    design, records, meta = simulate_session(
        study=STUDY, preset=preset, seed=seed, true_d=true_d, **kw
    )
    return analyse(
        session=design, records=records, client_meta=meta, run_llm=False,
        bootstrap_resamples=resamples,
    )


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


def test_recovers_a_planted_effect_on_average():
    """Across many simulated respondents, mean D should track the true D.

    Individual sessions are noisy — that is the honest finding about this
    instrument — so recovery is asserted on the mean of 30 respondents, not on
    any one of them.
    """
    for true_d in (0.0, 0.5, -0.5):
        ds = []
        for seed in range(30):
            r = run(true_d, seed=seed + int(abs(true_d) * 1000), resamples=0)
            if r["score"]["d"] is not None:
                ds.append(r["score"]["d"])
        assert len(ds) >= 25, "too many sessions were excluded to judge recovery"
        assert float(np.mean(ds)) == pytest.approx(true_d, abs=0.18)


def test_a_strong_effect_is_detected_and_a_null_is_not():
    strong = run(1.0, seed=42, resamples=1500)
    assert strong["score"]["d"] > 0.5
    assert strong["inference"]["p_permutation"] < 0.05
    assert strong["inference"]["ci_low"] > 0

    null = run(0.0, seed=43, resamples=1500)
    ci = (null["inference"]["ci_low"], null["inference"]["ci_high"])
    assert ci[0] < 0 < ci[1], "a null effect should give an interval spanning zero"


def test_direction_is_reported_correctly():
    pos = run(0.8, seed=7, resamples=0)
    neg = run(-0.8, seed=7, resamples=0)
    assert pos["score"]["direction"] == "a"
    assert neg["score"]["direction"] == "b"
    assert STUDY.target_a.label in pos["score"]["interpretation"]
    assert STUDY.target_b.label in neg["score"]["interpretation"]


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------


def test_a_careless_respondent_is_excluded():
    """Button-mashing must produce no score at all, not a small one."""
    r = run(0.0, seed=99, careless=True, resamples=0)
    assert r["score"]["excluded"] is True
    assert r["score"]["d"] is None
    assert r["quality"]["overall"] == "fail"
    assert r["quality"]["usable"] is False
    # And the report must say why rather than showing an empty panel.
    assert r["score"]["exclusion_reasons"]


def test_a_normal_respondent_passes_quality():
    r = run(0.4, seed=3, resamples=0)
    assert r["quality"]["usable"] is True
    assert r["score"]["excluded"] is False
    keys = {c["key"] for c in r["quality"]["checks"]}
    assert {"too_fast", "error_rate", "cell_counts", "refresh"} <= keys


# --------------------------------------------------------------------------
# Inference and reliability
# --------------------------------------------------------------------------


def test_confidence_interval_brackets_the_point_estimate():
    r = run(0.5, seed=11)
    inf, d = r["inference"], r["score"]["d"]
    assert inf["ci_low"] <= d <= inf["ci_high"]
    assert 0 < inf["p_permutation"] <= 1
    assert inf["n_resamples"] > 0


def test_more_trials_give_a_tighter_interval():
    """Reducing trial count must widen the interval, or the CI is decorative."""
    full = run(0.5, seed=21, preset="full", resamples=1200)["inference"]
    express = run(0.5, seed=21, preset="express", resamples=1200)["inference"]
    width_full = full["ci_high"] - full["ci_low"]
    width_express = express["ci_high"] - express["ci_low"]
    assert width_express > width_full


def test_split_half_diagnostic_is_produced():
    r = run(0.6, seed=31, resamples=0)
    rel = r["reliability"]
    assert rel["d_odd_trials"] is not None
    assert rel["d_even_trials"] is not None
    assert rel["absolute_difference"] >= 0


def test_sensitivity_analyses_are_reported():
    r = run(0.5, seed=5, resamples=0)
    labels = {s["label"] for s in r["sensitivity"]}
    assert any("600" in l for l in labels)
    assert any("Richetin" in l for l in labels)
    for s in r["sensitivity"]:
        assert s["d"] is not None


# --------------------------------------------------------------------------
# Calibration covariates
# --------------------------------------------------------------------------


def test_calibration_recovers_a_reading_slope():
    """The simulator builds in ~55 ms per word; the estimator should find it."""
    _, records, _ = simulate_session(study=STUDY, preset="full", seed=4, true_d=0.3)
    cal = calibrate(records)
    assert cal.motor_median_ms is not None
    assert 150 < cal.motor_median_ms < 500
    assert cal.reading_slope_ms_per_word is not None
    assert 20 < cal.reading_slope_ms_per_word < 120


def test_calibration_trials_are_not_scored_into_d():
    design, records, meta = simulate_session(study=STUDY, preset="full", seed=8, true_d=0.4)
    trials = to_trials(records)
    scored = [t for t in trials if t.block_role]
    assert all(not t.stimulus_category.startswith("motor") for t in scored)
    assert all(t.stimulus_category != "reading" for t in scored)


# --------------------------------------------------------------------------
# Charts and payload
# --------------------------------------------------------------------------


def test_chart_data_is_present_and_binned_consistently():
    r = run(0.5, seed=13, resamples=0)
    charts = r["charts"]
    con = charts["histograms"]["congruent"]
    inc = charts["histograms"]["incongruent"]
    assert con["edges"] == inc["edges"], "conditions must share bin edges to be comparable"
    assert sum(con["counts"]) > 0 and sum(inc["counts"]) > 0
    assert len(charts["series"]) > 0
    assert charts["density"]["congruent"]["x"]


def test_llm_payload_contains_no_raw_trial_data():
    """The model must not be able to see a latency it could recompute from."""
    r = run(0.5, seed=17, resamples=300)
    payload = r["llm_payload"]
    assert "trials" not in payload
    assert "charts" not in payload
    flat = str(payload)
    assert "latency_ms" not in flat
    # It should still have everything it needs to write the report.
    assert payload["score"]["d"] is not None
    assert payload["inference"]["p_permutation"] is not None
    assert payload["quality"]["overall"]


def test_deterministic_insight_runs_without_a_model_configured():
    design, records, meta = simulate_session(study=STUDY, preset="standard", seed=19, true_d=0.5)
    r = analyse(
        session=design, records=records, client_meta=meta, run_llm=True,
        bootstrap_resamples=300,
    )
    insight = r["insight"]
    assert insight is not None
    assert insight["summary_markdown"]
    assert insight["verification"]["verified"] is True
    assert "prompt_shown" in insight
    assert insight["prompt_shown"]["system"]


# --------------------------------------------------------------------------
# SC-IAT
# --------------------------------------------------------------------------


def test_sciat_end_to_end():
    study = STUDIES["aurelia-sciat"]
    design, records, meta = simulate_session(
        study=study, preset="full", seed=23, true_d=0.5
    )
    r = analyse(
        session=design, records=records, client_meta=meta, run_llm=False,
        bootstrap_resamples=300,
    )
    assert r["instrument"] == "sciat"
    assert r["score"]["d"] is not None
    # A single-category test yields one quotient, not two averaged.
    assert r["score"]["d_practice"] is None
    assert r["score"]["d"] == pytest.approx(r["score"]["d_test"])
    assert not r["sensitivity"], "IAT-only sensitivity analyses must not run on an SC-IAT"


def test_sciat_only_scores_critical_blocks():
    study = STUDIES["aurelia-sciat"]
    design, records, meta = simulate_session(study=study, preset="full", seed=24, true_d=0.4)
    scored_blocks = {b["index"] for b in design["blocks"] if b["role"] == "sciat"}
    trials = to_trials(records)
    used = {t.block_index for t in trials if t.block_role == "sciat"}
    assert used == scored_blocks
    assert len(used) == 2


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------


def test_malformed_rows_are_dropped_not_fatal():
    design, records, meta = simulate_session(study=STUDY, preset="standard", seed=27, true_d=0.4)
    records.append({"index": "not-an-int", "block_index": 3, "latency_ms": "x", "correct": True})
    r = analyse(
        session=design, records=records, client_meta=meta, run_llm=False,
        bootstrap_resamples=0,
    )
    assert r["score"]["d"] is not None
    assert any(c["key"] == "malformed" for c in r["quality"]["checks"])


def test_empty_submission_does_not_crash():
    design, _, meta = simulate_session(study=STUDY, preset="express", seed=29, true_d=0.0)
    r = analyse(
        session=design, records=[], client_meta=meta, run_llm=False, bootstrap_resamples=0
    )
    assert r["score"]["d"] is None
    assert r["score"]["excluded"] is True
