"""Cohort pipeline: ingestion, screening, vectorised scoring, group inference.

The tests worth reading first:

*   ``test_vectorised_d_matches_reference_scorer`` — the fast path scores every
    task in a panel and must agree with :func:`app.scoring.dscore.compute_d`
    to 1e-9, with and without trial trimming. Two implementations of one
    algorithm are a liability unless something pins them together.
*   ``test_planted_respondents_are_screened_out`` — every simulated fast,
    inaccurate and dropout respondent must be excluded, at the right stage.
*   ``test_planted_segment_effects_are_recovered`` /
    ``test_null_variable_stays_null`` — the analysis finds the effects that
    were built in, and does not find the one that was not.
*   ``test_lower_trim_cannot_rescue_a_fast_responder`` — the latency screen
    runs before trimming, so the slider cannot delete the evidence.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from app.analysis import to_trials
from app.cohort import stats as st
from app.cohort.analysis import analyse_cohort
from app.cohort.ingest import IngestError, ingest, to_csv_bytes
from app.cohort.scoring import CohortParams, prepare, score
from app.cohort.simulate import simulate_batch
from app.scoring.dscore import compute_d
from app.scoring.models import ErrorPenalty, LatencySource, ScoringConfig


@pytest.fixture(scope="module")
def panel():
    sim = simulate_batch(160, seed=11)
    prep = prepare(sim.trials, list(sim.meta["segments"]))
    return sim, prep


@pytest.fixture(scope="module")
def big_panel():
    sim = simulate_batch(520, seed=2026)
    prep = prepare(sim.trials, list(sim.meta["segments"]))
    return sim, prep


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [
    CohortParams(),
    CohortParams(lower_ms=400.0, upper_ms=2500.0),
])
def test_vectorised_d_matches_reference_scorer(panel, params):
    sim, prep = panel
    sc = score(prep, params)
    # The cohort screen replaces compute_d's own fast-responder rule, so it is
    # switched off here; everything else is the production configuration.
    cfg = ScoringConfig(
        error_penalty=ErrorPenalty.BUILT_IN, latency_source=LatencySource.TO_CORRECT,
        upper_cutoff_ms=params.upper_ms, lower_cutoff_ms=params.lower_ms or None,
        fast_trial_proportion_limit=1.0, min_trials_per_block=params.min_trials,
    )
    pid_idx = {p: i for i, p in enumerate(prep.pids)}
    checked = 0
    for (pid, attr), g in sim.trials.groupby(["participant_id", "attribute"], sort=False):
        i, j = pid_idx[pid], prep.attrs.index(attr)
        if not sc.included[i]:
            continue
        ref = compute_d(to_trials(g.rename(columns={"trial_index": "index"}).to_dict("records")), cfg)
        assert ref.d is not None
        assert sc.d[i, j] == pytest.approx(ref.d, abs=1e-9)
        assert sc.d_practice[i, j] == pytest.approx(ref.d_practice, abs=1e-9)
        assert sc.d_test[i, j] == pytest.approx(ref.d_test, abs=1e-9)
        checked += 1
    assert checked > 300


def test_planted_respondents_are_screened_out(panel):
    sim, prep = panel
    sc = score(prep, CohortParams())
    status = pd.Series(sc.status, index=prep.pids)
    kind = sim.truth.groupby("participant_id")["kind"].first()
    assert (status[kind[kind == "fast"].index] == "fast").all()
    assert (status[kind[kind == "inaccurate"].index] == "inaccurate").all()
    assert (status[kind[kind == "dropout"].index] == "incomplete").all()
    # Walking away for a single trial costs that trial, not the participant.
    assert (status[kind[kind == "distracted"].index] == "included").all()
    genuine = status[kind[kind == "genuine"].index]
    assert (genuine == "included").mean() > 0.97


def test_funnel_is_consistent(panel):
    _, prep = panel
    sc = score(prep, CohortParams())
    f = sc.funnel
    assert f[0]["n"] == prep.n_participants
    running = f[0]["n"]
    for stage in f[1:-1]:
        running -= stage["removed"]
        assert stage["n"] == running
    assert f[-1]["n"] == int(sc.included.sum())


def test_upper_cutoff_removes_trials_not_people(big_panel):
    sim, prep = big_panel
    sc = score(prep, CohortParams())
    kind = sim.truth.groupby("participant_id")["kind"].first()
    assert (kind == "distracted").any()
    assert sc.trimming["removed_upper"] > 0


def test_lower_trim_cannot_rescue_a_fast_responder(panel):
    _, prep = panel
    base = score(prep, CohortParams())
    trimmed = score(prep, CohortParams(lower_ms=400.0))
    assert ((base.status == "fast") == (trimmed.status == "fast")).all()


def test_stricter_accuracy_rule_never_admits_more(panel):
    _, prep = panel
    counts = [int(score(prep, CohortParams(min_accuracy=a)).included.sum())
              for a in (0.5, 0.75, 0.85, 0.9)]
    assert counts == sorted(counts, reverse=True)


def test_excluded_participants_get_no_d(panel):
    _, prep = panel
    sc = score(prep, CohortParams())
    assert np.isnan(sc.d[~sc.included]).all()
    assert np.isfinite(sc.d[sc.included]).all()


def test_params_are_clamped():
    p = CohortParams.from_query({"lower_ms": "-50", "upper_ms": "50", "min_accuracy": "7"})
    assert p.lower_ms == 0.0
    assert p.upper_ms >= p.lower_ms + 100
    assert p.min_accuracy == 1.0
    assert CohortParams.from_query({"fast_ms": "banana"}).fast_ms == 300.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_bh_matches_hand_computation():
    # Worked example: m = 5. Raw ranked adjustments p*m/rank are
    # .005 .025 .0333 .0375 .05, already monotone, so q is exactly that.
    p = [0.01, 0.001, 0.03, 0.02, 0.05]
    q = st.bh_fdr(p)
    assert q == pytest.approx([0.025, 0.005, 0.0375, 0.0333333, 0.05], abs=1e-6)


def test_bh_enforces_monotonicity_and_passes_none():
    q = st.bh_fdr([0.04, None, 0.041, 0.9])
    assert q[1] is None
    assert q[0] == pytest.approx(0.0615) and q[2] == pytest.approx(0.0615)
    assert q[3] == pytest.approx(0.9)


def test_permutation_and_t_agree_on_clean_data():
    rng = np.random.default_rng(1)
    X = rng.normal(0.25, 1.0, size=(200, 1))
    p_perm = st.signflip_p(X, st.rng_for("t"))[0]
    _, p_t, _ = st.ttest_one(X)
    assert abs(np.log10(p_perm) - np.log10(max(p_t[0], 1 / 5001))) < 0.6


def test_null_permutation_p_is_uniformish():
    rng = np.random.default_rng(5)
    ps = [st.perm_diff_p(rng.normal(size=(40, 1)), rng.normal(size=(40, 1)),
                         st.rng_for("null", i), n_perm=999)[0] for i in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10


def test_resampling_is_deterministic(big_panel):
    sim, prep = big_panel
    a = analyse_cohort(prep, sim.meta, segment_by="exposure", include_distributions=False)
    b = analyse_cohort(prep, sim.meta, segment_by="exposure", include_distributions=False)
    a.pop("compute_ms"), b.pop("compute_ms")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------
# The planted story
# --------------------------------------------------------------------------


def _comparison(report, attr):
    return next(c for c in report["comparisons"] if c["attribute"] == attr)


@pytest.mark.parametrize("method", ["permutation", "parametric"])
def test_planted_segment_effects_are_recovered(big_panel, method):
    sim, prep = big_panel
    loyalty = analyse_cohort(prep, sim.meta, segment_by="loyalty", include_distributions=False)
    c = _comparison(loyalty, "premium")
    assert c[method]["sig"] and c["diff"] > 0.1 and c["level_a"] == "High"

    exposure = analyse_cohort(prep, sim.meta, segment_by="exposure", include_distributions=False)
    c = _comparison(exposure, "trust")
    assert c[method]["sig"] and c["diff"] > 0.1 and c["level_a"] == "Exposed"
    assert not _comparison(exposure, "premium")[method]["sig"]

    age = analyse_cohort(prep, sim.meta, segment_by="age_band", include_distributions=False)
    young = [c for c in age["comparisons"] if c["attribute"] == "eco" and c["level_a"] == "18-34"]
    assert young and all(c[method]["sig"] and c["diff"] < 0 for c in young)

    overall = {a["key"]: a["overall"] for a in exposure["attributes"]}
    assert overall["premium"]["mean_d"] > 0 and overall["premium"][method]["sig"]
    assert overall["eco"]["mean_d"] < 0 and overall["eco"][method]["sig"]


@pytest.mark.parametrize("method", ["permutation", "parametric"])
def test_null_variable_stays_null(big_panel, method):
    sim, prep = big_panel
    r = analyse_cohort(prep, sim.meta, segment_by="device", include_distributions=False)
    assert not any(c[method]["sig"] for c in r["comparisons"])


def test_raw_rts_are_skewed_and_logs_much_less(big_panel):
    sim, prep = big_panel
    r = analyse_cohort(prep, sim.meta)
    for attr in r["distributions"].values():
        for pairing in ("congruent", "incongruent"):
            s = attr["shape"][pairing]
            assert s["skew"] > 1.0
            assert abs(s["log_skew"]) < s["skew"] / 2


def test_takeaways_never_give_direction_to_a_nonsignificant_result(big_panel):
    sim, prep = big_panel
    r = analyse_cohort(prep, sim.meta, segment_by="device",
                       params=CohortParams(min_accuracy=0.99), include_distributions=False)
    for attr in r["attributes"]:
        ta = next(t for t in r["takeaways"]["permutation"]
                  if t.get("attribute") == attr["key"] and t["kind"] == "association")
        if not attr["overall"]["permutation"]["sig"]:
            assert ta["headline"].startswith("No clear winner")


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def test_csv_round_trip_reproduces_the_analysis(panel):
    sim, prep = panel
    raw = to_csv_bytes(sim.trials, list(sim.meta["segments"]))
    res = ingest(raw, "panel.csv")
    assert res.report["rows_rejected"] == 0
    assert set(res.meta["segments"]) == set(sim.meta["segments"])
    prep2 = prepare(res.trials, list(res.meta["segments"]))
    a = score(prep, CohortParams())
    b = score(prep2, CohortParams())
    order = [list(prep2.pids).index(p) for p in prep.pids]
    np.testing.assert_allclose(a.d, b.d[order], atol=1e-9, equal_nan=True)


def test_ingest_reports_and_drops_bad_rows():
    sim = simulate_batch(12, seed=4)
    df = sim.trials.copy()
    df["latency_ms"] = df["latency_ms"].astype(object)
    df["correct"] = df["correct"].astype(object)
    df.loc[0, "latency_ms"] = "fast"
    df.loc[1, "correct"] = "maybe"
    df.loc[2, "pairing"] = "sideways"
    df = pd.concat([df, df.iloc[[10]]])            # a duplicated row
    res = ingest(df.to_csv(index=False).encode(), "dirty.csv")
    r = res.report
    assert r["rows_rejected"] == 3
    assert r["duplicates_removed"] == 1
    assert "latency is missing or not a number" in r["rejection_reasons"]
    assert r["rows_accepted"] == len(sim.trials) - 3


def test_ingest_maps_aliases_and_bins_continuous_variables():
    sim = simulate_batch(30, seed=8)
    df = sim.trials.rename(columns={"participant_id": "Respondent ID", "latency_ms": "RT"})
    ages = {p: 18 + i for i, p in enumerate(df["Respondent ID"].unique())}
    df["age"] = df["Respondent ID"].map(ages)
    df["free_text"] = np.arange(len(df)).astype(str)   # varies within participant
    res = ingest(df.to_csv(index=False).encode(), "x.csv")
    assert res.report["columns_renamed"]["RT"] == "latency_ms"
    assert "age" in res.meta["segments"] and len(res.meta["segments"]["age"]["levels"]) == 3
    assert "free_text" in res.report["skipped_columns"]
    assert "trial-level" in res.report["skipped_columns"]["free_text"]


def test_json_with_separate_participant_table():
    sim = simulate_batch(10, seed=9)
    segs = list(sim.meta["segments"])
    parts = sim.trials.groupby("participant_id")[segs].first().reset_index()
    body = {"participants": parts.to_dict("records"),
            "trials": sim.trials.drop(columns=segs).to_dict("records")}
    res = ingest(json.dumps(body, default=str).encode(), "p.json")
    assert set(segs) <= set(res.meta["segments"])
    assert res.report["participants"] == 10


def test_ingest_rejects_files_missing_required_columns():
    with pytest.raises(IngestError, match="latency_ms"):
        ingest(b"participant_id,correct,block_role,pairing\nP1,1,test,congruent\n", "x.csv")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_api_upload_then_analyse(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.cohort.api import build_router
    from app.storage import Store

    from fastapi import FastAPI
    app = FastAPI()
    router, _ = build_router(Store(str(tmp_path / "t.db")))
    app.include_router(router)
    c = TestClient(app)

    sample = c.get("/api/cohort/sample.csv?n=40&seed=3")
    assert sample.status_code == 200
    up = c.post("/api/cohort/batches?filename=s.csv&name=Wave%202", content=sample.content)
    assert up.status_code == 200, up.text
    bid = up.json()["batch"]["id"]
    assert up.json()["batch"]["name"] == "Wave 2"

    r = c.get(f"/api/cohort/batches/{bid}/analysis?segment_by=exposure&lower_ms=250")
    assert r.status_code == 200
    body = r.json()
    assert body["params"]["lower_ms"] == 250
    assert body["segment_by"] == "exposure"
    assert body["sample"]["recruited"] == 40

    bad = c.post("/api/cohort/batches?filename=x.csv", content=b"a,b\n1,2\n")
    assert bad.status_code == 422
    assert c.get("/api/cohort/batches/nope/analysis").status_code == 404
    listed = c.get("/api/cohort/batches").json()["batches"]
    assert listed[0]["id"] == "demo" and any(b["id"] == bid for b in listed)
