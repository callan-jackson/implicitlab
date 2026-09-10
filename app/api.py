"""HTTP API.

Deliberately small. Four things happen over the wire: ask for a design, send
back trials, read a report, export the raw data. Everything else is a read of
something already computed.
"""

from __future__ import annotations

import csv
import io
import json
import secrets

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .analysis import analyse
from .config import get_settings
from .design import DIMENSIONS, PRESETS, STUDIES, build_session, custom_study
from .llm.client import InsightAgent
from .scoring.dscore import THRESHOLD_NOTE
from .storage import Store

router = APIRouter(prefix="/api")
settings = get_settings()
store = Store(settings.database_path)
agent = InsightAgent(settings)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class SessionRequest(BaseModel):
    study_id: str = Field(default="gin-premium")
    preset: str = Field(default="standard")
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    include_calibration: bool = True
    custom_brand_a: str | None = None
    custom_brand_b: str | None = None
    custom_dimension: str = "premium"


class TrialRecord(BaseModel):
    index: int
    block_index: int
    block_role: str | None = None
    pairing: str | None = None
    stimulus: str = ""
    stimulus_category: str = ""
    correct_key: str = ""
    response_key: str = ""
    latency_ms: float
    latency_to_correct_ms: float | None = None
    correct: bool
    timed_out: bool = False
    n_corrections: int = 0
    onset_uncertainty_ms: float = 0.0
    dispatch_delay_ms: float | None = None
    used_event_timestamp: bool | None = None
    focus_lost: bool = False
    onset_degraded: bool = False
    word_count: int = 1


class ResultsRequest(BaseModel):
    token: str
    trials: list[TrialRecord]
    client_meta: dict = Field(default_factory=dict)
    consent: bool = False
    run_llm: bool = True


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": settings.version}


@router.get("/meta")
def meta() -> dict:
    """What this deployment is and what it can do — read by the front end."""
    return {
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "llm": {
            "configured": settings.llm_configured,
            "provider": "Azure OpenAI" if settings.llm_configured else None,
            "deployment": settings.azure_openai_deployment if settings.llm_configured else None,
            "role": (
                "The model receives only computed statistics and writes prose. It "
                "never sees a raw latency and is never the source of a number; every "
                "figure it emits is checked against the statistics before the summary "
                "is published."
            ),
        },
        "presets": {k: {"label": v["label"], "minutes": v["minutes"], "note": v["note"]}
                    for k, v in PRESETS.items()},
        "dimensions": {k: v.name for k, v in DIMENSIONS.items()},
        "interpretation_note": THRESHOLD_NOTE,
        "corpus": store.stats(),
    }


@router.get("/studies")
def studies() -> dict:
    return {
        "studies": [
            {
                "id": s.id, "name": s.name, "sector": s.sector, "blurb": s.blurb,
                "instrument": s.instrument, "fictitious": s.fictitious,
                "dimension": s.dimension.name,
                "target_a": s.target_a.label,
                "target_b": s.target_b.label if s.target_b else None,
            }
            for s in STUDIES.values()
        ]
    }


# --------------------------------------------------------------------------
# Running a session
# --------------------------------------------------------------------------


@router.post("/sessions")
def create_session(req: SessionRequest) -> dict:
    """Generate a seeded, counterbalanced design and hand back a token."""
    if req.custom_brand_a:
        study = custom_study(
            req.custom_brand_a, req.custom_brand_b, dimension_key=req.custom_dimension
        )
    else:
        study = STUDIES.get(req.study_id)
        if study is None:
            raise HTTPException(404, f"Unknown study '{req.study_id}'")

    if req.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset '{req.preset}'")

    design = build_session(
        study, preset=req.preset, seed=req.seed,
        include_calibration=req.include_calibration,
    )
    token = store.put_design(design)
    return {"token": token, "design": design}


@router.post("/results")
def submit_results(req: ResultsRequest) -> dict:
    """Score a completed session, persist it, and return the full report."""
    design = store.get_design(req.token)
    if design is None:
        raise HTTPException(404, "Unknown or expired session token.")

    if not req.trials:
        raise HTTPException(400, "No trials submitted.")

    records = [t.model_dump() for t in req.trials]

    # The design is the server's copy. Reconciling against it means a client
    # cannot report a trial as belonging to a different condition than the one
    # it was actually shown in — which, under counterbalancing, is the one thing
    # that would silently flip the sign of D.
    by_index = {t["index"]: t for t in design["trials"]}
    reconciled = []
    mismatches = 0
    for r in records:
        truth = by_index.get(r["index"])
        if truth is None:
            mismatches += 1
            continue
        r["block_index"] = truth["block_index"]
        r["block_role"] = truth["block_role"]
        r["pairing"] = truth["pairing"]
        r["stimulus"] = truth["stimulus"]
        r["stimulus_category"] = truth["stimulus_category"]
        r["correct_key"] = truth["correct_key"]
        r["word_count"] = truth.get("word_count", 1)
        r["correct"] = (r.get("response_key") or "").upper() == truth["correct_key"]
        reconciled.append(r)

    client_meta = dict(req.client_meta)
    client_meta["consent_given"] = bool(req.consent)
    if mismatches:
        client_meta["unreconciled_trials"] = mismatches

    dispatches = [r["dispatch_delay_ms"] for r in reconciled if r.get("dispatch_delay_ms") is not None]
    if dispatches:
        dispatches.sort()
        client_meta["median_dispatch_delay_ms"] = dispatches[len(dispatches) // 2]

    report = analyse(
        session=design, records=reconciled, client_meta=client_meta,
        run_llm=req.run_llm, agent=agent,
        bootstrap_resamples=settings.bootstrap_resamples,
    )

    session_id = None
    if settings.retain_sessions:
        session_id = store.save(
            session=design, trials=reconciled, client_meta=client_meta,
            d=report["score"]["d"], excluded=report["score"]["excluded"],
            app_version=settings.version,
        )
        store.consume_design(req.token)

    report["session_id"] = session_id
    report["norms"] = _norms(design["study"]["id"], report["score"]["d"])
    return report


def _norms(study_id: str, d: float | None) -> dict:
    """Where this D falls among the sessions already collected for this study."""
    norms = store.norms(study_id)
    values = sorted(norms["d_values"])
    percentile = None
    if d is not None and len(values) >= 5:
        below = sum(1 for v in values if v < d)
        percentile = round(100 * below / len(values), 1)
    return {
        "study_id": study_id,
        "n": norms["n"],
        "d_values": [round(v, 4) for v in values],
        "percentile": percentile,
        "note": (
            "The distribution of D across every usable session of this study run "
            "on this deployment. It is a convenience sample of whoever opened the "
            "link, not a representative panel, and it should not be read as a "
            "population norm."
        ),
    }


# --------------------------------------------------------------------------
# Reading and exporting
# --------------------------------------------------------------------------


@router.get("/sessions/{session_id}")
def get_session(session_id: str, run_llm: bool = Query(default=False)) -> dict:
    stored = store.get(session_id)
    if stored is None:
        raise HTTPException(404, "Unknown session.")
    records = []
    for t in stored["trials"]:
        records.append({
            "index": t["idx"], "block_index": t["block_index"],
            "block_role": t["block_role"], "pairing": t["pairing"],
            "stimulus": t["stimulus"], "stimulus_category": t["stimulus_category"],
            "correct_key": t["correct_key"], "response_key": t["response_key"],
            "latency_ms": t["latency_ms"],
            "latency_to_correct_ms": t["latency_to_correct_ms"],
            "correct": bool(t["correct"]), "timed_out": bool(t["timed_out"]),
            "n_corrections": t["n_corrections"],
            "onset_uncertainty_ms": t["onset_uncertainty_ms"],
            "dispatch_delay_ms": t["dispatch_delay_ms"],
            "focus_lost": bool(t["focus_lost"]),
        })
    report = analyse(
        session=stored["session"], records=records,
        client_meta=stored["client_meta"], run_llm=run_llm, agent=agent,
        bootstrap_resamples=settings.bootstrap_resamples,
    )
    report["session_id"] = session_id
    report["created_at"] = stored["created_at"]
    report["norms"] = _norms(stored["session"].get("study", {}).get("id", "unknown"),
                             report["score"]["d"])
    return report


CSV_COLUMNS = [
    "session_id", "trial_index", "block_index", "block_role", "pairing",
    "stimulus", "stimulus_category", "correct_key", "response_key",
    "latency_ms", "latency_to_correct_ms", "correct", "timed_out",
    "n_corrections", "onset_uncertainty_ms", "dispatch_delay_ms", "focus_lost",
]


@router.get("/sessions/{session_id}/export.csv")
def export_csv(session_id: str) -> StreamingResponse:
    """One row per trial. The whole point of a research tool is that the raw
    data leaves it in a form somebody else can re-analyse."""
    stored = store.get(session_id)
    if stored is None:
        raise HTTPException(404, "Unknown session.")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for t in stored["trials"]:
        w.writerow([
            session_id, t["idx"], t["block_index"], t["block_role"] or "",
            t["pairing"] or "", t["stimulus"], t["stimulus_category"],
            t["correct_key"], t["response_key"] or "", t["latency_ms"],
            "" if t["latency_to_correct_ms"] is None else t["latency_to_correct_ms"],
            int(t["correct"]), int(t["timed_out"]), t["n_corrections"],
            t["onset_uncertainty_ms"],
            "" if t["dispatch_delay_ms"] is None else t["dispatch_delay_ms"],
            int(t["focus_lost"]),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="implicitlab_{session_id}.csv"'},
    )


@router.get("/sessions/{session_id}/export.json", response_class=PlainTextResponse)
def export_json(session_id: str) -> PlainTextResponse:
    """The complete record: design, counterbalance, seed, trials, client metadata."""
    stored = store.get(session_id)
    if stored is None:
        raise HTTPException(404, "Unknown session.")
    body = json.dumps({
        "format": "implicitlab/session/v1",
        "app_version": stored["app_version"],
        "session_id": session_id,
        "created_at": stored["created_at"],
        "design": stored["session"],
        "client_meta": stored["client_meta"],
        "trials": stored["trials"],
    }, indent=2)
    return PlainTextResponse(
        body, media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="implicitlab_{session_id}.json"'},
    )


@router.get("/studies/{study_id}/norms")
def study_norms(study_id: str) -> dict:
    return _norms(study_id, None)


@router.get("/sessions")
def recent_sessions(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"sessions": store.recent(limit)}


# --------------------------------------------------------------------------
# Worked example
# --------------------------------------------------------------------------


class DemoRequest(BaseModel):
    study_id: str = "gin-premium"
    preset: str = "standard"
    true_d: float = Field(default=0.45, ge=-2.0, le=2.0)
    seed: int | None = None
    careless: bool = False


@router.post("/demo")
def demo(req: DemoRequest) -> dict:
    """Run a synthetic respondent through the identical pipeline.

    This exists because a full IAT is 190 trials and nobody evaluating the tool
    wants to sit through them before seeing what it produces. The synthetic
    session is generated from an ex-Gaussian latency model with a known effect
    built in, then scored by exactly the same code path a real session uses —
    no shortcuts, no pre-baked numbers. It is marked as simulated in the
    response, on the dashboard, and it is never written to the session store,
    so it cannot contaminate the comparison distribution.
    """
    from .simulate import simulate_session

    study = STUDIES.get(req.study_id)
    if study is None:
        raise HTTPException(404, f"Unknown study '{req.study_id}'")
    if req.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset '{req.preset}'")

    seed = req.seed if req.seed is not None else secrets.randbelow(2**31 - 1)
    design, records, client_meta = simulate_session(
        study=study, preset=req.preset, seed=seed,
        true_d=req.true_d, careless=req.careless,
    )
    report = analyse(
        session=design, records=records, client_meta=client_meta,
        run_llm=True, agent=agent, bootstrap_resamples=settings.bootstrap_resamples,
    )
    report["session_id"] = None
    report["simulated"] = True
    report["simulated_true_d"] = req.true_d
    report["norms"] = _norms(design["study"]["id"], report["score"]["d"])
    return report
