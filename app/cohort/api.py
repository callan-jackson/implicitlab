"""HTTP surface for cohort analysis.

Every analysis endpoint takes the same query parameters, and the export
endpoints take them too, so the file a research manager downloads is built
from exactly the state of the dashboard they were looking at. The parameters
are also written into the workbook and onto the last slide: a deliverable
that cannot say which exclusion rules produced it cannot be defended.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ..storage import Store
from .analysis import METHODS, analyse_cohort, clean, participant_table
from .ingest import IngestError, ingest, to_csv_bytes
from .registry import DEMO_ID, Registry
from .scoring import CohortParams
from .simulate import simulate_batch

MAX_UPLOAD_BYTES = 60 * 1024 * 1024

PARAM_KEYS = ("lower_ms", "upper_ms", "fast_ms", "fast_limit", "min_accuracy", "min_trials")


def build_router(store: Store) -> tuple[APIRouter, Registry]:
    router = APIRouter(prefix="/api/cohort", tags=["cohort"])
    registry = Registry(store)

    def batch_or_404(batch_id: str):
        b = registry.get(batch_id)
        if b is None:
            raise HTTPException(404, f"Unknown batch '{batch_id}'.")
        return b

    def params_from(request: Request) -> CohortParams:
        return CohortParams.from_query({k: v for k, v in request.query_params.items()
                                        if k in PARAM_KEYS})

    def method_from(method: str) -> str:
        if method not in METHODS:
            raise HTTPException(400, f"method must be one of {', '.join(METHODS)}")
        return method

    def filename(batch, ext: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", batch.meta.get("name", "batch")).strip("_")[:48]
        return f"{stem}_{datetime.now(timezone.utc):%Y%m%d}.{ext}"

    # ---------------------------------------------------------------- batches

    @router.get("/batches")
    def list_batches() -> dict:
        return {"batches": registry.list()}

    @router.get("/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict:
        b = batch_or_404(batch_id)
        return clean({"batch": b.summary(), "ingest_report": b.report})

    @router.post("/batches")
    async def upload_batch(
        request: Request,
        filename_: str = Query(default="upload.csv", alias="filename"),
        name: str | None = Query(default=None, max_length=80),
    ) -> dict:
        """Ingest a raw CSV or JSON dump sent as the request body.

        The body is the file itself (``Content-Type`` is ignored), which keeps
        the endpoint usable from ``curl --data-binary @panel.csv`` as well as
        the browser without a multipart dependency.
        """
        raw = await request.body()
        if not raw:
            raise HTTPException(400, "Empty upload.")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
        try:
            result = ingest(raw, filename_, name=name)
        except IngestError as exc:
            raise HTTPException(422, str(exc)) from exc
        b = registry.add(result)
        return clean({"batch": b.summary(), "ingest_report": b.report})

    @router.get("/sample.csv")
    def sample_csv(n: int = Query(default=150, ge=20, le=1000),
                   seed: int = Query(default=7, ge=0, le=2**31 - 1)) -> Response:
        """A synthetic raw dump in the canonical upload format.

        Download it, upload it, and the ingestion path is demonstrated end to
        end on a real file rather than on data that never left the server.
        """
        sim = simulate_batch(n, seed=seed)
        body = to_csv_bytes(sim.trials, list(sim.meta["segments"]))
        return Response(body, media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="implicitlab_panel_n{n}_seed{seed}.csv"'
        })

    @router.get("/batches/{batch_id}/raw.csv")
    def raw_csv(batch_id: str) -> Response:
        b = batch_or_404(batch_id)
        body = to_csv_bytes(b.trials, list(b.meta.get("segments", {})))
        return Response(body, media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="{filename(b, "csv")}"'
        })

    # --------------------------------------------------------------- analysis

    @router.get("/batches/{batch_id}/analysis")
    def analysis(request: Request, batch_id: str,
                 segment_by: str | None = Query(default=None)) -> dict:
        b = batch_or_404(batch_id)
        report = analyse_cohort(b.prep, b.meta, params_from(request), segment_by=segment_by)
        report["batch"]["id"] = b.id
        return report

    @router.get("/batches/{batch_id}/participants")
    def participants(request: Request, batch_id: str) -> dict:
        b = batch_or_404(batch_id)
        table, _ = participant_table(b.prep, b.meta, params_from(request))
        return clean({"columns": list(table.columns),
                      "rows": table.astype(object).where(table.notna(), None).values.tolist()})

    # ---------------------------------------------------------------- exports

    @router.get("/batches/{batch_id}/export.xlsx")
    def export_xlsx(request: Request, batch_id: str,
                    segment_by: str | None = Query(default=None),
                    method: str = Query(default="permutation")) -> Response:
        from .export_xlsx import build_workbook

        b = batch_or_404(batch_id)
        body = build_workbook(b, params_from(request), segment_by=segment_by,
                              method=method_from(method))
        return Response(body, headers={
            "Content-Disposition": f'attachment; filename="{filename(b, "xlsx")}"'
        }, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @router.get("/batches/{batch_id}/export.pptx")
    def export_pptx(request: Request, batch_id: str,
                    segment_by: str | None = Query(default=None),
                    method: str = Query(default="permutation")) -> Response:
        from .export_pptx import build_deck

        b = batch_or_404(batch_id)
        body = build_deck(b, params_from(request), segment_by=segment_by,
                          method=method_from(method))
        return Response(body, headers={
            "Content-Disposition": f'attachment; filename="{filename(b, "pptx")}"'
        }, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

    return router, registry


__all__ = ["build_router", "DEMO_ID"]
