"""FastAPI application: API plus the static front end, from one process.

Serving the page and the API from the same origin removes CORS from the picture
entirely, which for a demo is one fewer thing that can be broken by a proxy on
someone else's network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hashlib
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="ImplicitLab",
    version=settings.version,
    description=(
        "A browser-based implicit association testing platform: millisecond "
        "response-latency capture, the Greenwald (2003) D-score with its "
        "data-quality rules, distribution-free resampling inference, and an "
        "LLM reporting layer that is kept strictly downstream of the statistics."
    ),
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.include_router(router)


# The HTML shell must never be cached: it is what points at the versioned asset
# URLs, so a stale copy of it pins every other file to the old build.
NO_STORE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}

_ASSET_REF = re.compile(r'(src|href)="(/static/[^"?]+)"')


def _build_stamp() -> str:
    """A short digest over every static asset.

    Appended to each asset URL in the served HTML so that a deploy invalidates
    the browser cache for exactly the files that changed. Without it a browser
    happily keeps a month-old bundle and the deployment appears not to have
    happened — which is a confusing failure to debug precisely because nothing
    is broken, just old.
    """
    h = hashlib.sha256()
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(str(path.stat().st_mtime_ns).encode())
    return h.hexdigest()[:10]


BUILD = _build_stamp()


def _render(name: str) -> HTMLResponse:
    html = (STATIC_DIR / name).read_text(encoding="utf-8")
    html = _ASSET_REF.sub(rf'\1="\2?v={BUILD}"', html)
    return HTMLResponse(html, headers=NO_STORE)


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return _render("index.html")


@app.get("/method", include_in_schema=False)
def method() -> HTMLResponse:
    return _render("method.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "assets" / "favicon.svg")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
