"""Where batches live between requests.

Two sources: the built-in demonstration panel, generated deterministically
from a seed (baked into the container image at build time, or simulated on
first use if no bake is present), and operator uploads, persisted in SQLite. Either way the expensive
part — parsing and reducing to integer-coded arrays — happens once per process
and is cached, so moving a slider re-runs the statistics and nothing else.
"""

from __future__ import annotations

import gzip
import io
import os
import pickle
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..storage import Store
from .ingest import IngestResult, to_csv_bytes
from .scoring import Prepared, prepare
from .simulate import simulate_batch

DEMO_ID = "demo"
DEMO_SEED = 2026
DEMO_N = 520
CACHE_SIZE = 4

#: Where a pre-built demo panel is baked into the container image. Simulating
#: 175k trials is ~1 s on a real core and over a minute on a free-tier CPU
#: slice; loading the baked copy is a fraction of a second on either.
BAKED_DEMO = Path(os.environ.get("IMPLICITLAB_BAKED_DEMO", "cache/demo_panel.pkl"))


def _demo_key() -> tuple:
    """Identifies the demo panel a baked file was built from — including the
    simulator's own source, so editing the generator invalidates a stale bake
    rather than silently serving yesterday's panel."""
    src = Path(__file__).with_name("simulate.py").read_bytes()
    return (DEMO_N, DEMO_SEED, zlib.crc32(src))


@dataclass(slots=True)
class Batch:
    id: str
    meta: dict
    report: dict
    trials: pd.DataFrame
    prep: Prepared
    created_at: str

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.meta.get("name"),
            "source": self.meta.get("source"),
            "created_at": self.created_at,
            "n_participants": int(self.prep.n_participants),
            "n_rows": int(len(self.trials)),
            "attributes": self.meta.get("attributes", {}),
            "segments": self.meta.get("segments", {}),
            "note": self.meta.get("note"),
        }


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Low-cardinality text columns to categoricals: ~170 MB → ~15 MB for the demo."""
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "string":
            if df[c].nunique(dropna=False) < max(64, len(df) // 20):
                df[c] = df[c].astype("category")
    return df


def _load_baked():
    """The baked demo panel, if present and built from the current parameters."""
    try:
        with BAKED_DEMO.open("rb") as fh:
            blob = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    if blob.get("key") != _demo_key():
        return None
    return blob["trials"], blob["meta"]


def bake_demo(path: Path = BAKED_DEMO) -> None:
    """Build the demo panel once and write it to ``path`` (run at image build)."""
    sim = simulate_batch(DEMO_N, seed=DEMO_SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"key": _demo_key(), "trials": _compact(sim.trials),
                     "meta": sim.meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)


class Registry:
    def __init__(self, store: Store):
        self.store = store
        self._cache: OrderedDict[str, Batch] = OrderedDict()
        self._lock = threading.Lock()
        self.prewarm = lambda: self.get(DEMO_ID)

    def _remember(self, batch: Batch) -> Batch:
        with self._lock:
            self._cache[batch.id] = batch
            self._cache.move_to_end(batch.id)
            while len(self._cache) > CACHE_SIZE:
                oldest = next(iter(self._cache))
                if oldest == DEMO_ID:  # the demo is pinned
                    self._cache.move_to_end(oldest)
                    oldest = next(iter(self._cache))
                self._cache.pop(oldest)
        return batch

    def get(self, batch_id: str) -> Batch | None:
        with self._lock:
            hit = self._cache.get(batch_id)
            if hit is not None:
                self._cache.move_to_end(batch_id)
                return hit
        if batch_id == DEMO_ID:
            return self._build_demo()
        row = self.store.get_batch(batch_id)
        if row is None:
            return None
        trials = pd.read_csv(io.BytesIO(gzip.decompress(row["data"])),
                             keep_default_na=False, na_values=[""])
        for c in row["meta"].get("segments", {}):
            trials[c] = trials[c].astype(str)
        trials["block_role"] = trials["block_role"].astype(str)
        trials["pairing"] = trials["pairing"].astype(str)
        trials = _compact(trials)
        prep = prepare(trials, list(row["meta"].get("segments", {})))
        return self._remember(Batch(batch_id, row["meta"], row["report"], trials, prep,
                                    row["created_at"]))

    def _build_demo(self) -> Batch:
        with self._lock:
            if DEMO_ID in self._cache:
                return self._cache[DEMO_ID]
            baked = _load_baked()
            if baked is not None:
                trials, sim_meta = baked
            else:
                sim = simulate_batch(DEMO_N, seed=DEMO_SEED)
                trials, sim_meta = _compact(sim.trials), sim.meta
            prep = prepare(trials, list(sim_meta["segments"]))
            meta = {**sim_meta, "id": DEMO_ID}
            report = {
                "rows_read": len(trials), "rows_accepted": len(trials), "rows_rejected": 0,
                "rejection_reasons": {}, "duplicates_removed": 0,
                "participants": prep.n_participants, "attributes": prep.attrs,
                "segment_variables": list(sim_meta["segments"]), "skipped_columns": {},
                "columns_renamed": {}, "warnings": [],
            }
            batch = Batch(DEMO_ID, meta, report, trials, prep,
                          datetime.now(timezone.utc).isoformat())
            self._cache[DEMO_ID] = batch
            return batch

    def add(self, result: IngestResult) -> Batch:
        segs = list(result.meta.get("segments", {}))
        blob = gzip.compress(to_csv_bytes(result.trials, segs), compresslevel=6)
        bid = self.store.put_batch(
            name=result.meta["name"], source=result.meta["source"],
            n_participants=result.meta["n_participants"], n_rows=len(result.trials),
            meta=result.meta, report=result.report, data=blob,
        )
        meta = {**result.meta, "id": bid}
        trials = _compact(result.trials.copy())
        prep = prepare(trials, segs)
        return self._remember(Batch(bid, meta, result.report, trials, prep,
                                    datetime.now(timezone.utc).isoformat()))

    def list(self) -> list[dict]:
        demo = self.get(DEMO_ID)
        rows = [demo.summary() | {"built_in": True}]
        rows += [{**r, "built_in": False} for r in self.store.list_batches()]
        return rows


if __name__ == "__main__":  # python -m app.cohort.registry  → bake the demo panel
    bake_demo()
    print(f"baked demo panel to {BAKED_DEMO}")
