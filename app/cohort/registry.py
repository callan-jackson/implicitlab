"""Where batches live between requests.

Two sources: the built-in demonstration panel, generated deterministically
from a seed on first use (so a fresh container on a free host has it without a
database), and operator uploads, persisted in SQLite. Either way the expensive
part — parsing and reducing to integer-coded arrays — happens once per process
and is cached, so moving a slider re-runs the statistics and nothing else.
"""

from __future__ import annotations

import gzip
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..storage import Store
from .ingest import IngestResult, to_csv_bytes
from .scoring import Prepared, prepare
from .simulate import simulate_batch

DEMO_ID = "demo"
DEMO_SEED = 2026
DEMO_N = 520
CACHE_SIZE = 4


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


class Registry:
    def __init__(self, store: Store):
        self.store = store
        self._cache: OrderedDict[str, Batch] = OrderedDict()
        self._lock = threading.Lock()

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
            sim = simulate_batch(DEMO_N, seed=DEMO_SEED)
            trials = _compact(sim.trials)
            prep = prepare(trials, list(sim.meta["segments"]))
            meta = {**sim.meta, "id": DEMO_ID}
            report = {
                "rows_read": len(trials), "rows_accepted": len(trials), "rows_rejected": 0,
                "rejection_reasons": {}, "duplicates_removed": 0,
                "participants": prep.n_participants, "attributes": prep.attrs,
                "segment_variables": list(sim.meta["segments"]), "skipped_columns": {},
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
