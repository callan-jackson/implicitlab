"""Persistence: SQLite, one file, no server.

Sessions are stored so that (a) a result can be re-opened by URL, (b) the raw
trial data can be exported for re-analysis, and (c) an individual D can be
placed against the distribution of everyone else who has taken the same study.
That last one matters: a single D-score means very little in isolation, and the
most honest thing a demo can show alongside it is where it falls among the
scores already collected.

The schema keeps raw trials as submitted. Derived values are recomputed on
read rather than stored, so improving the scoring code improves every historical
session instead of leaving a mix of old and new numbers in one table.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    study_id        TEXT NOT NULL,
    study_name      TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    preset          TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    counterbalance  TEXT NOT NULL,
    client_meta     TEXT NOT NULL,
    study_json      TEXT NOT NULL,
    session_json    TEXT NOT NULL,
    d               REAL,
    excluded        INTEGER NOT NULL DEFAULT 0,
    n_trials        INTEGER NOT NULL DEFAULT 0,
    app_version     TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    session_id      TEXT NOT NULL,
    idx             INTEGER NOT NULL,
    block_index     INTEGER NOT NULL,
    block_role      TEXT,
    pairing         TEXT,
    stimulus        TEXT NOT NULL,
    stimulus_category TEXT NOT NULL,
    correct_key     TEXT NOT NULL,
    response_key    TEXT,
    latency_ms      REAL NOT NULL,
    latency_to_correct_ms REAL,
    correct         INTEGER NOT NULL,
    timed_out       INTEGER NOT NULL DEFAULT 0,
    n_corrections   INTEGER NOT NULL DEFAULT 0,
    onset_uncertainty_ms REAL,
    dispatch_delay_ms REAL,
    focus_lost      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, idx)
);

CREATE TABLE IF NOT EXISTS designs (
    token           TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    study_id        TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    design_json     TEXT NOT NULL,
    consumed        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_study ON sessions(study_id);
CREATE INDEX IF NOT EXISTS idx_trials_session ON trials(session_id);
"""


class Store:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        # WAL keeps a reader from blocking the writer, which matters as soon as
        # two people open the demo at once.
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- designs -----------------------------------------------------------

    def put_design(self, design: dict) -> str:
        """Store a generated design and return its one-use token.

        The design is persisted at generation time rather than round-tripped
        through the browser. That means the condition a participant was in is a
        server-side fact: the client cannot report back a different
        counterbalance than the one it was given, and a session whose results
        never arrive still leaves a record that it was started.
        """
        token = uuid.uuid4().hex
        with self._conn() as c:
            c.execute(
                "INSERT INTO designs (token, created_at, study_id, seed, design_json) "
                "VALUES (?,?,?,?,?)",
                (
                    token,
                    datetime.now(timezone.utc).isoformat(),
                    design.get("study", {}).get("id", "unknown"),
                    int(design.get("seed", 0)),
                    json.dumps(design),
                ),
            )
        return token

    def get_design(self, token: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT design_json FROM designs WHERE token = ?", (token,)).fetchone()
        return json.loads(row["design_json"]) if row else None

    def consume_design(self, token: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE designs SET consumed = 1 WHERE token = ?", (token,))

    # -- writes ------------------------------------------------------------

    def save(
        self,
        *,
        session: dict,
        trials: list[dict],
        client_meta: dict,
        d: float | None,
        excluded: bool,
        app_version: str,
    ) -> str:
        sid = uuid.uuid4().hex[:12]
        study = session.get("study", {})
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT INTO sessions
                   (id, created_at, study_id, study_name, instrument, preset, seed,
                    counterbalance, client_meta, study_json, session_json, d, excluded,
                    n_trials, app_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sid, now, study.get("id", "unknown"), study.get("name", "Unknown"),
                    session.get("instrument", "iat"), session.get("preset", "standard"),
                    int(session.get("seed", 0)),
                    json.dumps(session.get("counterbalance", {})),
                    json.dumps(client_meta),
                    json.dumps(study),
                    json.dumps({k: v for k, v in session.items() if k != "trials"}),
                    d, int(excluded), len(trials), app_version,
                ),
            )
            c.executemany(
                """INSERT INTO trials
                   (session_id, idx, block_index, block_role, pairing, stimulus,
                    stimulus_category, correct_key, response_key, latency_ms,
                    latency_to_correct_ms, correct, timed_out, n_corrections,
                    onset_uncertainty_ms, dispatch_delay_ms, focus_lost)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        sid, t["index"], t["block_index"], t.get("block_role"),
                        t.get("pairing"), t["stimulus"], t["stimulus_category"],
                        t["correct_key"], t.get("response_key"), float(t["latency_ms"]),
                        t.get("latency_to_correct_ms"), int(bool(t["correct"])),
                        int(bool(t.get("timed_out"))), int(t.get("n_corrections", 0)),
                        t.get("onset_uncertainty_ms"), t.get("dispatch_delay_ms"),
                        int(bool(t.get("focus_lost"))),
                    )
                    for t in trials
                ],
            )
        return sid

    # -- reads -------------------------------------------------------------

    def get(self, session_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            trials = c.execute(
                "SELECT * FROM trials WHERE session_id = ? ORDER BY idx", (session_id,)
            ).fetchall()
        session = json.loads(row["session_json"])
        session["study"] = json.loads(row["study_json"])
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "session": session,
            "client_meta": json.loads(row["client_meta"]),
            "trials": [dict(t) for t in trials],
            "d": row["d"],
            "excluded": bool(row["excluded"]),
            "app_version": row["app_version"],
        }

    def norms(self, study_id: str) -> dict:
        """The distribution of D across every usable session of one study."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT d FROM sessions WHERE study_id = ? AND excluded = 0 AND d IS NOT NULL",
                (study_id,),
            ).fetchall()
        values = [float(r["d"]) for r in rows]
        return {"study_id": study_id, "n": len(values), "d_values": values}

    def recent(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id, created_at, study_name, instrument, preset, d, excluded, n_trials
                   FROM sessions ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"]
            usable = c.execute("SELECT COUNT(*) n FROM sessions WHERE excluded = 0").fetchone()["n"]
            trials = c.execute("SELECT COUNT(*) n FROM trials").fetchone()["n"]
        return {"sessions": total, "usable_sessions": usable, "trials": trials}
