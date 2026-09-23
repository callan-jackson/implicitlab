"""Batch ingestion: a raw fieldwork dump in, a validated trial table out.

Real exports are never clean. Column names drift between platforms ("rt",
"RT_ms", "latency"), booleans arrive as ``1``/``TRUE``/``yes``, participant
variables are repeated on every trial row and occasionally disagree with
themselves, and somebody has always re-uploaded half a session. The job of this
module is to absorb that without either crashing or silently guessing, and to
hand back an **ingest report** that says exactly what was accepted, what was
rejected and why — because "we dropped 312 rows" is a sentence a research
manager needs to be able to defend to a client.

Accepted formats:

*   **CSV** — one row per trial, long format. Participant variables (loyalty,
    age band, campaign cell, …) are ordinary columns repeated on each row, which
    is how Gorilla, Inquisit and most panel platforms export.
*   **JSON** — either a list of trial objects, ``{"trials": [...]}``, or
    ``{"participants": [...], "trials": [...]}`` where participant variables
    live in their own table and are joined on ``participant_id``.

Anything that is not a known trial column is treated as a candidate participant
variable, and becomes a segmentation variable if it is constant within each
participant and has a sensible number of levels.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED = ["participant_id", "block_role", "pairing", "latency_ms", "correct"]

TRIAL_COLUMNS = [
    "participant_id", "attribute", "target_a", "target_b", "pole_a", "pole_b",
    "task_order", "trial_index", "block_index", "block_role", "pairing",
    "stimulus", "stimulus_category", "correct_key", "response_key",
    "latency_ms", "latency_to_correct_ms", "correct", "timed_out",
]

#: Common spellings from other platforms, mapped onto ours. Kept deliberately
#: conservative: an alias that could plausibly mean two things (``condition``,
#: ``block``) is not guessed at.
ALIASES = {
    "pid": "participant_id", "participant": "participant_id",
    "respondent_id": "participant_id", "respondent": "participant_id",
    "subject": "participant_id", "subject_id": "participant_id",
    "rt": "latency_ms", "rt_ms": "latency_ms", "latency": "latency_ms",
    "reaction_time": "latency_ms", "reaction_time_ms": "latency_ms",
    "rt_to_correct": "latency_to_correct_ms", "latency_to_correct": "latency_to_correct_ms",
    "is_correct": "correct", "accuracy": "correct", "acc": "correct",
    "trial": "trial_index", "trial_number": "trial_index",
    "block_number": "block_index", "role": "block_role", "task": "attribute",
    "timeout": "timed_out",
}

TRUE_STRINGS = {"1", "true", "t", "yes", "y", "correct"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "incorrect", "error"}

MAX_SEGMENT_LEVELS = 8


class IngestError(ValueError):
    """The file cannot be interpreted at all (as opposed to bad rows in it)."""


@dataclass(slots=True)
class IngestResult:
    trials: pd.DataFrame
    meta: dict
    report: dict = field(default_factory=dict)


def _normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    renamed: dict[str, str] = {}
    cols = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        key = ALIASES.get(key, key)
        if key != c:
            renamed[str(c)] = key
        cols[c] = key
    df = df.rename(columns=cols)
    # A dump that ends up with two columns mapped to the same name (e.g. both
    # "rt" and "latency") is ambiguous; keep the first and say so.
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        df = df.loc[:, ~df.columns.duplicated()]
    return df, {"renamed": renamed, "duplicate_columns_dropped": dupes}


def _parse_bool(s: pd.Series) -> pd.Series:
    """Map the usual boolean spellings to True/False, anything else to NA."""
    if s.dtype == bool:
        return s.astype("boolean")
    txt = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[txt.isin(TRUE_STRINGS)] = True
    out[txt.isin(FALSE_STRINGS)] = False
    return out


def _read(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    text_head = raw[:64].lstrip()
    is_json = name.endswith(".json") or text_head[:1] in (b"{", b"[")
    if is_json:
        try:
            obj = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IngestError(f"Could not parse JSON: {exc}") from exc
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict) and isinstance(obj.get("trials"), list):
            trials = pd.DataFrame(obj["trials"])
            parts = obj.get("participants")
            if isinstance(parts, list) and parts:
                p = pd.DataFrame(parts)
                p, _ = _normalise_columns(p)
                trials, _ = _normalise_columns(trials)
                if "participant_id" not in p.columns:
                    raise IngestError("The participants table has no participant_id column.")
                overlap = [c for c in p.columns if c in trials.columns and c != "participant_id"]
                trials = trials.drop(columns=overlap).merge(
                    p.astype({"participant_id": str}),
                    how="left", on="participant_id",
                ) if "participant_id" in trials.columns else trials
            return trials
        raise IngestError(
            "JSON must be a list of trials, {\"trials\": [...]}, or "
            "{\"participants\": [...], \"trials\": [...]}."
        )
    try:
        return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                           na_values=["", "NA", "NaN", "nan", "null", "NULL", "None"],
                           encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise IngestError(f"Could not parse CSV: {exc}") from exc


def ingest(raw: bytes, filename: str = "upload.csv", *, name: str | None = None) -> IngestResult:
    """Parse, normalise and validate a batch upload."""
    df = _read(raw, filename)
    if df.empty:
        raise IngestError("The file contains no rows.")
    df, colinfo = _normalise_columns(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise IngestError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected at least: " + ", ".join(REQUIRED) + "."
        )

    n_read = len(df)
    df = df.copy()
    df["participant_id"] = df["participant_id"].astype("string").str.strip()
    if "attribute" not in df.columns:
        df["attribute"] = "primary"
    df["attribute"] = df["attribute"].astype("string").str.strip().fillna("primary")

    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "latency_to_correct_ms" in df.columns:
        df["latency_to_correct_ms"] = pd.to_numeric(df["latency_to_correct_ms"], errors="coerce")
    else:
        df["latency_to_correct_ms"] = np.nan
    df["correct"] = _parse_bool(df["correct"])
    df["timed_out"] = (
        _parse_bool(df["timed_out"]).fillna(False) if "timed_out" in df.columns
        else pd.Series(False, index=df.index, dtype="boolean")
    )
    for c in ("block_role", "pairing"):
        df[c] = df[c].astype("string").str.strip().str.lower().fillna("")
    for c, default in (("trial_index", None), ("block_index", 0), ("task_order", 0)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif default is not None:
            df[c] = default
    if "trial_index" not in df.columns or df["trial_index"].isna().all():
        df["trial_index"] = df.groupby(["participant_id", "attribute"]).cumcount()

    # ---- row-level validation -------------------------------------------------
    reasons = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        m = mask.fillna(True) & (reasons == "")
        reasons[m] = why

    flag(df["participant_id"].isna() | (df["participant_id"] == ""), "missing participant_id")
    flag(df["latency_ms"].isna(), "latency is missing or not a number")
    flag(df["latency_ms"] <= 0, "latency is zero or negative")
    flag(df["correct"].isna(), "correct is not a recognisable true/false value")
    flag(~df["block_role"].isin(["", "practice", "test"]),
         "block_role is not practice, test or blank")
    flag(~df["pairing"].isin(["", "congruent", "incongruent"]),
         "pairing is not congruent, incongruent or blank")
    flag((df["block_role"] != "") & (df["pairing"] == ""),
         "scored block with no pairing")

    rejected = reasons != ""
    rejection_counts = reasons[rejected].value_counts().to_dict()
    rejected_examples = (
        df.loc[rejected].head(5).assign(reason=reasons[rejected].head(5))
        .astype(object).where(lambda x: x.notna(), None).to_dict(orient="records")
    )
    df = df.loc[~rejected].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["participant_id", "attribute", "trial_index"], keep="first")
    n_dupes = before - len(df)

    if df.empty:
        raise IngestError("No rows survived validation. " + json.dumps(rejection_counts))

    df["correct"] = df["correct"].astype(bool)
    df["timed_out"] = df["timed_out"].astype(bool)

    # ---- participant variables ------------------------------------------------
    candidates = [c for c in df.columns if c not in TRIAL_COLUMNS]
    segments: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    warnings: list[str] = []
    for c in candidates:
        per_p = df.groupby("participant_id")[c].nunique(dropna=True)
        varying = int((per_p > 1).sum())
        if varying > max(1, 0.05 * len(per_p)):
            # Changes within most participants: a trial-level column (a
            # stimulus property, a timestamp), not a participant variable.
            skipped[c] = f"varies within {varying} participants — a trial-level column"
            continue
        if varying:
            warnings.append(
                f"'{c}' varies within {varying} participant(s); their first recorded value was used."
            )
            first = df.groupby("participant_id")[c].first()
            df[c] = df["participant_id"].map(first)

        values = df.groupby("participant_id")[c].first()
        numeric = pd.to_numeric(values, errors="coerce")
        n_levels = values.nunique(dropna=True)
        if n_levels < 2:
            skipped[c] = "the same value for every participant"
            continue
        if numeric.notna().all() and n_levels > MAX_SEGMENT_LEVELS:
            # A continuous participant variable (age in years, spend) is cut
            # into tertiles rather than ignored. Tertiles, not a median split:
            # a median split throws away the ends, which is usually where the
            # interesting respondents are.
            bands = pd.qcut(numeric, 3, duplicates="drop")
            labels = [f"{iv.left:g}–{iv.right:g}" for iv in bands.cat.categories]
            bands = bands.cat.rename_categories(labels)
            df[c] = df["participant_id"].map(bands.astype(str))
            segments[c] = {"label": c.replace("_", " ").capitalize(), "levels": labels,
                           "note": "Continuous in the upload; cut into tertiles."}
            continue
        if n_levels > MAX_SEGMENT_LEVELS:
            skipped[c] = f"{n_levels} distinct values — too many to segment on"
            continue
        levels = sorted(str(v) for v in values.dropna().unique())
        df[c] = df[c].astype("string")
        segments[c] = {"label": c.replace("_", " ").capitalize(), "levels": levels, "note": ""}

    # ---- attribute metadata ---------------------------------------------------
    attributes: dict[str, dict] = {}
    for attr, g in df.groupby("attribute", sort=False):
        def first(col: str, default: str) -> str:
            if col in g.columns and g[col].notna().any():
                return str(g[col].dropna().iloc[0])
            return default
        pole_a = first("pole_a", str(attr).replace("_", " ").capitalize())
        attributes[str(attr)] = {
            "label": pole_a,
            "dimension": f"{pole_a} vs {first('pole_b', 'contrast')}",
            "target_a": first("target_a", "Target A"),
            "target_b": first("target_b", "Target B"),
            "pole_a": pole_a,
            "pole_b": first("pole_b", "contrast"),
        }

    scored = df[(df["block_role"] != "")]
    if scored.empty:
        raise IngestError("No rows belong to a scored (practice or test) block.")

    unscorable = int((~scored["correct"] & scored["latency_to_correct_ms"].isna()).sum())
    if unscorable:
        # The cohort scorer uses the built-in error penalty (forced correction,
        # time-to-correct scored). An export from a task that did not force
        # correction has no such latency for its errors, and substituting a
        # +600 ms penalty silently would change the algorithm under the
        # operator's feet. Say so instead.
        warnings.append(
            f"{unscorable:,} error trials have no latency_to_correct_ms, so they cannot be "
            f"scored under the built-in error penalty and are left out of D. If this task "
            f"did not force error correction, it needs the +600 ms procedure instead."
        )
    if not any(c in df.columns for c in ("target_a", "pole_a")):
        warnings.append(
            "No target_a / pole_a label columns: results are labelled generically "
            "(Target A, Target B). Add them to the file to get brand and attribute names."
        )

    meta = {
        "name": name or (filename or "Uploaded batch").rsplit(".", 1)[0],
        "source": "upload",
        "filename": filename,
        "n_participants": int(df["participant_id"].nunique()),
        "attributes": attributes,
        "segments": segments,
        "note": "Uploaded by the operator.",
    }
    report = {
        "rows_read": n_read,
        "rows_accepted": int(len(df)),
        "rows_rejected": int(rejected.sum()),
        "rejection_reasons": {str(k): int(v) for k, v in rejection_counts.items()},
        "rejected_examples": rejected_examples,
        "duplicates_removed": int(n_dupes),
        "participants": int(df["participant_id"].nunique()),
        "attributes": list(attributes),
        "segment_variables": list(segments),
        "skipped_columns": skipped,
        "columns_renamed": colinfo["renamed"],
        "warnings": warnings,
    }
    keep_cols = [c for c in TRIAL_COLUMNS if c in df.columns] + list(segments)
    return IngestResult(trials=df[keep_cols].reset_index(drop=True), meta=meta, report=report)


def to_csv_bytes(trials: pd.DataFrame, segments: list[str]) -> bytes:
    """The canonical long-format CSV — the same layout ``ingest`` reads back."""
    front = ["participant_id", *segments]
    rest = [c for c in TRIAL_COLUMNS if c in trials.columns and c not in front]
    return trials[front + rest].to_csv(index=False).encode("utf-8")
