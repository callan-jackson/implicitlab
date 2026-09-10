"""Data-quality assessment, separate from scoring.

Scoring answers "what is D". This module answers "should anyone believe it",
which is a different question and deserves its own code path. Keeping them
apart means a quality rule can be tightened without touching the algorithm, and
means the report can show *why* a session was rejected instead of just
producing a blank.

The checks fall into three groups:

*   **Engagement** — did the participant actually do the task? Too-fast
    responding, excessive errors, long stalls.
*   **Instrumentation** — did the browser measure what it claims? Low refresh
    rate, event-timestamp fallback, tab focus loss, high event-dispatch delay.
*   **Design** — is there enough data, and is it balanced? Trial counts per
    cell, block-order effects.

Each check returns a severity. ``fail`` blocks the score; ``warn`` lets it
through with the caveat attached; ``info`` is context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import ScoringConfig, Trial

Severity = Literal["pass", "info", "warn", "fail"]


@dataclass(slots=True)
class Check:
    key: str
    label: str
    severity: Severity
    detail: str
    value: float | str | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "severity": self.severity,
            "detail": self.detail,
            "value": self.value,
        }


def assess(
    trials: list[Trial],
    config: ScoringConfig,
    *,
    client_meta: dict | None = None,
) -> dict:
    """Run every quality check and summarise."""
    client_meta = client_meta or {}
    checks: list[Check] = []

    scored = [t for t in trials if t.block_role in ("practice", "test", "sciat") and t.pairing]
    n = len(scored)

    # ---- Engagement -------------------------------------------------------
    if n:
        fast = sum(1 for t in scored if t.latency_ms < config.fast_trial_threshold_ms)
        prop = fast / n
        checks.append(Check(
            key="too_fast",
            label="Fast-guess responding",
            severity="fail" if prop > config.fast_trial_proportion_limit else ("warn" if prop > 0.05 else "pass"),
            detail=(
                f"{fast} of {n} scored trials ({prop:.1%}) were faster than "
                f"{config.fast_trial_threshold_ms:.0f} ms. Responses that fast cannot "
                f"reflect having read the stimulus; the standard threshold is "
                f"{config.fast_trial_proportion_limit:.0%} of trials."
            ),
            value=round(prop, 4),
        ))

        errs = sum(1 for t in scored if not t.correct)
        err_rate = errs / n
        checks.append(Check(
            key="error_rate",
            label="Error rate",
            severity="fail" if err_rate > 0.40 else ("warn" if err_rate > 0.25 else "pass"),
            detail=(
                f"{errs} of {n} scored trials were errors ({err_rate:.1%}). Some errors are "
                f"expected and are informative — the task is meant to be hard in one "
                f"pairing. Above roughly 25% the participant is trading accuracy for "
                f"speed and D starts measuring that trade-off instead."
            ),
            value=round(err_rate, 4),
        ))

        slow = sum(1 for t in scored if t.latency_ms > config.upper_cutoff_ms)
        checks.append(Check(
            key="stalls",
            label="Stalled trials",
            severity="warn" if slow > n * 0.05 else "pass",
            detail=(
                f"{slow} trials exceeded {config.upper_cutoff_ms:.0f} ms and were removed. "
                f"These usually mean the participant looked away rather than deliberated."
            ),
            value=slow,
        ))

        lat = np.asarray([t.latency_ms for t in scored if t.correct], dtype=float)
        if lat.size > 4:
            checks.append(Check(
                key="latency_spread",
                label="Latency distribution",
                severity="info",
                detail=(
                    f"Correct-trial latencies: median {np.median(lat):.0f} ms, "
                    f"10th-90th percentile {np.percentile(lat, 10):.0f}-"
                    f"{np.percentile(lat, 90):.0f} ms. Typical IAT trials land between "
                    f"500 and 1,200 ms; a median far outside that suggests the task was "
                    f"either not understood or not taken seriously."
                ),
                value=round(float(np.median(lat)), 1),
            ))
    else:
        checks.append(Check(
            key="no_data", label="Scorable trials", severity="fail",
            detail="No scorable trials were submitted.", value=0,
        ))

    # ---- Design balance ---------------------------------------------------
    cells: dict[tuple, int] = {}
    for t in scored:
        cells[(t.block_role, t.pairing)] = cells.get((t.block_role, t.pairing), 0) + 1
    if cells:
        smallest = min(cells.values())
        checks.append(Check(
            key="cell_counts",
            label="Trials per condition",
            severity="fail" if smallest < config.min_trials_per_block else ("warn" if smallest < 10 else "pass"),
            detail=(
                "Retained trials per condition: "
                + ", ".join(f"{r}/{p}: {c}" for (r, p), c in sorted(cells.items(), key=str))
                + ". D is an average of within-cell means, so the smallest cell sets the "
                  "precision of the whole estimate."
            ),
            value=smallest,
        ))

    # ---- Instrumentation --------------------------------------------------
    hz = client_meta.get("refresh_hz")
    if hz:
        checks.append(Check(
            key="refresh",
            label="Display refresh rate",
            severity="pass" if hz >= 55 else "warn",
            detail=(
                f"Measured {hz:.1f} Hz, giving a frame interval of "
                f"{1000 / hz:.1f} ms and an irreducible stimulus-onset uncertainty of "
                f"about ±{500 / hz:.1f} ms. This is a constant offset that cancels in a "
                f"difference score like D."
                + ("" if hz >= 55 else " Below 55 Hz the timing resolution is poor enough "
                                        "to be worth noting in any report.")
            ),
            value=round(float(hz), 1),
        ))

    dispatch = [t for t in trials if getattr(t, "dispatch_delay_ms", None) is not None]
    dd = client_meta.get("median_dispatch_delay_ms")
    if dd is not None:
        checks.append(Check(
            key="dispatch_delay",
            label="Event-loop delay",
            severity="warn" if dd > 8 else "pass",
            detail=(
                f"Median delay between the browser stamping a keypress and this page's "
                f"handler running was {dd:.2f} ms. Latencies here are taken from the "
                f"browser's own event timestamp, so this delay is measured and excluded "
                f"rather than absorbed into the reaction time — which is what would "
                f"happen if the handler called performance.now() instead."
            ),
            value=round(float(dd), 3),
        ))

    if client_meta.get("used_event_timestamp") is False:
        checks.append(Check(
            key="timestamp_fallback", label="Timestamp source", severity="warn",
            detail=(
                "This browser did not provide a usable high-resolution event timestamp, "
                "so latencies fall back to a clock read inside the handler. That adds the "
                "event-loop delay to every trial."
            ),
        ))

    degraded = client_meta.get("degraded_onsets", 0)
    if degraded:
        checks.append(Check(
            key="degraded_onsets",
            label="Un-timed stimulus onsets",
            severity="fail" if degraded > n * 0.1 else "warn",
            detail=(
                f"{degraded} trials were presented while the browser was not delivering "
                f"animation frames, usually because the tab was in the background. Onset "
                f"for those trials is timestamped from the clock rather than from a "
                f"painted frame, so their latencies carry an unknown extra offset."
            ),
            value=degraded,
        ))

    focus = client_meta.get("focus_losses", 0)
    if focus:
        checks.append(Check(
            key="focus", label="Tab focus", severity="warn",
            detail=(
                f"The tab lost focus {focus} time(s) during the session. Browsers throttle "
                f"background timers, so any trial spanning a focus change is suspect."
            ),
            value=focus,
        ))

    severities = [c.severity for c in checks]
    overall: Severity = "fail" if "fail" in severities else ("warn" if "warn" in severities else "pass")

    return {
        "overall": overall,
        "usable": overall != "fail",
        "checks": [c.to_dict() for c in checks],
        "summary": {
            "pass": severities.count("pass"),
            "info": severities.count("info"),
            "warn": severities.count("warn"),
            "fail": severities.count("fail"),
        },
    }
