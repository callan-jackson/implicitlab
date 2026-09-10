"""Numeric guardrail for model-generated text.

The insight layer is downstream of the statistics, but "downstream" is a design
intention and intentions are not enforcement. This module is the enforcement:
it pulls every number out of the generated summary and checks that each one
actually appears in the payload the model was given.

Why bother, when the prompt already forbids inventing figures? Because a
plausible fabricated number is the single most damaging failure mode for this
kind of tool. Prose that is merely clumsy gets edited. A summary that says
"response times were 340 ms faster" when the real gap was 90 ms is
indistinguishable from a correct one to the person reading it, and it will be
quoted in a deck. A prompt instruction reduces the rate of that; it does not
make it observable. Checking the output does.

The check is deliberately conservative in what it *permits* and loud about what
it cannot verify: anything unmatched is surfaced to the caller rather than
silently stripped, and the API response carries the flag so the front end can
show it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches 12, 1,234, -0.35, +0.352, 45%, 380ms — with the sign and suffix
# captured so "+0.35" and "0.35" both normalise to the same value.
#
# The trailing lookahead is deliberately narrow. An earlier version rejected any
# number followed by a full stop, which meant a fabricated figure at the end of a
# sentence — "reliability was 0.91." — was never extracted and therefore never
# checked. That is precisely the case the guardrail exists for, so the lookahead
# now only refuses to stop mid-number: another digit, a thousands comma, or a
# decimal point that is itself followed by a digit.
NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                 # not mid-identifier
    ([+-]?)                    # sign
    (\d{1,3}(?:,\d{3})+|\d+)   # integer part, optionally comma-grouped
    (?:\.(\d+))?               # decimal part
    \s*(%)?                    # percent marker
    (?!\d)(?!,\d)(?!\.\d)      # not truncating a longer number
    """,
    re.VERBOSE,
)

#: Numbers that never need justifying: they are structural (bullet counts,
#: section numbers), or they are the published interpretation thresholds and the
#: conventional alpha level, all of which are stated in the payload's scale note.
ALWAYS_ALLOWED = {
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    0.05,                      # conventional alpha
    0.15, 0.35, 0.65,          # Greenwald effect-size thresholds
    300.0, 400.0, 600.0, 350.0, 1500.0, 10000.0,  # published cut-offs and penalties
    95.0, 100.0,               # confidence level, percentages
    2003.0, 2006.0,            # citation years
}

TOLERANCE = 0.006


@dataclass(slots=True)
class VerificationReport:
    verified: bool
    checked: int
    unverified: list[str]
    note: str

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "numbers_checked": self.checked,
            "unverified_numbers": self.unverified,
            "note": self.note,
        }


def _collect(obj, out: set[float]) -> None:
    """Walk the payload and collect every number, plus its plausible renderings."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        out.add(v)
        out.add(abs(v))
        for dp in (0, 1, 2, 3):
            out.add(round(v, dp))
            out.add(abs(round(v, dp)))
        # A proportion in the payload is very often written as a percentage.
        if -1.0 <= v <= 1.0:
            out.add(round(v * 100, 0))
            out.add(round(v * 100, 1))
            out.add(abs(round(v * 100, 0)))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect(value, out)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _collect(value, out)
        return
    if isinstance(obj, str):
        # Numbers embedded in payload strings (e.g. the plain-language reading
        # of D, or a warning that quotes a count) are legitimate to reuse.
        for m in NUMBER_RE.finditer(obj):
            try:
                out.add(abs(_value(m)))
                out.add(_value(m))
            except ValueError:
                continue


def _value(m: re.Match) -> float:
    sign, intpart, dec, pct = m.groups()
    raw = intpart.replace(",", "")
    text = f"{raw}.{dec}" if dec else raw
    v = float(text)
    if sign == "-":
        v = -v
    return v


def extract_numbers(text: str) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for m in NUMBER_RE.finditer(text):
        try:
            found.append((m.group(0).strip(), _value(m)))
        except ValueError:
            continue
    return found


def verify(text: str, payload: dict) -> VerificationReport:
    """Check that every number in ``text`` traces back to ``payload``."""
    allowed: set[float] = set(ALWAYS_ALLOWED)
    _collect(payload, allowed)
    allowed = {round(v, 4) for v in allowed}

    unverified: list[str] = []
    numbers = extract_numbers(text)

    for literal, value in numbers:
        candidates = {value, abs(value), round(value, 3), round(abs(value), 3)}
        # A percentage in the prose may correspond to a proportion in the payload.
        if literal.endswith("%"):
            candidates |= {value / 100, round(value / 100, 4), round(abs(value) / 100, 4)}
        if any(any(abs(c - a) <= TOLERANCE for a in allowed) for c in candidates):
            continue
        unverified.append(literal)

    ok = not unverified
    note = (
        "Every numeric claim in the summary was matched to a value computed by "
        "the statistics layer."
        if ok
        else (
            f"{len(unverified)} numeric claim(s) in the summary could not be "
            f"matched to any computed value: {', '.join(unverified[:8])}. "
            f"The summary has been flagged rather than published unchecked."
        )
    )
    return VerificationReport(verified=ok, checked=len(numbers), unverified=unverified, note=note)
