"""Tests for the numeric guardrail on model-generated text.

The fixture that matters is the deliberately hallucinating one: a summary that
reads perfectly and quotes a figure that was never computed. If the verifier
does not catch that, the guardrail is decoration.
"""

from __future__ import annotations

from app.llm.prompts import build_payload, fallback_summary
from app.llm.verify import extract_numbers, verify

PAYLOAD = build_payload(
    study={
        "name": "Aurelia vs Northvane",
        "sector": "Premium spirits",
        "instrument": "iat",
        "target_a": {"label": "AURELIA"},
        "target_b": {"label": "NORTHVANE"},
    },
    result={
        "d": 0.412,
        "d_practice": 0.388,
        "d_test": 0.436,
        "direction": "a",
        "magnitude": "moderate",
        "interpretation": "D = +0.412. A moderate relative association favouring AURELIA.",
        "n_trials_scored": 72,
        "n_trials_dropped": 3,
        "blocks": [
            {
                "block_index": 3, "block_role": "practice", "pairing": "congruent",
                "n_presented": 18, "n_retained": 18, "n_errors": 2, "error_rate": 0.111,
                "mean_correct_ms": 742.0, "mean_scored_ms": 780.0, "median_ms": 731.0,
                "sd_ms": 154.0,
            },
        ],
        "config": {"error_penalty": "built_in"},
        "warnings": [],
    },
    descriptives={"congruent": {"mean_ms": 742.0}, "incongruent": {"mean_ms": 851.0}},
    inference={
        "ci_low": 0.104, "ci_high": 0.702, "ci_level": 0.95,
        "p_permutation": 0.021, "n_resamples": 2000, "note": "bootstrap",
    },
    reliability={"d_odd_trials": 0.39, "d_even_trials": 0.44,
                 "absolute_difference": 0.05, "stable": True},
    quality={"overall": "pass", "excluded": False, "exclusion_reasons": []},
    session_meta={
        "preset_label": "Standard demo", "preset_note": "About 60% of trials.",
        "n_trials": 72, "counterbalance": {"congruent_pairing_first": True},
        "refresh_hz": 60.0, "onset_uncertainty_ms": 8.3,
    },
)


def test_extracts_signed_and_decimal_numbers():
    found = dict(extract_numbers("D was +0.412, the CI ran to 0.702, error rate 11%."))
    assert 0.412 in found.values()
    assert 0.702 in found.values()
    assert 11.0 in found.values()


def test_accepts_a_summary_that_only_uses_computed_figures():
    text = (
        "### Headline\n"
        "The participant showed a moderate relative association favouring AURELIA "
        "(D = +0.412).\n\n"
        "### What the numbers say\n"
        "- D came out at 0.412, with the practice pair at 0.388 and the test pair at 0.436.\n"
        "- 72 trials were scored and 3 were dropped by the latency cut-offs.\n"
        "- Mean correct latency in the congruent practice block was 742 ms.\n\n"
        "### How confident we can be\n"
        "The 95% bootstrap interval runs from 0.104 to 0.702 and the permutation "
        "p-value over 2000 relabellings was 0.021.\n"
    )
    report = verify(text, PAYLOAD)
    assert report.verified is True, report.unverified
    assert report.checked > 5


def test_catches_a_fabricated_statistic():
    """The core case: fluent prose, invented figure."""
    text = (
        "### Headline\n"
        "AURELIA showed a moderate implicit advantage (D = +0.412).\n\n"
        "### What the numbers say\n"
        "- Responses were 337 ms faster in the congruent pairing.\n"
        "- Reliability was excellent at alpha = 0.91.\n"
    )
    report = verify(text, PAYLOAD)
    assert report.verified is False
    assert "337" in " ".join(report.unverified)
    assert any("0.91" in u for u in report.unverified)
    assert "could not be matched" in report.note


def test_catches_a_subtly_wrong_number():
    """0.512 instead of 0.412 is the failure that would survive a human skim."""
    report = verify("The D-score was 0.512.", PAYLOAD)
    assert report.verified is False
    assert "0.512" in " ".join(report.unverified)


def test_percentages_match_proportions_in_the_payload():
    report = verify("The error rate in that block was 11.1%.", PAYLOAD)
    assert report.verified is True, report.unverified


def test_structural_and_citation_numbers_are_allowed():
    text = (
        "Three points follow. The conventional bands are 0.15, 0.35 and 0.65, "
        "from Greenwald et al. (2003), and the alpha level is 0.05."
    )
    assert verify(text, PAYLOAD).verified is True


def test_the_deterministic_summary_passes_its_own_guardrail():
    """The fallback must be self-consistent, or the fallback is the bug."""
    text = fallback_summary(PAYLOAD)
    report = verify(text, PAYLOAD)
    assert report.verified is True, report.unverified


def test_deterministic_summary_leads_with_the_null_when_the_ci_spans_zero():
    payload = {**PAYLOAD}
    payload["inference"] = {**PAYLOAD["inference"], "confidence_interval": [-0.21, 0.55],
                            "p_permutation": 0.31}
    text = fallback_summary(payload)
    headline = text.split("###")[1]
    assert "not produce an association effect distinguishable from zero" in headline


def test_deterministic_summary_refuses_to_score_an_excluded_session():
    payload = {**PAYLOAD}
    payload["quality"] = {"excluded": True, "exclusion_reasons": ["too many fast trials"]}
    payload["score"] = {**PAYLOAD["score"], "d": None}
    text = fallback_summary(payload)
    assert "No interpretable result" in text
    assert "too many fast trials" in text


def test_no_exclamation_marks_or_marketing_language_in_the_fallback():
    text = fallback_summary(PAYLOAD)
    assert "!" not in text
    for word in ("amazing", "incredible", "revolutionary", "unlock", "really thinks"):
        assert word not in text.lower()
