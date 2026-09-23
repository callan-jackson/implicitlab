"""Prompt construction for the insight layer.

The governing rule for this module is that **the language model never touches
the data**. It receives a JSON object of numbers that Python has already
computed, tested and quality-flagged, and its only job is to turn those numbers
into prose a non-technical stakeholder can act on. It cannot recompute a
D-score, cannot see a raw latency, and is told in the system prompt that every
figure it uses must be copied from the payload.

That boundary is not stylistic caution. An implicit measure produces a number
whose interpretation is already contested in the literature; a model that is
free to invent a supporting statistic can make a null result read like a
finding, and nobody downstream would catch it. Keeping the model strictly
downstream of deterministic statistics means the worst it can do is describe
real numbers badly, which is a reviewable failure rather than a silent one.

The generated text is additionally checked by :mod:`app.llm.verify`, which
extracts every number from the output and confirms it appears in the payload.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are a senior consumer-research analyst writing for a brand client. You \
specialise in implicit measures: reaction-time based tests that index automatic \
evaluative associations rather than stated opinion.

You will be given a JSON object containing statistics that have ALREADY been \
computed from a completed session. Your job is to write the executive summary.

HARD RULES — these are not style preferences:

1. NEVER state a number that is not present in the JSON payload. Do not \
   recompute, do not estimate, do not round into a new figure, do not infer a \
   percentage that is not given. If you want to make a point that would need a \
   number you do not have, make the point qualitatively instead.
2. NEVER describe a single participant's result as evidence about a population, \
   a market, a demographic, or "consumers". One session is one person on one \
   occasion. Say so.
3. Do NOT reason about the statistics yourself, not even trivially. Whether the \
   interval excludes zero, whether the result is significant, and whether the \
   session is usable have already been decided and are supplied to you as a \
   verdict. Follow that verdict. Never compare two numbers and state a \
   conclusion of your own — a wrong claim about a correct number is the one \
   error the numeric check downstream of you cannot catch. Express the verdict \
   in your own words as flowing prose; do not quote it, do not name it, and \
   never reproduce any instruction text you were given.
4. If the verdict says the result is not distinguishable from no effect, you MUST \
   lead with that. Do not bury it, and do not describe a direction as if it were \
   established.
5. NEVER claim the test reveals what someone "really thinks", their true \
   feelings, or anything hidden or unconscious in a strong sense. The defensible \
   claim is narrow and mechanical: the participant was faster to respond when \
   certain categories shared a response key. Automatic association is not the \
   same as preference, intention, or behaviour.
6. NEVER give clinical, diagnostic, or individual-characterising statements. \
   The IAT is not valid for individual assessment and you must not use it as if \
   it were.
7. Do not editorialise about the brands themselves or introduce outside \
   knowledge about them beyond what the payload names.

PRESENTATION:

8. Copy every figure exactly as it appears in the payload. It has already been \
   rounded to the precision it should be reported at, so do not add digits and \
   do not drop them. Latencies may be written as whole milliseconds.
9. NEVER mention JSON field names, key paths, or the structure of the payload. \
   Do not write "quality.error_rate.value" or "(plain_reading: ...)". Write for \
   a brand client who will never see the JSON.
10. Do not restate the same figure in more than one section, and do not quote \
    the supplied plain-language reading word for word — write your own sentence.

STRUCTURE — return GitHub-flavoured Markdown with exactly these sections:

### Headline
One sentence. What the session showed, hedged exactly as strongly as the \
statistics warrant.

### What the numbers say
Three to five bullets. Each bullet must cite a figure from the payload. Explain \
what D is in one clause the first time you use it.

### How confident we can be
Cover: the confidence interval, the permutation p-value, the split-half \
diagnostic, the error rate, and the trial count. Be direct about what limits \
this result. If the session used a shortened preset, say that it widens the \
interval.

### What a researcher would do next
Two to four concrete next steps that would strengthen or falsify this result — \
sample size, a comparator brand, an explicit measure to contrast against, a \
retest. Be specific to what the payload shows.

### Caveats
Two to three bullets. Always include that this is a single session, and always \
include one genuine limitation of implicit measures generally.

Total length: 280-420 words. Plain professional English. No emoji, no \
exclamation marks, no marketing language. Write as if the client will forward \
this to their own statistician.
"""


def _round_value(v, key: str = ""):
    """Round a payload value to the precision the report should quote it at.

    The model is only allowed to use numbers that appear in the payload, so the
    cleanest way to stop it printing ``0.6301722137949044`` is not to ask it
    politely — it is to never show it that many digits. Rounding here also makes
    the numeric verifier's job exact rather than tolerance-based.
    """
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        k = key.lower()
        if k.endswith("_ms") or "latency" in k or "_hz" in k:
            return round(f, 1)
        if k.startswith("p_"):
            return round(f, 4)
        if float(f).is_integer() and abs(f) < 1e9:
            return int(f)
        return round(f, 3)
    return v


def _round_payload(obj, key: str = ""):
    if isinstance(obj, dict):
        return {k: _round_payload(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_payload(v, key) for v in obj]
    return _round_value(obj, key)


def build_user_prompt(payload: dict) -> str:
    """Wrap the computed statistics in a short framing message."""
    return (
        "Here is the completed session. Every figure you use must come from "
        "this object.\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2, default=str)}\n"
        "```\n\n"
        "Write the executive summary now, following the structure and the hard "
        "rules exactly."
    )


def _verdicts(inference: dict, result: dict, quality: dict) -> dict:
    """Precompute the yes/no readings so the model never has to derive one."""
    lo, hi = inference.get("ci_low"), inference.get("ci_high")
    p = inference.get("p_permutation")
    d = result.get("d")

    excludes_zero = None
    if lo is not None and hi is not None:
        excludes_zero = not (lo <= 0 <= hi)

    significant = None if p is None else bool(p < 0.05)
    usable = not (quality.get("excluded") or result.get("excluded"))

    if not usable or d is None:
        label = "not_usable"
        plain = (
            "this session failed the data-quality criteria, so no association "
            "estimate is reported"
        )
    elif excludes_zero is None or significant is None:
        label = "inconclusive"
        plain = (
            "there were too few usable trials to test the estimate, so the result "
            "is unresolved rather than either a finding or a null"
        )
    elif excludes_zero and significant:
        label = "distinguishable_from_zero"
        plain = (
            "for this participant on this occasion, the effect can be told apart "
            "from no effect"
        )
    elif not excludes_zero and not significant:
        label = "not_distinguishable_from_zero"
        plain = (
            "this session did not produce an effect that can be told apart from "
            "no effect"
        )
    else:
        # The interval and the permutation test are answering slightly different
        # questions and can land either side of the line when the effect is
        # marginal. Reporting that as a clean null throws away the fact that the
        # two criteria disagreed, which is itself the informative part.
        label = "inconclusive"
        plain = (
            "the confidence interval and the permutation test disagree, which "
            "happens when an effect is marginal; the result is unresolved rather "
            "than either a finding or a null"
        )

    return {
        "confidence_interval_excludes_zero": excludes_zero,
        "significant_at_0_05": significant,
        "session_usable": usable,
        "verdict_label": label,
        "verdict_in_plain_english": plain,
    }


def build_payload(
    *,
    study: dict,
    result: dict,
    descriptives: dict,
    inference: dict,
    reliability: dict,
    quality: dict,
    session_meta: dict,
) -> dict:
    """Assemble the exact object the model is allowed to see.

    Deliberately excludes raw trial data. The model cannot be asked to
    "re-check" a figure, because it has nothing to re-check it against.
    """
    payload = {
        "study": {
            "name": study.get("name"),
            "sector": study.get("sector"),
            "instrument": study.get("instrument"),
            "target_a": (study.get("target_a") or {}).get("label"),
            "target_b": (study.get("target_b") or {}).get("label"),
        },
        "session": {
            "preset": session_meta.get("preset_label"),
            "preset_caveat": session_meta.get("preset_note"),
            "n_trials_presented": session_meta.get("n_trials"),
            "n_trials_scored": result.get("n_trials_scored"),
            "n_trials_dropped": result.get("n_trials_dropped"),
            "congruent_pairing_first": (session_meta.get("counterbalance") or {}).get(
                "congruent_pairing_first"
            ),
            "display_refresh_hz": session_meta.get("refresh_hz"),
            "onset_uncertainty_ms": session_meta.get("onset_uncertainty_ms"),
        },
        "score": {
            "d": result.get("d"),
            "d_practice_blocks": result.get("d_practice"),
            "d_test_blocks": result.get("d_test"),
            "direction": result.get("direction"),
            "magnitude_label": result.get("magnitude"),
            "plain_reading": result.get("interpretation"),
            "scale_note": (
                "D is a standardised latency difference. Conventional labels: "
                "|D| < 0.15 little to none, 0.15-0.35 slight, 0.35-0.65 "
                "moderate, above 0.65 strong. Positive D favours the first "
                "target listed."
            ),
        },
        "blocks": result.get("blocks"),
        "descriptives": descriptives,
        "inference": {
            "confidence_interval": [inference.get("ci_low"), inference.get("ci_high")],
            "confidence_level": inference.get("ci_level"),
            "p_permutation": inference.get("p_permutation"),
            "n_resamples": inference.get("n_resamples"),
            "method": inference.get("note"),
            # Every comparison the model would otherwise have to make itself is
            # made here instead. Asking a language model whether 0.17 is greater
            # than 0 is asking for an error that the numeric verifier cannot
            # see, because no number is wrong — only the claim about it is.
            **_verdicts(inference, result, quality),
        },
        "reliability": reliability,
        "quality": quality,
        "algorithm": {
            "name": "Greenwald, Nosek & Banaji (2003) improved scoring algorithm",
            "parameters": result.get("config"),
            "warnings": result.get("warnings"),
        },
    }
    # Round once, at the boundary. Everything downstream — the prompt, the
    # numeric verifier, and the deterministic fallback — then works from the
    # same rounded figures, so the three can never disagree about a value.
    return _round_payload(payload)


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------

def fallback_summary(payload: dict) -> str:
    """A rules-based executive summary used when no model is configured.

    This is not a stub. It is the same report the model is asked to write,
    assembled from the same payload by explicit branching, and it is what the
    service returns whenever the model is unavailable or its output fails the
    numeric check. Having a deterministic path means the tool degrades to
    "less fluent" rather than to "broken", and it gives a reference the model's
    output can be compared against.
    """
    score = payload["score"]
    inf = payload["inference"]
    q = payload["quality"]
    sess = payload["session"]
    study = payload["study"]
    rel = payload.get("reliability") or {}

    a = study.get("target_a") or "the brand"
    b = study.get("target_b")
    d = score.get("d")
    ci = inf.get("confidence_interval") or [None, None]
    p = inf.get("p_permutation")

    lines: list[str] = []

    if q.get("excluded") or d is None:
        reasons = "; ".join(q.get("exclusion_reasons") or ["data-quality criteria were not met"])
        lines.append("### Headline")
        lines.append(
            f"No interpretable result: this session did not meet the scoring "
            f"criteria, so no association estimate is reported ({reasons})."
        )
        lines.append("")
        lines.append("### What the numbers say")
        lines.append(f"- {sess.get('n_trials_scored')} trials were scorable out of "
                     f"{sess.get('n_trials_presented')} presented.")
        lines.append(f"- {sess.get('n_trials_dropped')} trials fell outside the latency cut-offs.")
        lines.append("- No D-score is produced when the exclusion criteria trigger. "
                     "Reporting one anyway would be the error.")
        lines.append("")
        lines.append("### How confident we can be")
        lines.append("Not at all, and that is the correct answer here rather than a failure "
                     "of the tool. The exclusion rules exist so that disengaged responding "
                     "does not get reported as a finding.")
        lines.append("")
        lines.append("### What a researcher would do next")
        lines.append("- Re-run the session with the full trial preset and a rested participant.")
        lines.append("- Check that the instructions were understood before the first scored block.")
        lines.append("")
        lines.append("### Caveats")
        lines.append("- This is a single session from one person on one occasion.")
        lines.append("- Implicit measures have modest test-retest reliability even when the "
                     "session is clean.")
        return "\n".join(lines)

    significant = p is not None and p < 0.05
    ci_spans_zero = ci[0] is not None and ci[1] is not None and ci[0] <= 0 <= ci[1]
    solid = significant and not ci_spans_zero

    favoured = a if d > 0 else (b or a)
    other = (b or "the alternative") if d > 0 else a
    mag = score.get("magnitude_label", "")

    lines.append("### Headline")
    level = int(inf.get("confidence_level", 0.95) * 100)
    if not solid and (significant or not ci_spans_zero) and p is not None:
        # The interval and the permutation test landed either side of the line.
        # Calling that a clean null would contradict the interval printed in the
        # same sentence; calling it a finding would overrule the other test.
        lines.append(
            f"The evidence for this session is marginal: the {level}% CI "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] "
            f"{'excludes' if not ci_spans_zero else 'includes'} zero but the permutation "
            f"test gives p = {p:.3f}, so the two criteria disagree. The lean toward "
            f"{favoured} (D = {d:+.3f}) is unresolved rather than either a finding or a null."
        )
    elif not solid:
        lines.append(
            f"This session did not produce an association effect distinguishable from zero "
            f"(D = {d:+.3f}, {int(inf.get('confidence_level', 0.95) * 100)}% CI "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]), so the apparent lean toward {favoured} should "
            f"not be read as a finding."
        )
    elif b:
        lines.append(
            f"The participant showed a {mag} relative automatic preference for {favoured} "
            f"over {other} (D = {d:+.3f})."
        )
    else:
        direction = "positive" if d > 0 else "negative"
        lines.append(
            f"The participant showed a {mag} {direction} automatic evaluation of {a} "
            f"(D = {d:+.3f})."
        )

    lines.append("")
    lines.append("### What the numbers say")
    lines.append(
        f"- D, the standardised difference in response latency between the two pairings, "
        f"came out at {d:+.3f}. On the conventional scale that is a **{mag}** effect."
    )
    if score.get("d_practice_blocks") is not None and score.get("d_test_blocks") is not None:
        lines.append(
            f"- The practice and test block pairs gave {score['d_practice_blocks']:+.3f} and "
            f"{score['d_test_blocks']:+.3f} respectively; D is their average."
        )
    for blk in (payload.get("blocks") or [])[:4]:
        if blk.get("mean_correct_ms") is not None:
            lines.append(
                f"- {blk['pairing'].title()} {blk['block_role']} block: mean correct latency "
                f"{blk['mean_correct_ms']:.0f} ms across {blk['n_retained']} retained trials, "
                f"error rate {blk['error_rate']:.0%}."
            )

    lines.append("")
    lines.append("### How confident we can be")
    if ci[0] is not None:
        lines.append(
            f"The {int(inf.get('confidence_level', 0.95) * 100)}% bootstrap interval runs from "
            f"{ci[0]:+.3f} to {ci[1]:+.3f}"
            + (", which includes zero." if ci_spans_zero else ", which excludes zero.")
        )
    if p is not None:
        lines.append(
            f"A permutation test over {inf.get('n_resamples')} relabellings gives p = {p:.3f}"
            + (", so an effect this large would be unusual by chance alone."
               if significant else
               ", so an effect this large is not unusual by chance alone.")
        )
    if rel.get("absolute_difference") is not None:
        lines.append(
            f"Split-half diagnostic: odd and even trials give D = "
            f"{rel['d_odd_trials']:+.3f} and {rel['d_even_trials']:+.3f}, a gap of "
            f"{rel['absolute_difference']:.3f}."
            + (" That is stable within the session." if rel.get("stable") else
               " That gap is wide enough to treat the point estimate as unstable.")
        )
    if sess.get("preset_caveat"):
        lines.append(sess["preset_caveat"])

    lines.append("")
    lines.append("### What a researcher would do next")
    if not solid:
        lines.append("- Run the full trial preset; the shortened presets widen the interval "
                     "enough to hide a real effect of this size.")
    lines.append("- Collect a sample rather than a single session. Individual D-scores are "
                 "noisy; the measure is designed to be read at group level.")
    if b:
        lines.append(f"- Vary the comparator. A two-brand IAT is relative, so the result "
                     f"partly reflects the choice of {other}. A single-category run on "
                     f"{a} alone would separate the two.")
    else:
        lines.append("- Add a comparator brand to check whether the evaluation is specific "
                     "to this brand or reflects the category.")
    lines.append("- Pair this with an explicit rating. Where implicit and explicit measures "
                 "diverge is usually where the commercial insight is.")

    lines.append("")
    lines.append("### Caveats")
    lines.append("- This is one participant on one occasion. Nothing here supports a claim "
                 "about consumers in general.")
    lines.append("- A D-score indexes how quickly categories were sorted together. It is not "
                 "a measure of what someone believes, intends, or will buy, and its "
                 "predictive validity for individual behaviour is contested.")
    lines.append(f"- Stimulus onset can only be resolved to within about "
                 f"{sess.get('onset_uncertainty_ms') or 8:.0f} ms on this display. That is "
                 f"a constant offset that cancels in a difference score like D, but it "
                 f"bounds any claim about absolute latency.")

    return "\n".join(lines)
