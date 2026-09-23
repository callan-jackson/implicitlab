"""Executive takeaways, written from the statistics by rules, not by a model.

The single-session report uses a language model for its prose and verifies
every number it emits. For a client deliverable the trade-off goes the other
way: a deck goes to a client under the agency's name, the claims in it are a
small, fixed set of shapes ("X is ahead on Y", "the gap is larger among Z",
"no reliable difference"), and those shapes are exactly what a template does
well and a model occasionally does creatively. So the takeaways are built here,
from the same numbers the tables show, and they only ever say three things
about a result: which way it points, whether it clears the corrected
significance bar, and how big it is on the published bands.

Wording rules, all enforced here rather than left to the reader:

*   A result that does not clear q < .05 is never described with a direction
    as though it were a finding.
*   D is relative. "Aurelia is more associated with Premium than Northvane"
    is what a two-target IAT can support; "Aurelia is premium" is not.
*   Low-base cells are called indicative, every time.
"""

from __future__ import annotations


def _brand(s: str) -> str:
    return s.title() if s and s.isupper() else (s or "")


def _fmt_d(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}"


def _fmt_q(q: float | None) -> str:
    if q is None:
        return "q n/a"
    return "q < .001" if q < 0.001 else f"q = {q:.3f}".replace("0.", ".")


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def association_sentence(attr: dict, cell: dict, method: str) -> tuple[str, str]:
    """Headline + detail for one attribute's overall result."""
    a, b = _brand(attr.get("target_a", "Target A")), _brand(attr.get("target_b", "Target B"))
    pole = attr.get("pole_a", attr.get("label", "the attribute"))
    res = cell[method]
    d = cell.get("mean_d")
    ci = f"95% CI [{_fmt_d(cell.get('ci_low'))}, {_fmt_d(cell.get('ci_high'))}]"
    if res.get("sig") and d is not None:
        lead, other = (a, b) if d > 0 else (b, a)
        share = _pct(cell["pct_positive"] if d > 0 else 1 - cell["pct_positive"])
        if cell["magnitude"] == "negligible":
            # Statistically reliable is not the same as commercially meaningful.
            # With 450 respondents a very small gap clears the bar, and the
            # deck should say it is small rather than let "significant" imply big.
            return f"{lead} edges {pole}", (
                f"{lead} is slightly more associated with {pole} than {other} "
                f"(mean D = {_fmt_d(d)}, {ci}, {_fmt_q(res.get('q'))}). The gap is "
                f"statistically reliable but below the 0.15 band conventionally "
                f"called slight — real, and small. {share} of respondents lean that way."
            )
        head = f"{lead} owns {pole}"
        detail = (
            f"{lead} is more strongly associated with {pole} than {other} "
            f"(mean D = {_fmt_d(d)}, {ci}, {_fmt_q(res.get('q'))}; a {cell['magnitude']} "
            f"effect). {share} of respondents lean the same way."
        )
        return head, detail
    head = f"No clear winner on {pole}"
    detail = (
        f"Neither brand holds a reliable implicit advantage on {pole} "
        f"(mean D = {_fmt_d(d)}, {ci}, {_fmt_q(res.get('q'))}). The interval includes zero "
        f"or the result does not survive correction for multiple comparisons."
    )
    return head, detail


def build_takeaways(report: dict, method: str) -> list[dict]:
    out: list[dict] = []
    sample = report["sample"]
    funnel = {s["key"]: s for s in report["funnel"]}
    removed = sample["recruited"] - sample["included"]
    parts = [
        f"{funnel[k]['removed']} {why}"
        for k, why in (
            ("fast", "for fast responding"),
            ("inaccurate", "for low accuracy"),
            ("incomplete", "with incomplete data"),
            ("insufficient", "for too few trials after trimming"),
        )
        if funnel.get(k, {}).get("removed")
    ]
    out.append({
        "kind": "sample",
        "headline": f"{sample['included']:,} of {sample['recruited']:,} respondents analysed",
        "detail": (
            f"{_pct(sample['included_pct'])} of the recruited sample met every quality criterion. "
            + (f"{removed} were excluded: {', '.join(parts)}." if parts else "Nobody was excluded.")
        ),
        "tone": "neutral",
    })

    for attr in report["attributes"]:
        head, detail = association_sentence(attr, attr["overall"], method)
        out.append({
            "kind": "association",
            "attribute": attr["key"],
            "headline": head,
            "detail": detail,
            "tone": "positive" if attr["overall"][method].get("sig") else "neutral",
        })

    seg = report.get("segment_by")
    seg_label = (report["batch"].get("segments", {}).get(seg, {}) or {}).get("label", seg)
    sig = [c for c in report["comparisons"] if c[method].get("sig")]
    sig.sort(key=lambda c: -abs(c.get("hedges_g") or 0))
    attrs = {a["key"]: a for a in report["attributes"]}
    for c in sig[:3]:
        attr = attrs[c["attribute"]]
        pole = attr.get("pole_a", c["attribute"])
        a_brand = _brand(attr.get("target_a", "Target A"))
        hi, lo = (c["level_a"], c["level_b"]) if (c["diff"] or 0) > 0 else (c["level_b"], c["level_a"])
        low = " Low base — indicative only." if c["low_base"] else ""
        out.append({
            "kind": "segment",
            "attribute": c["attribute"],
            "headline": f"{seg_label} moves {pole}",
            "detail": (
                f"The {a_brand}–{pole} association is stronger among {hi} than {lo} "
                f"respondents (D {_fmt_d(c['mean_a'])} vs {_fmt_d(c['mean_b'])}; difference "
                f"{_fmt_d(c['diff'])}, 95% CI [{_fmt_d(c['ci_low'])}, {_fmt_d(c['ci_high'])}], "
                f"{_fmt_q(c[method].get('q'))}; Hedges' g = {c['hedges_g']:.2f}).{low}"
            ),
            "tone": "positive",
        })
    if seg and not sig and report["comparisons"]:
        out.append({
            "kind": "segment",
            "headline": f"No reliable differences by {seg_label.lower()}",
            "detail": (
                f"None of the {len(report['comparisons'])} {seg_label.lower()} comparisons "
                f"survives correction for multiple comparisons. The segments read the brands "
                f"the same way on every attribute tested."
            ),
            "tone": "neutral",
        })

    ag = report["agreement"]
    other = "parametric" if method == "permutation" else "permutation"
    if ag["n_agree"] == ag["n_tests"]:
        robust = (f"All {ag['n_tests']} significance calls are the same under the {other} "
                  f"tests, so no conclusion here depends on the choice of test.")
    else:
        k = ag["n_tests"] - ag["n_agree"]
        robust = (f"{k} of {ag['n_tests']} significance calls change under the {other} tests. "
                  f"Treat those as borderline: they are flagged in the workbook.")
    out.append({"kind": "robustness", "headline": "Robust to the choice of test"
                if ag["n_agree"] == ag["n_tests"] else "Some calls are borderline",
                "detail": robust, "tone": "neutral"})
    return out
