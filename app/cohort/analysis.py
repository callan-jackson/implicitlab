"""Cohort orchestration: (batch, parameters) in, the complete report out.

This is the single function behind the Studio dashboard, the Excel workbook and
the PowerPoint deck. They all call it with the same arguments and render what
it returns, so a number cannot differ between the screen and the deliverable —
there is only one place it is computed.
"""

from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np
from scipy import stats as sps

from ..scoring.dscore import MODERATE, SLIGHT, STRONG, THRESHOLD_NOTE
from . import stats as st
from .scoring import STAGES, CohortParams, Prepared, Scored, score
from .takeaways import build_takeaways

METHODS = ("permutation", "parametric")


def clean(obj):
    """Recursively replace NaN/inf with None and numpy scalars with Python ones."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def magnitude(d: float | None) -> str:
    if d is None or not math.isfinite(d):
        return "none"
    a = abs(d)
    if a < SLIGHT:
        return "negligible"
    if a < MODERATE:
        return "slight"
    if a < STRONG:
        return "moderate"
    return "strong"


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


def _cells(D: np.ndarray, L: np.ndarray, label: str) -> list[dict]:
    """Stats for one group of participants across every attribute at once."""
    n, k = D.shape
    rng_b = st.rng_for("boot", label)
    rng_p = st.rng_for("signflip", label)
    lo, hi = st.bootstrap_mean_ci(D, rng_b)
    p_perm = st.signflip_p(D, rng_p)
    t, p_par, df = st.ttest_one(L)
    out = []
    for j in range(k):
        d = D[:, j]
        mean = float(d.mean()) if n else float("nan")
        out.append({
            "n": n,
            "low_base": n < st.LOW_BASE,
            "mean_d": mean,
            "sd_d": float(d.std(ddof=1)) if n > 1 else None,
            "median_d": float(np.median(d)) if n else None,
            "ci_low": lo[j], "ci_high": hi[j],
            "pct_positive": float((d > 0).mean()) if n else None,
            # Geometric-mean slowdown in the incongruent pairing: a
            # client-readable restatement of the log-RT difference.
            "slowdown_pct": float(np.expm1(L[:, j].mean())) if n else None,
            "magnitude": magnitude(mean),
            "parametric": {"test": "one-sample t on log-RT difference",
                           "t": t[j], "df": df, "p": p_par[j]},
            "permutation": {"test": f"sign-flip permutation on D ({st.N_PERM:,} draws)",
                            "p": p_perm[j]},
        })
    return out


def _flag(families: list[list[dict]]) -> None:
    """Attach BH q-values and significance flags within each family, per method."""
    for family in families:
        for method in METHODS:
            qs = st.bh_fdr([c[method]["p"] for c in family])
            for c, q in zip(family, qs):
                c[method]["q"] = q
                c[method]["sig"] = bool(q is not None and q < st.ALPHA)
        for c in family:
            c["agree"] = c["parametric"]["sig"] == c["permutation"]["sig"]


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


def _binned_kde(x: np.ndarray, lo: float, hi: float, n_grid: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE (Silverman bandwidth) via a fine histogram and convolution.

    Exact enough for display and O(bins) rather than O(n × grid), which is the
    difference between 2 ms and 200 ms on 30,000 latencies.
    """
    grid = np.linspace(lo, hi, n_grid)
    if x.size < 3:
        return grid, np.zeros_like(grid)
    sd = x.std(ddof=1)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    bw = 0.9 * spread * x.size ** (-1 / 5)
    nb = 2048
    edges = np.linspace(lo, hi, nb + 1)
    counts, _ = np.histogram(x, bins=edges)
    dx = edges[1] - edges[0]
    m = int(np.ceil(4 * bw / dx))
    kern = np.exp(-0.5 * ((np.arange(-m, m + 1) * dx) / bw) ** 2)
    dens = np.convolve(counts, kern, mode="same") / (x.size * bw * np.sqrt(2 * np.pi))
    centres = (edges[:-1] + edges[1:]) / 2
    return grid, np.interp(grid, centres, dens)


def _shape(x: np.ndarray) -> dict:
    if x.size < 3:
        return {"n": int(x.size)}
    lx = np.log(x)
    return {
        "n": int(x.size),
        "mean_ms": float(x.mean()), "median_ms": float(np.median(x)),
        "sd_ms": float(x.std(ddof=1)),
        "skew": float(sps.skew(x, bias=False)),
        "excess_kurtosis": float(sps.kurtosis(x, bias=False)),
        "log_skew": float(sps.skew(lx, bias=False)),
        "log_excess_kurtosis": float(sps.kurtosis(lx, bias=False)),
        # Share of trials beyond mean + 2 SD. A normal distribution puts 2.3%
        # there; a right-skewed RT distribution puts noticeably more.
        "beyond_2sd": float((x > x.mean() + 2 * x.std(ddof=1)).mean()),
    }


def _distributions(prep: Prepared, sc: Scored) -> dict:
    """Correct-trial latency distributions by pairing, for every attribute."""
    p = sc.params
    inc_rows = sc.included[prep.p]
    ok = inc_rows & prep.correct & ~prep.timed_out & np.isfinite(prep.first_lat)
    with np.errstate(invalid="ignore"):
        ok &= prep.first_lat <= p.upper_ms
        if p.lower_ms > 0:
            ok &= prep.first_lat >= p.lower_ms
    out = {}
    for ai, attr in enumerate(prep.attrs):
        entry = {"raw": {}, "log": {}, "shape": {}}
        for pi, pairing in enumerate(("congruent", "incongruent")):
            x = prep.first_lat[ok & (prep.a == ai) & (prep.pair == pi)]
            g, dens = _binned_kde(x, 150.0, 2500.0)
            lg, ldens = _binned_kde(np.log(x), math.log(150.0), math.log(4000.0))
            entry["raw"]["grid"] = g
            entry["raw"][pairing] = dens
            entry["log"]["grid"] = lg
            entry["log"][pairing] = ldens
            if x.size > 2:
                entry["raw"][f"{pairing}_normal"] = sps.norm.pdf(g, x.mean(), x.std(ddof=1))
                lx = np.log(x)
                entry["log"][f"{pairing}_normal"] = sps.norm.pdf(lg, lx.mean(), lx.std(ddof=1))
            entry["shape"][pairing] = _shape(x)
        out[attr] = entry
    return out


def _d_histograms(prep: Prepared, sc: Scored) -> dict:
    edges = np.round(np.arange(-1.6, 1.6001, 0.1), 2)
    out = {}
    for ai, attr in enumerate(prep.attrs):
        d = sc.d[sc.included, ai]
        counts, _ = np.histogram(np.clip(d, -1.5999, 1.5999), bins=edges)
        out[attr] = {"edges": edges, "counts": counts}
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def analyse_cohort(
    prep: Prepared,
    meta: dict,
    params: CohortParams | None = None,
    *,
    segment_by: str | None = None,
    include_distributions: bool = True,
) -> dict:
    t0 = time.perf_counter()
    params = params or CohortParams()
    sc = score(prep, params)

    segments = meta.get("segments", {})
    if segment_by not in segments:
        segment_by = next(iter(segments), None)

    inc = sc.included
    D = sc.d[inc]
    L = sc.dlog[inc]

    # ---- attribute × (overall + segment level) cells -----------------------
    overall = _cells(D, L, "overall")
    level_cells: dict[str, list[dict]] = {}
    levels: list[str] = []
    level_masks: dict[str, np.ndarray] = {}
    if segment_by:
        col = prep.segments[segment_by].astype(str).to_numpy()[inc]
        present = set(col)
        levels = [lv for lv in segments[segment_by]["levels"] if lv in present]
        levels += sorted(lv for lv in present if lv not in levels and lv not in ("nan", "<NA>"))
        for lv in levels:
            m = col == lv
            level_masks[lv] = m
            level_cells[lv] = _cells(D[m], L[m], f"{segment_by}={lv}")

    attributes = []
    association_family: list[dict] = []
    for j, attr in enumerate(prep.attrs):
        info = meta.get("attributes", {}).get(attr, {})
        ov = overall[j]
        ov["level"] = None
        association_family.append(ov)
        lvl = []
        for lv in levels:
            c = level_cells[lv][j]
            c["level"] = lv
            lvl.append(c)
            association_family.append(c)
        attributes.append({"key": attr, **info, "overall": ov, "levels": lvl})

    # ---- segment comparisons -----------------------------------------------
    comparisons: list[dict] = []
    if segment_by and len(levels) >= 2:
        if len(levels) <= 4:
            pairs = [((a,), (b,)) for a, b in combinations(levels, 2)]
        else:
            # Beyond four levels, every-pair comparison explodes; each level is
            # compared against the rest of the sample instead.
            pairs = [((a,), tuple(x for x in levels if x != a)) for a in levels]
        for ga, gb in pairs:
            ma = np.any([level_masks[x] for x in ga], axis=0)
            mb = np.any([level_masks[x] for x in gb], axis=0)
            la = ga[0]
            lb = gb[0] if len(gb) == 1 else "rest of sample"
            tag = f"{segment_by}:{la}|{lb}"
            Da, Db, La, Lb = D[ma], D[mb], L[ma], L[mb]
            lo, hi = st.bootstrap_diff_ci(Da, Db, st.rng_for("bootdiff", tag))
            p_perm = st.perm_diff_p(Da, Db, st.rng_for("perm", tag))
            t, p_par, df = st.welch(La, Lb)
            g = st.hedges_g(Da, Db)
            for j, attr in enumerate(prep.attrs):
                comparisons.append({
                    "attribute": attr,
                    "variable": segment_by,
                    "level_a": la, "level_b": lb,
                    "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                    "mean_a": float(Da[:, j].mean()) if len(Da) else None,
                    "mean_b": float(Db[:, j].mean()) if len(Db) else None,
                    "diff": float(Da[:, j].mean() - Db[:, j].mean()) if len(Da) and len(Db) else None,
                    "ci_low": lo[j], "ci_high": hi[j],
                    "hedges_g": g[j],
                    "low_base": bool(ma.sum() < st.LOW_BASE or mb.sum() < st.LOW_BASE),
                    "parametric": {"test": "Welch t on log-RT difference",
                                   "t": t[j], "df": df[j] if np.ndim(df) else df, "p": p_par[j]},
                    "permutation": {"test": f"label permutation on D ({st.N_PERM:,} draws)",
                                    "p": p_perm[j]},
                })

    _flag([association_family, comparisons])

    all_tests = association_family + comparisons
    disagreements = [c for c in all_tests if not c["agree"]]
    agreement = {
        "n_tests": len(all_tests),
        "n_agree": len(all_tests) - len(disagreements),
        "note": (
            "Significance is judged on Benjamini–Hochberg q < .05 within each family "
            "(associations; segment differences). Parametric = t-tests on log-transformed "
            "RTs; permutation = distribution-free tests on D."
        ),
    }

    by_segment = {}
    for var in segments:
        col = prep.segments[var].astype(str)
        by_segment[var] = {
            "recruited": col.value_counts().to_dict(),
            "included": col[inc].value_counts().to_dict(),
        }

    report = {
        "batch": {k: v for k, v in meta.items() if k not in ("truth",)},
        "params": params.to_dict(),
        "defaults": CohortParams().to_dict(),
        "is_default": params.is_default(),
        "segment_by": segment_by,
        "levels": levels,
        "funnel": sc.funnel,
        "trimming": sc.trimming,
        "sample": {
            "recruited": int(prep.n_participants),
            "included": int(inc.sum()),
            "included_pct": float(inc.mean()) if prep.n_participants else 0.0,
            "trial_rows": int(prep.n_rows_total),
            "by_segment": by_segment,
        },
        "attributes": attributes,
        "comparisons": comparisons,
        "agreement": agreement,
        "d_histograms": _d_histograms(prep, sc),
        "distributions": _distributions(prep, sc) if include_distributions else None,
        "interpretation_note": THRESHOLD_NOTE,
        "method_note": {
            "unit": "Participant. Each respondent contributes one D per attribute.",
            "ci": f"Percentile bootstrap over participants, {st.N_BOOT:,} resamples.",
            "parametric": "t-tests on the per-participant difference in mean log RT "
                          "(incongruent − congruent); Welch's test between segments.",
            "permutation": f"Sign-flip (one-sample) and label-shuffle (two-group) "
                           f"permutation tests on D, {st.N_PERM:,} draws, add-one corrected.",
            "fdr": "Benjamini–Hochberg within each family; flags use q < .05.",
            "low_base": f"Cells with fewer than {st.LOW_BASE} participants are flagged "
                        f"as low base — indicative only.",
        },
    }
    report["takeaways"] = {m: build_takeaways(report, m) for m in METHODS}
    report["compute_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return clean(report)


def participant_table(prep: Prepared, meta: dict, params: CohortParams):
    """One row per participant: segment variables, screening outcome, D per attribute."""
    import pandas as pd

    sc = score(prep, params)
    labels = {key: label for key, label, _ in STAGES}
    df = prep.segments.copy()
    df.insert(0, "status", ["Included" if s == "included" else f"Excluded — {labels[s]}"
                            for s in sc.status])
    df.insert(1, "accuracy", np.round(sc.accuracy, 4))
    df.insert(2, "fast_trial_share", np.round(sc.fast_prop, 4))
    for j, attr in enumerate(prep.attrs):
        lab = meta.get("attributes", {}).get(attr, {}).get("label", attr)
        df[f"D {lab}"] = np.round(sc.d[:, j], 4)
        df[f"D practice {lab}"] = np.round(sc.d_practice[:, j], 4)
        df[f"D test {lab}"] = np.round(sc.d_test[:, j], 4)
        df[f"log-RT diff {lab}"] = np.round(sc.dlog[:, j], 5)
        df[f"Trials analysed {lab}"] = sc.n_trials_kept[:, j]
    df.index.name = "participant_id"
    return df.reset_index(), sc
