"""Static figures for the client deliverables.

The dashboard draws its charts in the browser; the workbook and the deck need
the same charts as images. These are drawn with matplotlib from the *same
report dictionary* the dashboard renders, so a figure in the deck can never
show a number the dashboard does not.

The palette is the light-surface counterpart of the dashboard's, and it is
semantic rather than decorative — the same colour means the same thing on every
chart in the deck:

*   **blue / orange** — the direction of an association: blue leans towards
    target A, orange towards target B. Used by the forest plot and heatmap.
*   **aqua / violet** — the two pairings of an RT distribution. Aqua is below
    3:1 contrast on white, so every aqua mark is also named in a legend.
*   **red** — only for participants removed by the cleaning pipeline.

Filled markers mean "clears q < .05 after FDR correction"; hollow means it does
not. Significance is never carried by colour alone.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# ---------------------------------------------------------------- palette

POS = "#2a78d6"        # leans target A
NEG = "#eb6834"        # leans target B
MID = "#f0efec"        # diverging midpoint
CON = "#1baf7a"        # congruent pairing
INC = "#4a3aa7"        # incongruent pairing
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
KEPT = "#6b6a65"
REMOVED = "#d03b3b"
SURFACE = "#ffffff"

DPI = 200


def _font_family() -> list[str]:
    have = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    return [f for f in wanted if f in have] or ["DejaVu Sans"]


plt.rcParams.update({
    "font.family": _font_family(),
    "font.sans-serif": _font_family(),
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buf.getvalue()


def _fmt_d(v: float | None) -> str:
    return "—" if v is None else f"{v:+.2f}"


def _brand(s: str | None) -> str:
    return s.title() if s and s.isupper() else (s or "")


def _rows(report: dict) -> list[dict]:
    """Attribute × (All + level) rows, in the order the tables use."""
    out = []
    for attr in report["attributes"]:
        label = attr.get("pole_a") or attr.get("label") or attr["key"]
        out.append({"attr": attr, "attr_label": label, "level": "All respondents",
                    "cell": attr["overall"], "is_all": True})
        for c in attr["levels"]:
            out.append({"attr": attr, "attr_label": label, "level": c["level"],
                        "cell": c, "is_all": False})
    return out


# ---------------------------------------------------------------- funnel


def funnel(report: dict, *, width: float = 7.2, height: float = 3.6) -> bytes:
    """Horizontal waterfall of the cleaning pipeline, one row per stage."""
    stages = report["funnel"]
    labels, kept, removed = [], [], []
    for s in stages:
        if s["key"] in ("recruited", "final"):
            labels.append(s["label"])
            kept.append(s["n"])
            removed.append(0)
        else:
            labels.append(s["label"])
            kept.append(s["n"])
            removed.append(s["removed"])
    total = stages[0]["n"] or 1

    fig, ax = plt.subplots(figsize=(width, height))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, kept, color=KEPT, height=0.58, zorder=2)
    ax.barh(y, removed, left=kept, color=REMOVED, height=0.58, zorder=2)
    for yi, k, r, s in zip(y, kept, removed, stages):
        if s["key"] in ("recruited", "final"):
            ax.text(k + total * 0.01, yi, f"{k:,}", va="center", ha="left",
                    fontsize=10, color=INK, fontweight="bold")
        elif r:
            ax.text(k + r + total * 0.01, yi, f"−{r:,}", va="center", ha="left",
                    fontsize=9.5, color=REMOVED, fontweight="bold")
        else:
            ax.text(k + total * 0.01, yi, "−0", va="center", ha="left",
                    fontsize=9.5, color=MUTED)
    final = stages[-1]["n"]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, total * 1.12)
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Participants")
    ax.set_title(f"{final:,} of {total:,} recruited participants in the analytic sample "
                 f"({final / total:.0%})", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KEPT, label="Retained"), Patch(color=REMOVED, label="Removed at this stage")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    return _png(fig)


# ---------------------------------------------------------------- distributions


def rt_distribution(report: dict, attribute: str, *, width: float = 9.0,
                    height: float = 3.4) -> bytes:
    """Raw and log-transformed RT density, congruent vs incongruent."""
    dist = (report.get("distributions") or {}).get(attribute)
    attr = next(a for a in report["attributes"] if a["key"] == attribute)
    a_lab, pa, pb = _brand(attr.get("target_a")), attr.get("pole_a"), attr.get("pole_b")
    names = {"congruent": f"{a_lab} + {pa}", "incongruent": f"{a_lab} + {pb}"}
    colours = {"congruent": CON, "incongruent": INC}

    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    for ax, space in zip(axes, ("raw", "log")):
        d = dist[space]
        grid = np.asarray(d["grid"], dtype=float)
        xs = grid if space == "raw" else np.exp(grid)
        for pairing in ("congruent", "incongruent"):
            y = np.asarray(d.get(pairing) or [], dtype=float)
            if y.size:
                ax.plot(xs, y, color=colours[pairing], lw=2, label=names[pairing])
            yn = d.get(f"{pairing}_normal")
            if yn:
                ax.plot(xs, np.asarray(yn, dtype=float), color=colours[pairing], lw=1,
                        ls=(0, (4, 3)), alpha=0.9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True)
        if space == "log":
            ax.set_xscale("log")
            ticks = [200, 300, 500, 800, 1300, 2000, 3000]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(150, 4000)
        else:
            ax.set_xlim(150, 2500)
        sh = dist["shape"]
        sk = sh.get("congruent", {}).get("skew" if space == "raw" else "log_skew")
        sk2 = sh.get("incongruent", {}).get("skew" if space == "raw" else "log_skew")
        title = "Raw latencies" if space == "raw" else "Log-transformed latencies"
        ax.set_title(title)
        if sk is not None and sk2 is not None:
            ax.text(0.98, 0.95, f"skew {sk:.2f} / {sk2:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=INK_2)
        ax.set_xlabel("Response latency (ms)" + (", log scale" if space == "log" else ""))
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
    labels.append("Normal curve, same mean & SD")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _png(fig)


# ---------------------------------------------------------------- forest


def forest(report: dict, method: str, *, width: float = 7.4, height: float | None = None) -> bytes:
    rows = _rows(report)
    n = len(rows)
    n_attr = len(report["attributes"])
    height = height or max(3.0, 0.32 * n + 0.35 * n_attr + 0.6)
    fig, ax = plt.subplots(figsize=(width, height))

    ys = []
    y = 0.0
    prev_attr = None
    for r in rows:
        if prev_attr is not None and r["attr"]["key"] != prev_attr:
            y += 0.6
        prev_attr = r["attr"]["key"]
        ys.append(y)
        y += 1
    ys = np.array(ys)

    lo_all = [r["cell"]["ci_low"] for r in rows if r["cell"].get("ci_low") is not None]
    hi_all = [r["cell"]["ci_high"] for r in rows if r["cell"].get("ci_high") is not None]
    lim = max([0.3] + [abs(v) for v in lo_all + hi_all]) * 1.15

    for yi, r in zip(ys, rows):
        c = r["cell"]
        m = c.get("mean_d")
        if m is None:
            continue
        col = POS if m >= 0 else NEG
        sig = bool(c.get(method, {}).get("sig"))
        if c.get("ci_low") is not None:
            ax.plot([c["ci_low"], c["ci_high"]], [yi, yi], color=col, lw=2 if r["is_all"] else 1.5,
                    solid_capstyle="round", zorder=2)
        ax.scatter([m], [yi], s=70 if r["is_all"] else 48, zorder=3,
                   facecolor=col if sig else SURFACE, edgecolor=col, linewidth=1.6)
        ax.text(lim * 1.02, yi, _fmt_d(m) + ("  low base" if c.get("low_base") else ""),
                va="center", ha="left", fontsize=8.5,
                color=INK if r["is_all"] else INK_2)

    tick_labels = []
    for r in rows:
        c = r["cell"]
        tick_labels.append(f"{r['attr_label']} — all (n={c['n']})" if r["is_all"]
                           else f"{r['level']} (n={c['n']})")
    ax.set_yticks(ys, tick_labels)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r["is_all"]:
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    attr0 = report["attributes"][0] if report["attributes"] else {}
    a, b = _brand(attr0.get("target_a")), _brand(attr0.get("target_b"))
    ax.set_xlabel(f"Mean D with 95% bootstrap CI   ←  leans {b}   ·   leans {a}  →")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, label="q < .05 (FDR)"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, label="Not significant"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.12 * 3.6 / height - 0.04), ncol=2, fontsize=8.5)
    return _png(fig)


# ---------------------------------------------------------------- heatmap


def heatmap(report: dict, method: str, *, width: float = 7.4, height: float | None = None) -> bytes:
    attrs = report["attributes"]
    cols = ["All"] + list(report.get("levels") or [])
    vals = np.full((len(attrs), len(cols)), np.nan)
    sig = np.zeros_like(vals, dtype=bool)
    low = np.zeros_like(vals, dtype=bool)
    for i, a in enumerate(attrs):
        cells = [a["overall"]] + list(a["levels"])
        for j, c in enumerate(cells[: len(cols)]):
            if c.get("mean_d") is not None:
                vals[i, j] = c["mean_d"]
            sig[i, j] = bool(c.get(method, {}).get("sig"))
            low[i, j] = bool(c.get("low_base"))
    lim = max(0.35, float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.35)
    cmap = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    height = height or max(2.2, 0.62 * len(attrs) + 1.3)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(np.nan_to_num(vals), cmap=cmap, norm=norm, aspect="auto")
    # White gutters between cells.
    for x in np.arange(-0.5, len(cols), 1):
        ax.axvline(x, color=SURFACE, lw=3)
    for y in np.arange(-0.5, len(attrs), 1):
        ax.axhline(y, color=SURFACE, lw=3)
    ax.axvline(0.5, color=SURFACE, lw=7)
    for i in range(len(attrs)):
        for j in range(len(cols)):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                continue
            strong = abs(v) / lim > 0.55
            txt = f"{v:+.2f}" + (" ●" if sig[i, j] else " ○")
            if low[i, j]:
                txt += "\nlow base"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5 if not low[i, j] else 8.5,
                    color=SURFACE if strong else INK,
                    fontweight="bold" if sig[i, j] else "normal")
    seg = report.get("segment_by")
    seg_label = (report["batch"].get("segments", {}).get(seg) or {}).get("label", seg or "")
    ax.set_xticks(range(len(cols)), [c if c != "All" else "All respondents" for c in cols])
    ax.set_yticks(range(len(attrs)), [a.get("pole_a") or a["key"] for a in attrs])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    a0 = attrs[0] if attrs else {}
    ax.set_xlabel(f"Columns: {seg_label}   ·   blue leans {_brand(a0.get('target_a'))}, "
                  f"orange leans {_brand(a0.get('target_b'))}   ·   ● q < .05 (FDR)  ○ not significant",
                  fontsize=8.5, color=INK_2, labelpad=8)
    return _png(fig)


# ---------------------------------------------------------------- D spread


def d_histograms(report: dict, *, width: float = 9.0, height: float = 2.4) -> bytes:
    attrs = report["attributes"]
    fig, axes = plt.subplots(1, len(attrs), figsize=(width, height), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], attrs):
        h = report["d_histograms"][a["key"]]
        edges = np.asarray(h["edges"], dtype=float)
        counts = np.asarray(h["counts"], dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(centres, counts, width=0.085, color=[POS if c >= 0 else NEG for c in centres])
        ax.axvline(0, color=INK_2, lw=0.8)
        m = a["overall"].get("mean_d")
        ax.set_title(f"{a.get('pole_a') or a['key']}  (mean {_fmt_d(m)})", fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Participant D")
    fig.tight_layout()
    return _png(fig)

