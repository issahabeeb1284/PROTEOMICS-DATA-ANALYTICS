#!/usr/bin/env python3
"""
Precision proteomics visualization pipeline (anonymized)
========================================================

Builds a publication-grade, multi-panel graphical abstract from a label-free
quantitative proteomics experiment with several genotypes and biological
replicates. Every genotype is shown only as "Sample A", "Sample B",
"Sample C", ...: Sample A is the reference (control) and every other sample
is compared against it. Protein and pathway identifiers are replaced with
neutral codes (Protein_0001, Pathway 1, ...) so no gene or sample name is shown.

Panels
  a  PCA of all replicates (sample-level reproducibility)
  b  Volcano plot for each sample vs Sample A (shared axes)
  c  Up/down differential-abundance counts (diverging bars)
  d  UpSet-style overlap of the differentially abundant protein sets
  e  Clustered z-score heatmap of the top differentially abundant proteins
  f  Pathway over-representation dot plot (hypergeometric test, BH-FDR)

Analysis steps
  log2 transform (if needed) -> valid-value filter -> median normalization ->
  left-shifted normal imputation (MNAR) -> Welch t-test -> Benjamini-Hochberg FDR

Usage
  python proteomics_graphical_abstract.py                     # simulated demo data
  python proteomics_graphical_abstract.py --input matrix.csv  # your own data
        [--pathways pathways.csv] [--reference "<column group>"] [--outdir out]

Input matrix format (CSV): the first column holds protein IDs; every other
column is one replicate, named "<group>_R<n>" (e.g. WT_R1, WT_R2, KO1_R1, ...).
Groups are relabeled Sample A, B, C, ... in column order (the reference group
first), so the original names never reach a figure or a table.
Pathway file (optional, CSV): two columns, Pathway and Protein.

Requirements: numpy, pandas, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import re
import string
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
FDR_THRESH = 0.05          # Benjamini-Hochberg adjusted p-value
LOG2FC_THRESH = 1.0        # |log2 fold change|
MIN_VALID_FRACTION = 0.70  # valid values required in at least one sample
HEATMAP_TOP_N = 60         # max proteins in the heatmap
SEED = 42

# ---------------------------------------------------------------------------
# Visual theme: validated categorical palette, fixed order, never cycled
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NS_COLOR = "#c3c2b7"
UP_COLOR = "#e34948"
DOWN_COLOR = "#2a78d6"
REFERENCE_COLOR = "#52514e"   # Sample A (reference) is always neutral gray
SAMPLE_HUES = ["#2a78d6", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300", "#eb6834"]

DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8a2323"], N=256)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"], N=256)

MM = 1 / 25.4


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def sample_colors(samples: list[str]) -> dict[str, str]:
    """Reference is gray; the other samples take hues in fixed order."""
    colors = {samples[0]: REFERENCE_COLOR}
    for i, s in enumerate(samples[1:]):
        colors[s] = SAMPLE_HUES[i % len(SAMPLE_HUES)]
    return colors


# ---------------------------------------------------------------------------
# Data: simulated demo dataset or user matrix (always anonymized)
# ---------------------------------------------------------------------------
def simulate_dataset(n_proteins=3000, n_samples=4, n_reps=4, seed=SEED):
    """Realistic LFQ-like matrix with shared and sample-specific remodeling."""
    rng = np.random.default_rng(seed)
    samples = [f"Sample {string.ascii_uppercase[i]}" for i in range(n_samples)]
    proteins = np.array([f"Protein_{i:04d}" for i in range(1, n_proteins + 1)])
    base = rng.normal(22.0, 2.0, n_proteins)

    # Pathway gene sets (overlapping, varied size)
    pathways = {f"Pathway {k + 1}": rng.choice(n_proteins, rng.integers(60, 160), replace=False)
                for k in range(10)}

    effects = {s: np.zeros(n_proteins) for s in samples}
    # Shared core response (all non-reference samples, same direction)
    core = rng.choice(np.concatenate([pathways["Pathway 1"], pathways["Pathway 2"]]), 110, replace=False)
    core_fc = rng.choice([-1, 1], core.size, p=[0.45, 0.55]) * rng.normal(1.9, 0.45, core.size)
    for s in samples[1:]:
        effects[s][core] = core_fc * rng.normal(1.0, 0.15, core.size)
    # Partially shared and sample-specific responses
    for i, s in enumerate(samples[1:]):
        own_path = pathways[f"Pathway {i + 3}"]
        specific = np.union1d(rng.choice(own_path, min(70, own_path.size), replace=False),
                              rng.choice(n_proteins, 90, replace=False))
        effects[s][specific] += rng.choice([-1, 1], specific.size) * rng.normal(1.7, 0.5, specific.size)
    pair = rng.choice(pathways["Pathway 6"], 45, replace=False)
    for s in samples[1:3]:
        effects[s][pair] += rng.normal(-1.8, 0.4, pair.size)

    cols, data = [], []
    for s in samples:
        for r in range(1, n_reps + 1):
            col = base + effects[s] + rng.normal(0, 0.35, n_proteins) + rng.normal(0, 0.05)
            cols.append(f"{s}_R{r}")
            data.append(col)
    mat = pd.DataFrame(np.column_stack(data), index=proteins, columns=cols)

    # Missing values: intensity-dependent (MNAR) + a few at random (MCAR)
    low = mat.values < np.quantile(mat.values, 0.05)
    mnar = low & (rng.random(mat.shape) < 0.6)
    mcar = rng.random(mat.shape) < 0.01
    mat = mat.mask(mnar | mcar)

    pw = pd.DataFrame([(name, proteins[i]) for name, idx in pathways.items() for i in idx],
                      columns=["Pathway", "Protein"])
    return mat, pw


def load_dataset(path: Path, reference: str | None, pathway_path: Path | None,
                 keep_pathway_names: bool):
    """Read a user matrix and relabel groups, proteins and pathways."""
    raw = pd.read_csv(path, index_col=0)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    groups = [re.sub(r"_R?\d+$", "", c) for c in raw.columns]
    order = list(dict.fromkeys(groups))
    if reference:
        if reference not in order:
            raise SystemExit(f"--reference group not found among {len(order)} column groups")
        order.remove(reference)
        order.insert(0, reference)
    letter = {g: f"Sample {string.ascii_uppercase[i]}" for i, g in enumerate(order)}
    rep_counter: dict[str, int] = {}
    new_cols = []
    for g in groups:
        rep_counter[g] = rep_counter.get(g, 0) + 1
        new_cols.append(f"{letter[g]}_R{rep_counter[g]}")
    raw.columns = new_cols
    raw = raw[[c for s in letter.values() for c in new_cols if c.startswith(s + "_")]]
    print("Group -> label assignment (console only, never written to disk):")
    for g, s in letter.items():
        print(f"   {s}  <-  column group #{order.index(g) + 1}")

    protein_key = {p: f"Protein_{i:04d}" for i, p in enumerate(raw.index, start=1)}
    raw.index = raw.index.map(protein_key)

    pw = None
    if pathway_path is not None:
        pw = pd.read_csv(pathway_path).iloc[:, :2]
        pw.columns = ["Pathway", "Protein"]
        pw["Protein"] = pw["Protein"].map(protein_key)
        pw = pw.dropna()
        if not keep_pathway_names:
            names = {n: f"Pathway {i}" for i, n in enumerate(pw["Pathway"].unique(), start=1)}
            pw["Pathway"] = pw["Pathway"].map(names)
    return raw, pw


def design_of(mat: pd.DataFrame) -> pd.Series:
    return pd.Series([c.rsplit("_R", 1)[0] for c in mat.columns], index=mat.columns)


# ---------------------------------------------------------------------------
# Processing and statistics
# ---------------------------------------------------------------------------
def preprocess(mat: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = mat.copy()
    if np.nanmax(x.values) > 100:            # raw intensities -> log2
        x = np.log2(x.where(x > 0))
    design = design_of(x)
    valid = x.notna().T.groupby(design).mean().T
    x = x.loc[(valid >= MIN_VALID_FRACTION).any(axis=1)]
    med = x.median()
    x = x - med + med.mean()                 # median normalization
    for c in x.columns:                      # left-shifted normal imputation
        col = x[c]
        miss = col.isna()
        if miss.any():
            mu, sd = col.mean() - 1.8 * col.std(), 0.3 * col.std()
            x.loc[miss, c] = rng.normal(mu, sd, miss.sum())
    print(f"Proteins after valid-value filter: {len(x)} / {len(mat)}")
    return x


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def differential(x: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    design = design_of(x)
    ref = x.loc[:, design == samples[0]]
    rows = []
    for s in samples[1:]:
        test = x.loc[:, design == s]
        t, p = stats.ttest_ind(test, ref, axis=1, equal_var=False)
        df = pd.DataFrame({"Protein": x.index, "Sample": s,
                           "log2FC": test.mean(axis=1).values - ref.mean(axis=1).values,
                           "pvalue": p, "adj_pvalue": bh_fdr(p)})
        df["neglog10_adjp"] = -np.log10(df["adj_pvalue"].clip(lower=1e-300))
        df["call"] = np.select(
            [(df.adj_pvalue < FDR_THRESH) & (df.log2FC > LOG2FC_THRESH),
             (df.adj_pvalue < FDR_THRESH) & (df.log2FC < -LOG2FC_THRESH)],
            ["up", "down"], "ns")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def enrichment(de: pd.DataFrame, pw: pd.DataFrame, universe: pd.Index) -> pd.DataFrame:
    pw = pw[pw["Protein"].isin(universe)].drop_duplicates()
    sets = pw.groupby("Pathway")["Protein"].apply(set)
    sets = sets[sets.apply(len) >= 10]
    N = len(universe)
    rows = []
    for s, sub in de.groupby("Sample", sort=False):
        hits = set(sub.loc[sub["call"] != "ns", "Protein"])
        n = len(hits)
        for name, members in sets.items():
            k = len(hits & members)
            K = len(members)
            p = stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
            rows.append({"Sample": s, "Pathway": name, "overlap": k, "set_size": K,
                         "fold_enrichment": (k / n) / (K / N) if n else 0, "pvalue": p})
    res = pd.DataFrame(rows)
    res["FDR"] = res.groupby("Sample")["pvalue"].transform(bh_fdr)
    return res


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 10),
                textcoords="offset points", fontsize=9, fontweight="bold",
                va="bottom", ha="left")


def plot_pca(ax, x, samples, colors):
    design = design_of(x)
    X = x.T.values - x.T.values.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = U * S
    var = S ** 2 / np.sum(S ** 2) * 100
    for s in samples:
        m = (design == s).values
        ax.scatter(pcs[m, 0], pcs[m, 1], s=26, c=colors[s], edgecolors="white",
                   linewidths=0.8, label=s, zorder=3)
        cx, cy = pcs[m, 0].mean(), pcs[m, 1].mean()
        ax.annotate(s.replace("Sample ", ""), (cx, cy), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold", color=INK)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title("Replicate structure (PCA)")
    ax.grid(True, zorder=0)
    ax.margins(0.18)


def plot_volcano(ax, sub, title_color, xlim, ylim, show_ylabel):
    for call, c, sz, a in [("ns", NS_COLOR, 2.5, 0.55), ("down", DOWN_COLOR, 4, 0.9),
                           ("up", UP_COLOR, 4, 0.9)]:
        d = sub[sub["call"] == call]
        ax.scatter(d["log2FC"], d["neglog10_adjp"], s=sz, c=c, alpha=a, linewidths=0,
                   rasterized=True)
    for v in (-LOG2FC_THRESH, LOG2FC_THRESH):
        ax.axvline(v, color=MUTED, lw=0.5, ls="--")
    ax.axhline(-np.log10(FDR_THRESH), color=MUTED, lw=0.5, ls="--")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, ylim)
    n_up, n_dn = (sub["call"] == "up").sum(), (sub["call"] == "down").sum()
    s = sub["Sample"].iloc[0]
    ax.set_title(f"{s} vs Sample A", color=INK)
    ax.text(0.04, 0.96, f"▼ {n_dn}", transform=ax.transAxes, color=INK2, fontsize=6.5, va="top")
    ax.text(0.96, 0.96, f"{n_up} ▲", transform=ax.transAxes, color=INK2, fontsize=6.5,
            va="top", ha="right")
    ax.set_xlabel("log$_2$FC")
    if show_ylabel:
        ax.set_ylabel("−log$_{10}$(FDR)")
    else:
        ax.tick_params(labelleft=False)


def plot_counts(ax, de, samples):
    others = samples[1:]
    up = [((de.Sample == s) & (de.call == "up")).sum() for s in others]
    dn = [((de.Sample == s) & (de.call == "down")).sum() for s in others]
    y = np.arange(len(others))[::-1]
    ax.barh(y, up, height=0.62, color=UP_COLOR, label="Up vs A")
    ax.barh(y, [-d for d in dn], height=0.62, color=DOWN_COLOR, label="Down vs A")
    m = max(max(up), max(dn))
    for yi, u, d in zip(y, up, dn):
        ax.text(u + m * 0.04, yi, str(u), va="center", fontsize=6.5, color=INK2)
        ax.text(-d - m * 0.04, yi, str(d), va="center", ha="right", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y, others)
    ax.set_xlim(-m * 1.6, m * 1.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.set_xlabel("Differentially abundant proteins")
    ax.set_title("Direction of change")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              handlelength=1, columnspacing=1)


def plot_upset(fig, cell, de, samples, colors):
    others = samples[1:]
    sets = {s: set(de.loc[(de.Sample == s) & (de.call != "ns"), "Protein"]) for s in others}
    universe = set().union(*sets.values())
    combos = []
    for r in range(1, len(others) + 1):
        for combo in combinations(others, r):
            inside = set.intersection(*(sets[s] for s in combo))
            outside = set().union(*(sets[s] for s in others if s not in combo)) if r < len(others) else set()
            combos.append((combo, len(inside - outside)))
    combos = sorted([c for c in combos if c[1] > 0], key=lambda c: -c[1])[:10]

    sg = cell.subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
    axb = fig.add_subplot(sg[0])
    axm = fig.add_subplot(sg[1], sharex=axb)
    xs = np.arange(len(combos))
    bars = axb.bar(xs, [c[1] for c in combos], width=0.62, color=INK2)
    for b, (_, n) in zip(bars, combos):
        axb.text(b.get_x() + b.get_width() / 2, n, str(n), ha="center", va="bottom",
                 fontsize=6, color=INK2)
    axb.set_ylabel("Proteins")
    axb.set_title(f"Shared vs unique changes (n = {len(universe)})")
    axb.tick_params(labelbottom=False, bottom=False)
    axb.margins(y=0.15)
    for i, s in enumerate(others):
        yi = len(others) - 1 - i
        for xi, (combo, _) in enumerate(combos):
            on = s in combo
            axm.scatter(xi, yi, s=22 if on else 14, c=colors[s] if on else GRID, zorder=3,
                        edgecolors="white", linewidths=0.6)
        for xi, (combo, _) in enumerate(combos):
            ys = [len(others) - 1 - others.index(t) for t in combo]
            if len(ys) > 1:
                axm.plot([xi, xi], [min(ys), max(ys)], color=INK2, lw=1.0, zorder=2)
    axm.set_yticks(range(len(others)), others[::-1])
    axm.set_ylim(-0.6, len(others) - 0.4)
    axm.tick_params(bottom=False, labelbottom=False, left=False)
    for sp in ("left", "bottom"):
        axm.spines[sp].set_visible(False)
    return axb


def plot_heatmap(fig, ax, x, de, samples, colors):
    sig = de[de["call"] != "ns"].sort_values("adj_pvalue")
    per = max(HEATMAP_TOP_N // max(len(samples) - 1, 1), 1)
    top = list(dict.fromkeys(sig.groupby("Sample", sort=False).head(per)["Protein"]))[:HEATMAP_TOP_N]
    design = design_of(x)
    cols = [c for s in samples for c in x.columns[design == s]]
    z = x.loc[top, cols]
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1, ddof=1), axis=0)
    order = leaves_list(linkage(z.values, method="average", metric="correlation"))
    z = z.iloc[order]
    im = ax.imshow(z.values, aspect="auto", cmap=DIVERGING, vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    # Group bar + separators
    bounds = np.cumsum([(design == s).sum() for s in samples])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.5)
    starts = np.concatenate([[0], bounds[:-1]])
    ax.set_xticks((starts + bounds - 1) / 2, [s.replace("Sample ", "") for s in samples])
    ax.tick_params(axis="x", length=0, labelsize=7)
    for lab, s in zip(ax.get_xticklabels(), samples):
        lab.set_fontweight("bold")
    for s0, s1, s in zip(starts, bounds, samples):
        ax.add_patch(mpl.patches.Rectangle((s0 - 0.5 + 0.15, -2.6), s1 - s0 - 0.3, 1.6,
                                           color=colors[s], clip_on=False))
    ax.set_yticks([])
    ax.set_ylabel(f"Top {len(top)} differentially abundant proteins")
    ax.set_xlabel("Sample (replicates)")
    ax.set_title("Proteome signature (row z-score)", pad=14)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, shrink=0.45, aspect=14)
    cb.set_label("z-score", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)


def plot_dots(fig, ax, enr, samples):
    top = (enr[enr.FDR < FDR_THRESH].groupby("Pathway")["FDR"].min()
           .sort_values().head(8).index.tolist())
    if not top:
        top = enr.groupby("Pathway")["pvalue"].min().sort_values().head(8).index.tolist()
    sub = enr[enr.Pathway.isin(top)].copy()
    others = samples[1:]
    xi = {s: i for i, s in enumerate(others)}
    yi = {p: i for i, p in enumerate(top[::-1])}
    sub["score"] = -np.log10(sub["FDR"].clip(lower=1e-12))
    vmax = max(sub["score"].max(), 2)
    sig = sub[sub.FDR < FDR_THRESH]
    nsig = sub[sub.FDR >= FDR_THRESH]
    ax.scatter(nsig.Sample.map(xi), nsig.Pathway.map(yi), s=nsig.overlap * 3 + 4,
               facecolors="none", edgecolors=NS_COLOR, linewidths=0.7)
    sc = ax.scatter(sig.Sample.map(xi), sig.Pathway.map(yi), s=sig.overlap * 3 + 4,
                    c=sig.score, cmap=SEQUENTIAL, vmin=0, vmax=vmax, edgecolors="white",
                    linewidths=0.6)
    ax.set_xticks(range(len(others)), others)
    ax.set_yticks(range(len(top)), top[::-1])
    ax.set_xlim(-0.6, len(others) - 0.4)
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Pathway over-representation")
    cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.03, shrink=0.8, aspect=12)
    cb.set_label("−log$_{10}$(FDR)", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    handles = [Line2D([], [], ls="", marker="o", ms=np.sqrt(n * 3 + 4), mfc=INK2, mec="white")
               for n in (10, 30, 60)]
    ax.legend(handles, ["10", "30", "60"], title="Proteins", loc="upper left",
              bbox_to_anchor=(1.35, 1.0), frameon=False, labelspacing=1.1, title_fontsize=6.5)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def build_figure(x, de, enr, samples, out: Path):
    colors = sample_colors(samples)
    n_vol = len(samples) - 1
    fig = plt.figure(figsize=(183 * MM, 165 * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 * MM, h_pad=2 * MM, wspace=0.06, hspace=0.06)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05])

    ax_pca = fig.add_subplot(gs[0, 0])
    plot_pca(ax_pca, x, samples, colors)
    panel_letter(ax_pca, "a")

    vg = gs[0, 1:].subgridspec(1, n_vol, wspace=0.08)
    xlim = np.ceil(de["log2FC"].abs().quantile(0.999) + 0.5)
    ylim = de["neglog10_adjp"].quantile(0.999) * 1.12
    for i, s in enumerate(samples[1:]):
        ax = fig.add_subplot(vg[i])
        plot_volcano(ax, de[de.Sample == s], colors[s], xlim, ylim, i == 0)
        if i == 0:
            panel_letter(ax, "b")

    ax_cnt = fig.add_subplot(gs[1, 0])
    plot_counts(ax_cnt, de, samples)
    panel_letter(ax_cnt, "c")

    ax_up = plot_upset(fig, gs[1, 1], de, samples, colors)
    panel_letter(ax_up, "d")

    ax_hm = fig.add_subplot(gs[1:, 2:])
    plot_heatmap(fig, ax_hm, x, de, samples, colors)
    panel_letter(ax_hm, "e")

    ax_dot = fig.add_subplot(gs[2, :2])
    plot_dots(fig, ax_dot, enr, samples)
    panel_letter(ax_dot, "f")

    fig.suptitle("Precision proteomics: genotype-dependent remodeling relative to Sample A",
                 fontsize=9.5, fontweight="bold", color=INK)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"graphical_abstract_proteomics_python.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="protein x replicate CSV (omit for demo data)")
    ap.add_argument("--pathways", type=Path, help="CSV with columns Pathway, Protein")
    ap.add_argument("--reference", help="column group to use as Sample A (default: first)")
    ap.add_argument("--keep-pathway-names", action="store_true",
                    help="show real pathway names instead of Pathway 1, 2, ...")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).with_name("output_python"))
    args = ap.parse_args()

    apply_style()
    if args.input:
        mat, pw = load_dataset(args.input, args.reference, args.pathways, args.keep_pathway_names)
    else:
        print("No --input given: running on simulated demo data.")
        mat, pw = simulate_dataset()

    samples = list(dict.fromkeys(design_of(mat)))
    x = preprocess(mat)
    de = differential(x, samples)
    if pw is None:
        raise SystemExit("A --pathways file is needed for panel f when --input is used.")
    enr = enrichment(de, pw, x.index)

    args.outdir.mkdir(parents=True, exist_ok=True)
    de.to_csv(args.outdir / "differential_abundance.csv", index=False)
    enr.to_csv(args.outdir / "pathway_enrichment.csv", index=False)
    summary = de.groupby("Sample", sort=False)["call"].value_counts().unstack(fill_value=0)
    summary.to_csv(args.outdir / "de_summary_counts.csv")
    print(summary)

    build_figure(x, de, enr, samples, args.outdir)
    print(f"Saved figure and tables to {args.outdir}")


if __name__ == "__main__":
    main()
