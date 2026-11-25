#!/usr/bin/env python3
"""
Anonymized proteomics plot gallery: 28 publication-grade figures (Python)
=========================================================================

Everything is anonymous: genotypes are "Sample A" (reference), "Sample B",
"Sample C", "Sample D"; proteins are coded Protein_0001 ...; pathways are
"Pathway 1" ... No real gene, protein or sample name appears anywhere.

The script runs top to bottom as a normal Python file, and each "# %%" block
is also a runnable cell in VS Code (Python Interactive / Jupyter).

QC & data structure                 Differential abundance
 01 identifications per replicate     13 volcano plots
 02 missing-value map                 14 MA plots
 03 intensity boxplots raw vs norm.   15 p-value histograms
 04 intensity density curves          16 log2FC distributions
 05 imputation QC                     17 up/down counts
 06 dynamic range (abundance rank)    18 UpSet overlap
 07 CV distributions (violins)        19 log2FC concordance between samples
 08 replicate correlation heatmap     20 clustered heatmap of top proteins
 09 replicate dendrogram              21 k-means abundance-profile clusters
 10 PCA scores                        22 top-protein box/strip plots
 11 PCA scree                         23 ranked log2FC waterfall
 12 PCA loadings
                                     Pathway level
                                      24 over-representation dot plot
                                      25 enrichment bar chart
                                      26 GSEA running enrichment score
                                      27 pathway activity heatmap
                                      28 pathway activity radar

Output: one PNG (300 dpi) per plot + all plots in one multi-page PDF.
Requirements: numpy, pandas, scipy, matplotlib
To use real data, replace simulate_dataset() with a loader that returns a
protein x replicate log2 matrix (columns "Sample X_R<n>") and a two-column
Pathway/Protein table, after recoding every name to the anonymous scheme.
"""
# %% Setup ---------------------------------------------------------------
from __future__ import annotations

import string
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.cluster.vq import kmeans2

try:
    OUT = Path(__file__).resolve().parent / "output_plots"
except NameError:                     # running as a notebook cell
    OUT = Path.cwd() / "output_plots"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FDR_THRESH, LOG2FC_THRESH = 0.05, 1.0
MIN_VALID_FRACTION = 0.70

# Validated palette; reference sample is neutral gray, others in fixed order
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NS_COLOR, UP_COLOR, DOWN_COLOR = "#c3c2b7", "#e34948", "#2a78d6"
REFERENCE_COLOR = "#52514e"
SAMPLE_HUES = ["#2a78d6", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300"]
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#8a2323"], N=256)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#b5d3f7", "#6da7ec", "#2a78d6", "#184f95"], N=256)
MM = 1 / 25.4
W1, W15 = 89 * MM, 130 * MM           # single / 1.5 column widths

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.max_open_warning": 0, "figure.constrained_layout.use": True,
})

PLOTS: list[tuple[str, plt.Figure]] = []


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    PLOTS.append((name, fig))
    print(f"  saved {name}.png")


# %% Data: simulated anonymous LFQ experiment ----------------------------
def simulate_dataset(n_proteins=3000, n_samples=4, n_reps=4, seed=SEED):
    rng = np.random.default_rng(seed)
    samples = [f"Sample {string.ascii_uppercase[i]}" for i in range(n_samples)]
    proteins = np.array([f"Protein_{i:04d}" for i in range(1, n_proteins + 1)])
    base = rng.normal(22.0, 2.2, n_proteins)
    pathways = {f"Pathway {k + 1}": rng.choice(n_proteins, rng.integers(60, 160), replace=False)
                for k in range(12)}
    eff = {s: np.zeros(n_proteins) for s in samples}
    core = rng.choice(np.concatenate([pathways["Pathway 1"], pathways["Pathway 2"]]), 110, replace=False)
    core_fc = rng.choice([-1, 1], core.size, p=[0.45, 0.55]) * rng.normal(1.9, 0.45, core.size)
    for s in samples[1:]:
        eff[s][core] = core_fc * rng.normal(1.0, 0.15, core.size)
    for i, s in enumerate(samples[1:]):
        own = pathways[f"Pathway {i + 3}"]
        spec = np.union1d(rng.choice(own, min(70, own.size), replace=False),
                          rng.choice(n_proteins, 90, replace=False))
        eff[s][spec] += rng.choice([-1, 1], spec.size) * rng.normal(1.7, 0.5, spec.size)
    pair = rng.choice(pathways["Pathway 6"], 45, replace=False)
    for s in samples[1:3]:
        eff[s][pair] += rng.normal(-1.8, 0.4, pair.size)
    cols, data = [], []
    for j, s in enumerate(samples):
        for r in range(1, n_reps + 1):
            load = rng.normal(0, 0.25)                 # per-run loading offset
            cols.append(f"{s}_R{r}")
            data.append(base + eff[s] + rng.normal(0, 0.35, n_proteins) + load)
    mat = pd.DataFrame(np.column_stack(data), index=proteins, columns=cols)
    low = mat.values < np.quantile(mat.values, 0.06)
    mat = mat.mask((low & (rng.random(mat.shape) < 0.65)) | (rng.random(mat.shape) < 0.012))
    pw = pd.DataFrame([(n, proteins[i]) for n, idx in pathways.items() for i in idx],
                      columns=["Pathway", "Protein"])
    return mat, pw, samples


def design_of(cols) -> pd.Series:
    return pd.Series([c.rsplit("_R", 1)[0] for c in cols], index=cols)


def preprocess(raw: pd.DataFrame, seed=SEED):
    rng = np.random.default_rng(seed)
    design = design_of(raw.columns)
    valid = raw.notna().T.groupby(design).mean().T
    x = raw.loc[(valid >= MIN_VALID_FRACTION).any(axis=1)]
    med = x.median()
    norm = x - med + med.mean()
    imputed = norm.isna()
    out = norm.copy()
    for c in out.columns:
        m = imputed[c]
        mu, sd = norm[c].mean() - 1.8 * norm[c].std(), 0.3 * norm[c].std()
        out.loc[m, c] = rng.normal(mu, sd, m.sum())
    return out, norm, imputed


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.clip(q, 0, 1)
    return res


def differential(x, samples):
    design = design_of(x.columns)
    ref = x.loc[:, design == samples[0]]
    rows = []
    for s in samples[1:]:
        tst = x.loc[:, design == s]
        t, p = stats.ttest_ind(tst, ref, axis=1, equal_var=False)
        df = pd.DataFrame({"Protein": x.index, "Sample": s,
                           "log2FC": tst.mean(axis=1).values - ref.mean(axis=1).values,
                           "A_mean": (tst.mean(axis=1).values + ref.mean(axis=1).values) / 2,
                           "t": t, "pvalue": p, "adj_pvalue": bh_fdr(p)})
        df["neglog10_adjp"] = -np.log10(df["adj_pvalue"].clip(lower=1e-300))
        df["call"] = np.select(
            [(df.adj_pvalue < FDR_THRESH) & (df.log2FC > LOG2FC_THRESH),
             (df.adj_pvalue < FDR_THRESH) & (df.log2FC < -LOG2FC_THRESH)], ["up", "down"], "ns")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def enrichment(de, pw, universe):
    pw = pw[pw.Protein.isin(universe)]
    sets = pw.groupby("Pathway")["Protein"].apply(set)
    N, rows = len(universe), []
    for s, sub in de.groupby("Sample", sort=False):
        hits = set(sub.loc[sub.call != "ns", "Protein"])
        n = len(hits)
        for name, mem in sets.items():
            k, K = len(hits & mem), len(mem)
            rows.append({"Sample": s, "Pathway": name, "overlap": k, "set_size": K,
                         "fold_enrichment": (k / n) / (K / N) if n else 0,
                         "pvalue": stats.hypergeom.sf(k - 1, N, K, n) if k else 1.0})
    res = pd.DataFrame(rows)
    res["FDR"] = res.groupby("Sample")["pvalue"].transform(bh_fdr)
    return res, sets


raw, pathway_table, SAMPLES = simulate_dataset()
X, NORM, IMPUTED = preprocess(raw)
DESIGN = design_of(X.columns)
DE = differential(X, SAMPLES)
ENR, SETS = enrichment(DE, pathway_table, X.index)
OTHERS = SAMPLES[1:]
COLORS = {SAMPLES[0]: REFERENCE_COLOR, **{s: SAMPLE_HUES[i] for i, s in enumerate(OTHERS)}}
COL_COLORS = [COLORS[DESIGN[c]] for c in X.columns]
SHORT = [c.replace("Sample ", "").replace("_", "-") for c in X.columns]   # e.g. A-R1
print(f"{len(raw)} proteins simulated, {len(X)} retained; {len(SAMPLES)} samples x "
      f"{(DESIGN == SAMPLES[0]).sum()} replicates")


def sample_legend(ax, samples=SAMPLES, **kw):
    h = [Line2D([], [], ls="", marker="s", ms=6, color=COLORS[s]) for s in samples]
    ax.legend(h, samples, handletextpad=0.3, **kw)


def color_ticklabels(labels):
    for lab in labels:
        lab.set_color(COLORS["Sample " + lab.get_text()[0]])


# %% 01 Protein identifications per replicate -----------------------------
fig, ax = plt.subplots(figsize=(W1, 55 * MM))
ids = raw.notna().sum()
ax.bar(range(len(ids)), ids.values, color=[COLORS[DESIGN[c]] for c in raw.columns], width=0.72)
ax.set_xticks(range(len(ids)), SHORT, rotation=90)
ax.set_ylim(ids.min() * 0.97, ids.max() * 1.01)
ax.set_ylabel("Quantified proteins")
ax.set_title("Protein identifications per replicate")
sample_legend(ax, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.28))
save(fig, "01_identifications_per_replicate")

# %% 02 Missing-value map -------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 70 * MM))
miss = raw.isna()
miss = miss[miss.any(axis=1)]
miss = miss.loc[raw.loc[miss.index].mean(axis=1).sort_values().index]
ax.imshow(miss.values, aspect="auto", cmap=LinearSegmentedColormap.from_list("m", ["#f0efec", INK2]),
          interpolation="nearest")
ax.set_xticks(range(len(SHORT)), SHORT, rotation=90)
ax.set_yticks([])
ax.set_ylabel(f"{len(miss)} proteins with ≥1 missing value\n(sorted by mean abundance, low → high)")
ax.set_title(f"Missing-value map ({100 * raw.isna().values.mean():.1f}% missing overall)")
ax.legend([mpl.patches.Patch(color=INK2), mpl.patches.Patch(color="#f0efec", ec=MUTED)],
          ["Missing", "Observed"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "02_missing_value_map")

# %% 03 Intensity boxplots: raw vs normalized -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(W15, 60 * MM), sharey=True)
for ax, data, ttl in [(axes[0], raw, "Before normalization"), (axes[1], NORM, "After median normalization")]:
    vals = [data[c].dropna().values for c in data.columns]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.2), whiskerprops=dict(color=MUTED, lw=0.6),
                    capprops=dict(color=MUTED, lw=0.6))
    for patch, c in zip(bp["boxes"], COL_COLORS):
        patch.set(facecolor=c, edgecolor=c)
    ax.set_xticks(range(1, len(SHORT) + 1), SHORT, rotation=90)
    ax.set_title(ttl)
axes[0].set_ylabel("log$_2$ intensity")
save(fig, "03_intensity_boxplots_raw_vs_normalized")

# %% 04 Intensity density curves -------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
grid = np.linspace(np.nanmin(NORM.values), np.nanmax(NORM.values), 300)
for c in NORM.columns:
    kde = stats.gaussian_kde(NORM[c].dropna())
    ax.plot(grid, kde(grid), color=COLORS[DESIGN[c]], lw=1, alpha=0.85)
ax.set_xlabel("log$_2$ intensity (normalized)")
ax.set_ylabel("Density")
ax.set_title("Intensity distributions per replicate")
sample_legend(ax, loc="upper right")
save(fig, "04_intensity_density")

# %% 05 Imputation QC --------------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
obs = X.values[~IMPUTED.values]
imp = X.values[IMPUTED.values]
bins = np.linspace(X.values.min(), X.values.max(), 60)
ax.hist(obs, bins=bins, color=NS_COLOR, label=f"Observed (n = {obs.size:,})")
ax.hist(imp, bins=bins, color=DOWN_COLOR, label=f"Imputed (n = {imp.size:,})")
ax.set_xlabel("log$_2$ intensity")
ax.set_ylabel("Values")
ax.set_title("Left-shifted imputation of missing values")
ax.legend(loc="upper right")
save(fig, "05_imputation_qc")

# %% 06 Dynamic range ------------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
for s in SAMPLES:
    m = X.loc[:, DESIGN == s].mean(axis=1).sort_values(ascending=False)
    ax.plot(np.arange(1, len(m) + 1), m.values, color=COLORS[s], lw=1.4, label=s)
ref_mean = X.loc[:, DESIGN == SAMPLES[0]].mean(axis=1)
span = ref_mean.quantile(0.99) - ref_mean.quantile(0.01)
ax.text(0.97, 0.9, f"≈{span / np.log2(10):.1f} orders of magnitude\n(1st–99th percentile)",
        transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=6.5)
ax.set_xlabel("Abundance rank")
ax.set_ylabel("Mean log$_2$ intensity")
ax.set_title("Dynamic range of the quantified proteome")
ax.legend(loc="lower left")
save(fig, "06_dynamic_range")

# %% 07 CV distributions -----------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
cvs = []
for s in SAMPLES:
    lin = 2 ** NORM.loc[:, DESIGN == s].dropna()
    cvs.append((lin.std(axis=1, ddof=1) / lin.mean(axis=1) * 100).clip(upper=100).values)
vp = ax.violinplot(cvs, showextrema=False, widths=0.8)
for body, s in zip(vp["bodies"], SAMPLES):
    body.set(facecolor=COLORS[s], edgecolor="none", alpha=0.85)
for i, v in enumerate(cvs, start=1):
    med = np.median(v)
    ax.plot([i - 0.18, i + 0.18], [med, med], color="white", lw=1.6)
    ax.text(i, 104, f"median\n{med:.1f}%", ha="center", va="bottom", fontsize=6, color=INK2)
ax.axhline(20, color=MUTED, ls="--", lw=0.6)
ax.set_ylim(0, 118)
ax.set_yticks(range(0, 101, 20))
ax.set_xticks(range(1, len(SAMPLES) + 1), SAMPLES)
ax.set_ylabel("Coefficient of variation (%)")
ax.set_title("Replicate precision (dashed line = 20% CV)")
save(fig, "07_cv_distribution")

# %% 08 Replicate correlation heatmap ----------------------------------------
fig, ax = plt.subplots(figsize=(W1, 80 * MM))
corr = np.corrcoef(X.T.values)
im = ax.imshow(corr, cmap=SEQUENTIAL, vmin=np.floor(corr.min() * 100) / 100, vmax=1)
ax.set_xticks(range(len(SHORT)), SHORT, rotation=90)
ax.set_yticks(range(len(SHORT)), SHORT)
color_ticklabels(ax.get_xticklabels())
color_ticklabels(ax.get_yticklabels())
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Pearson r")
cb.outline.set_visible(False)
ax.set_title("Replicate-to-replicate correlation")
save(fig, "08_replicate_correlation_heatmap")

# %% 09 Replicate dendrogram ----------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
Z = linkage(X.T.values, method="average", metric="correlation")
dendrogram(Z, labels=SHORT, ax=ax, color_threshold=0, above_threshold_color=INK2, leaf_rotation=90, leaf_font_size=6.5)
color_ticklabels(ax.get_xticklabels())
ax.set_ylabel("1 − Pearson r")
ax.set_title("Unsupervised clustering of replicates")
ax.spines["bottom"].set_visible(False)
save(fig, "09_replicate_dendrogram")

# %% 10–12 PCA ----------------------------------------------------------------
Xc = X.T.values - X.T.values.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
PCS = U * S
VAR = S ** 2 / np.sum(S ** 2) * 100

fig, ax = plt.subplots(figsize=(W1, 70 * MM))
for s in SAMPLES:
    m = (DESIGN == s).values
    ax.scatter(PCS[m, 0], PCS[m, 1], s=40, c=COLORS[s], edgecolors="white", lw=0.8, label=s, zorder=3)
    ax.annotate(s.replace("Sample ", ""), (PCS[m, 0].mean(), PCS[m, 1].max()), xytext=(0, 6),
                textcoords="offset points", ha="center", fontweight="bold")
ax.set_xlabel(f"PC1 ({VAR[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({VAR[1]:.1f}%)")
ax.set_title("Principal component analysis")
ax.grid(True, zorder=0)
ax.margins(0.15)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
save(fig, "10_pca_scores")

fig, ax = plt.subplots(figsize=(W1, 55 * MM))
k = min(10, len(VAR))
ax.bar(range(1, k + 1), VAR[:k], color=INK2, width=0.65, label="Per component")
ax.plot(range(1, k + 1), np.cumsum(VAR[:k]), color=DOWN_COLOR, marker="o", ms=4, lw=1.4,
        label="Cumulative")
ax.set_xticks(range(1, k + 1), [f"PC{i}" for i in range(1, k + 1)])
ax.set_ylabel("Variance explained (%)")
ax.set_ylim(0, 105)
ax.set_title("PCA scree plot")
ax.legend(loc="center right")
save(fig, "11_pca_scree")

fig, ax = plt.subplots(figsize=(W1, 70 * MM))
load = Vt[:2].T
mag = np.hypot(load[:, 0], load[:, 1])
top = mag >= np.quantile(mag, 0.98)
ax.scatter(load[~top, 0], load[~top, 1], s=3, c=NS_COLOR, lw=0, rasterized=True, label="All proteins")
ax.scatter(load[top, 0], load[top, 1], s=8, c=DOWN_COLOR, lw=0, label="Top 2% contributors")
ax.axhline(0, color=MUTED, lw=0.5)
ax.axvline(0, color=MUTED, lw=0.5)
ax.set_xlabel("PC1 loading")
ax.set_ylabel("PC2 loading")
ax.set_title("PCA loadings: proteins driving separation")
ax.legend(loc="upper left", markerscale=2)
save(fig, "12_pca_loadings")

# %% 13 Volcano plots ---------------------------------------------------------
xlim = np.ceil(DE.log2FC.abs().quantile(0.999) + 0.5)
ylim = DE.neglog10_adjp.quantile(0.999) * 1.12
fig, axes = plt.subplots(1, len(OTHERS), figsize=(183 * MM, 60 * MM), sharey=True)
for ax, s in zip(axes, OTHERS):
    sub = DE[DE.Sample == s]
    for call, c, sz in [("ns", NS_COLOR, 2.5), ("down", DOWN_COLOR, 5), ("up", UP_COLOR, 5)]:
        d = sub[sub.call == call]
        ax.scatter(d.log2FC, d.neglog10_adjp, s=sz, c=c, lw=0, alpha=0.9, rasterized=True)
    ax.axvline(-LOG2FC_THRESH, color=MUTED, ls="--", lw=0.5)
    ax.axvline(LOG2FC_THRESH, color=MUTED, ls="--", lw=0.5)
    ax.axhline(-np.log10(FDR_THRESH), color=MUTED, ls="--", lw=0.5)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, ylim)
    ax.text(0.04, 0.96, f"▼ {(sub.call == 'down').sum()}", transform=ax.transAxes, va="top", color=INK2)
    ax.text(0.96, 0.96, f"{(sub.call == 'up').sum()} ▲", transform=ax.transAxes, va="top", ha="right",
            color=INK2)
    ax.set_title(f"{s} vs Sample A")
    ax.set_xlabel("log$_2$ fold change")
axes[0].set_ylabel("−log$_{10}$(FDR)")
save(fig, "13_volcano_plots")

# %% 14 MA plots ----------------------------------------------------------------
fig, axes = plt.subplots(1, len(OTHERS), figsize=(183 * MM, 58 * MM), sharey=True)
for ax, s in zip(axes, OTHERS):
    sub = DE[DE.Sample == s]
    for call, c, sz in [("ns", NS_COLOR, 2.5), ("down", DOWN_COLOR, 5), ("up", UP_COLOR, 5)]:
        d = sub[sub.call == call]
        ax.scatter(d.A_mean, d.log2FC, s=sz, c=c, lw=0, alpha=0.9, rasterized=True)
    ax.axhline(0, color=INK2, lw=0.6)
    for v in (-LOG2FC_THRESH, LOG2FC_THRESH):
        ax.axhline(v, color=MUTED, ls="--", lw=0.5)
    ax.set_title(f"{s} vs Sample A")
    ax.set_xlabel("Mean log$_2$ intensity (A)")
axes[0].set_ylabel("log$_2$ fold change (M)")
save(fig, "14_ma_plots")

# %% 15 p-value histograms ------------------------------------------------------
fig, axes = plt.subplots(1, len(OTHERS), figsize=(183 * MM, 50 * MM), sharey=True)
for ax, s in zip(axes, OTHERS):
    p = DE.loc[DE.Sample == s, "pvalue"]
    ax.hist(p, bins=40, range=(0, 1), color=COLORS[s])
    ax.axhline(len(p) / 40, color=INK2, ls="--", lw=0.6)
    ax.set_title(f"{s} vs Sample A")
    ax.set_xlabel("Raw p-value")
axes[0].set_ylabel("Proteins")
axes[-1].text(1, len(DE[DE.Sample == OTHERS[-1]]) / 40, "uniform\nnull", color=INK2, fontsize=6,
              ha="right", va="bottom")
save(fig, "15_pvalue_histograms")

# %% 16 log2FC distributions ----------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 60 * MM))
g = np.linspace(-5, 5, 400)
for s in OTHERS:
    fc = DE.loc[DE.Sample == s, "log2FC"]
    ax.hist(fc, bins=np.linspace(-5, 5, 61), histtype="step", color=COLORS[s], lw=1.3, label=s)
ax.set_yscale("log")
ax.set_ylim(0.8, None)
ax.axvline(0, color=INK2, lw=0.6)
for v in (-LOG2FC_THRESH, LOG2FC_THRESH):
    ax.axvline(v, color=MUTED, ls="--", lw=0.5)
ax.set_xlabel("log$_2$ fold change vs Sample A")
ax.set_ylabel("Proteins (log scale)")
ax.set_title("Fold-change distributions")
ax.legend(loc="upper right")
save(fig, "16_log2fc_distributions")

# %% 17 Up/down counts ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 50 * MM))
up = [((DE.Sample == s) & (DE.call == "up")).sum() for s in OTHERS]
dn = [((DE.Sample == s) & (DE.call == "down")).sum() for s in OTHERS]
y = np.arange(len(OTHERS))[::-1]
ax.barh(y, up, color=UP_COLOR, height=0.6, label="Higher than Sample A")
ax.barh(y, [-d for d in dn], color=DOWN_COLOR, height=0.6, label="Lower than Sample A")
m = max(up + dn)
for yi, u, d in zip(y, up, dn):
    ax.text(u + m * 0.03, yi, str(u), va="center", color=INK2)
    ax.text(-d - m * 0.03, yi, str(d), va="center", ha="right", color=INK2)
ax.axvline(0, color=INK2, lw=0.6)
ax.set_yticks(y, OTHERS)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(-m * 1.3, m * 1.3)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
ax.set_xlabel("Differentially abundant proteins")
ax.set_title("Direction of change")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2)
save(fig, "17_up_down_counts")

# %% 18 UpSet overlap -------------------------------------------------------------
sets = {s: set(DE.loc[(DE.Sample == s) & (DE.call != "ns"), "Protein"]) for s in OTHERS}
combos = []
for r in range(1, len(OTHERS) + 1):
    for cmb in combinations(OTHERS, r):
        inside = set.intersection(*(sets[s] for s in cmb))
        outside = set().union(*(sets[s] for s in OTHERS if s not in cmb))
        combos.append((cmb, len(inside - outside)))
combos = sorted([c for c in combos if c[1]], key=lambda c: -c[1])
fig, (axb, axm) = plt.subplots(2, 1, figsize=(W1, 70 * MM), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1]))
xs = np.arange(len(combos))
axb.bar(xs, [c[1] for c in combos], color=INK2, width=0.6)
for xi, (_, n) in zip(xs, combos):
    axb.text(xi, n, str(n), ha="center", va="bottom", fontsize=6, color=INK2)
axb.set_ylabel("Proteins")
axb.set_title("Shared and unique changes (UpSet)")
axb.tick_params(bottom=False)
for i, s in enumerate(OTHERS):
    yi = len(OTHERS) - 1 - i
    for xi, (cmb, _) in enumerate(combos):
        axm.scatter(xi, yi, s=26 if s in cmb else 16, c=COLORS[s] if s in cmb else GRID, zorder=3)
for xi, (cmb, _) in enumerate(combos):
    ys = [len(OTHERS) - 1 - OTHERS.index(t) for t in cmb]
    if len(ys) > 1:
        axm.plot([xi, xi], [min(ys), max(ys)], color=INK2, lw=1)
axm.set_yticks(range(len(OTHERS)), OTHERS[::-1])
axm.set_ylim(-0.6, len(OTHERS) - 0.4)
axm.tick_params(bottom=False, labelbottom=False, left=False)
for sp in ("left", "bottom"):
    axm.spines[sp].set_visible(False)
save(fig, "18_upset_overlap")

# %% 19 log2FC concordance --------------------------------------------------------
wide = DE.pivot(index="Protein", columns="Sample", values="log2FC")
sigany = DE[DE.call != "ns"].Protein.unique()
pairs = list(combinations(OTHERS, 2))
fig, axes = plt.subplots(1, len(pairs), figsize=(183 * MM, 60 * MM))
lim = np.ceil(wide.abs().quantile(0.999).max())
for ax, (s1, s2) in zip(np.atleast_1d(axes), pairs):
    ns = ~wide.index.isin(sigany)
    ax.scatter(wide.loc[ns, s1], wide.loc[ns, s2], s=2.5, c=NS_COLOR, lw=0, rasterized=True)
    ax.scatter(wide.loc[~ns, s1], wide.loc[~ns, s2], s=5, c=INK2, lw=0, alpha=0.8, rasterized=True)
    r = np.corrcoef(wide[s1], wide[s2])[0, 1]
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, ls="--", lw=0.5)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.axvline(0, color=GRID, lw=0.5)
    ax.text(0.04, 0.96, f"r = {r:.2f}", transform=ax.transAxes, va="top", color=INK2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"log$_2$FC {s1}")
    ax.set_ylabel(f"log$_2$FC {s2}")
fig.suptitle("Concordance of fold changes between samples (dark = significant in any sample)",
             fontweight="bold", fontsize=8)
save(fig, "19_log2fc_concordance")

# %% 20 Clustered heatmap of top proteins -----------------------------------------
topn = list(dict.fromkeys(DE[DE.call != "ns"].sort_values("adj_pvalue").groupby("Sample", sort=False)
                          .head(20).Protein))
Zm = X.loc[topn]
Zm = Zm.sub(Zm.mean(axis=1), axis=0).div(Zm.std(axis=1, ddof=1), axis=0)
Zm = Zm.iloc[leaves_list(linkage(Zm.values, "average", metric="correlation"))]
fig, ax = plt.subplots(figsize=(W1, 110 * MM))
im = ax.imshow(Zm.values, aspect="auto", cmap=DIVERGING, vmin=-2.5, vmax=2.5, interpolation="nearest")
bounds = np.cumsum([(DESIGN == s).sum() for s in SAMPLES])
for b in bounds[:-1]:
    ax.axvline(b - 0.5, color="white", lw=1.5)
ax.set_xticks(range(len(SHORT)), SHORT, rotation=90)
color_ticklabels(ax.get_xticklabels())
ax.set_yticks(range(len(Zm)), Zm.index, fontsize=4.5)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, shrink=0.4)
cb.set_label("Row z-score")
cb.outline.set_visible(False)
ax.set_title(f"Top {len(Zm)} differentially abundant proteins")
save(fig, "20_top_protein_heatmap")

# %% 21 k-means abundance-profile clusters ----------------------------------------
sig = DE[DE.call != "ns"].Protein.unique()
means = pd.DataFrame({s: X.loc[sig, DESIGN == s].mean(axis=1) for s in SAMPLES})
zp = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1, ddof=1), axis=0)
np.random.seed(SEED)
_, lab = kmeans2(zp.values, 4, minit="++", seed=SEED)
fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 52 * MM), sharey=True)
xs = np.arange(len(SAMPLES))
for k, ax in enumerate(axes):
    mem = zp.values[lab == k]
    for row in mem:
        ax.plot(xs, row, color=NS_COLOR, lw=0.4, alpha=0.6)
    ax.plot(xs, mem.mean(axis=0), color=INK, lw=2, marker="o", ms=4)
    ax.set_xticks(xs, [s.replace("Sample ", "") for s in SAMPLES])
    ax.set_title(f"Cluster {k + 1} (n = {len(mem)})")
    ax.grid(axis="y")
axes[0].set_ylabel("Abundance z-score")
fig.suptitle("Co-regulated protein clusters (k-means on differentially abundant proteins)",
             fontweight="bold", fontsize=8)
save(fig, "21_kmeans_profile_clusters")

# %% 22 Top-protein box/strip plots -----------------------------------------------
best = DE.sort_values("adj_pvalue").drop_duplicates("Protein").head(6).Protein.tolist()
fig, axes = plt.subplots(2, 3, figsize=(W15, 80 * MM))
rng = np.random.default_rng(SEED)
for ax, p in zip(axes.flat, best):
    data = [X.loc[p, DESIGN == s].values for s in SAMPLES]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=MUTED, lw=0.6),
                    capprops=dict(color=MUTED, lw=0.6))
    for patch, s in zip(bp["boxes"], SAMPLES):
        patch.set(facecolor=COLORS[s], alpha=0.25, edgecolor=COLORS[s])
    for i, (d, s) in enumerate(zip(data, SAMPLES), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=10, c=COLORS[s], edgecolors="white",
                   lw=0.4, zorder=3)
    ax.set_xticks(range(1, len(SAMPLES) + 1), [s.replace("Sample ", "") for s in SAMPLES])
    ax.set_title(p, fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("log$_2$ intensity")
fig.suptitle("Most significant proteins across samples", fontweight="bold", fontsize=8)
save(fig, "22_top_protein_boxplots")

# %% 23 Ranked log2FC waterfall ----------------------------------------------------
fig, axes = plt.subplots(len(OTHERS), 1, figsize=(W1, 85 * MM), sharex=True, sharey=True)
for ax, s in zip(axes, OTHERS):
    sub = DE[DE.Sample == s].sort_values("log2FC", ascending=False).reset_index(drop=True)
    col = np.where(sub.call == "up", UP_COLOR, np.where(sub.call == "down", DOWN_COLOR, NS_COLOR))
    ax.bar(sub.index, sub.log2FC, width=1.0, color=col, lw=0)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.text(0.99, 0.92, s, transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    ax.set_ylabel("log$_2$FC")
axes[-1].set_xlabel("Proteins ranked by fold change")
axes[0].set_title("Ranked fold-change waterfall")
save(fig, "23_ranked_log2fc_waterfall")

# %% 24 Pathway over-representation dot plot --------------------------------------
top_pw = ENR.groupby("Pathway").FDR.min().sort_values().head(10).index.tolist()
sub = ENR[ENR.Pathway.isin(top_pw)].copy()
sub["score"] = -np.log10(sub.FDR.clip(lower=1e-12))
xi = {s: i for i, s in enumerate(OTHERS)}
yi = {p: i for i, p in enumerate(top_pw[::-1])}
fig, ax = plt.subplots(figsize=(W1, 75 * MM))
sg, ns = sub[sub.FDR < FDR_THRESH], sub[sub.FDR >= FDR_THRESH]
ax.scatter(ns.Sample.map(xi), ns.Pathway.map(yi), s=ns.overlap * 3 + 5, facecolors="none",
           edgecolors=NS_COLOR, lw=0.7)
sc = ax.scatter(sg.Sample.map(xi), sg.Pathway.map(yi), s=sg.overlap * 3 + 5, c=sg.score,
                cmap=SEQUENTIAL, vmin=0, edgecolors="white", lw=0.5)
ax.set_xticks(range(len(OTHERS)), OTHERS)
ax.set_yticks(range(len(top_pw)), top_pw[::-1])
ax.set_xlim(-0.6, len(OTHERS) - 0.4)
ax.grid(True)
ax.set_axisbelow(True)
cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.04, shrink=0.6)
cb.set_label("−log$_{10}$(FDR)")
cb.outline.set_visible(False)
h = [Line2D([], [], ls="", marker="o", ms=np.sqrt(n * 3 + 5), mfc=INK2, mec="white") for n in (10, 30, 60)]
ax.legend(h, ["10", "30", "60"], title="Proteins in pathway", loc="upper center",
          bbox_to_anchor=(0.5, -0.1), ncol=3, columnspacing=1.5)
ax.set_title("Pathway over-representation")
save(fig, "24_pathway_dotplot")

# %% 25 Enrichment bar chart -------------------------------------------------------
fig, ax = plt.subplots(figsize=(W1, 75 * MM))
yb = np.arange(len(top_pw))
hgt = 0.8 / len(OTHERS)
for i, s in enumerate(OTHERS):
    v = ENR[ENR.Sample == s].set_index("Pathway").loc[top_pw[::-1], "FDR"]
    ax.barh(yb + (i - (len(OTHERS) - 1) / 2) * hgt, -np.log10(v.clip(lower=1e-12)), height=hgt * 0.9,
            color=COLORS[s], label=s)
ax.axvline(-np.log10(FDR_THRESH), color=INK2, ls="--", lw=0.6)
ax.text(-np.log10(FDR_THRESH), len(top_pw) - 0.4, " FDR 0.05", color=INK2, fontsize=6, va="bottom")
ax.set_yticks(yb, top_pw[::-1])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("−log$_{10}$(FDR)")
ax.set_title("Pathway enrichment by sample")
ax.legend(loc="lower right")
save(fig, "25_pathway_enrichment_bars")

# %% 26 GSEA running enrichment score ----------------------------------------------
s_gsea = OTHERS[0]
ranked = DE[DE.Sample == s_gsea].set_index("Protein").t.sort_values(ascending=False)
best_pw = ENR[ENR.Sample == s_gsea].sort_values("FDR").Pathway.iloc[0]
hits = ranked.index.isin(SETS[best_pw])
w = np.abs(ranked.values) * hits
p_hit = np.cumsum(w) / w.sum()
p_miss = np.cumsum(~hits) / (~hits).sum()
es = p_hit - p_miss
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(W1, 75 * MM), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 0.5, 1.4]))
a1.plot(es, color=COLORS[s_gsea], lw=1.4)
a1.axhline(0, color=INK2, lw=0.5)
peak = np.argmax(np.abs(es))
a1.axvline(peak, color=MUTED, ls="--", lw=0.5)
a1.text(peak, es[peak], f"ES = {es[peak]:.2f}  ", va="center", color=INK2,
        ha="right" if peak > len(es) / 2 else "left")
a1.set_ylabel("Running ES")
a1.set_title(f"GSEA: {best_pw}, {s_gsea} vs Sample A")
a2.vlines(np.where(hits)[0], 0, 1, color=INK, lw=0.4)
a2.set_yticks([])
a2.spines["left"].set_visible(False)
a3.fill_between(np.arange(len(ranked)), ranked.values, color=NS_COLOR, lw=0)
a3.axhline(0, color=INK2, lw=0.5)
a3.set_ylabel("t statistic")
a3.set_xlabel("Protein rank")
save(fig, "26_gsea_running_score")

# %% 27 Pathway activity heatmap ----------------------------------------------------
zall = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1, ddof=1), axis=0)
act = pd.DataFrame({p: zall.loc[zall.index.isin(m)].mean() for p, m in SETS.items()}).T
act = act.loc[top_pw]
act = act.iloc[leaves_list(linkage(act.values, "average"))]
fig, ax = plt.subplots(figsize=(W1, 65 * MM))
lim = np.abs(act.values).max()
im = ax.imshow(act.values, aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim)
for b in bounds[:-1]:
    ax.axvline(b - 0.5, color="white", lw=1.5)
ax.set_xticks(range(len(SHORT)), SHORT, rotation=90)
color_ticklabels(ax.get_xticklabels())
ax.set_yticks(range(len(act)), act.index)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("Mean z-score of members")
cb.outline.set_visible(False)
ax.set_title("Pathway activity per replicate")
save(fig, "27_pathway_activity_heatmap")

# %% 28 Pathway activity radar ------------------------------------------------------
radar_pw = top_pw[:8]
smean = pd.DataFrame({s: act.loc[radar_pw, DESIGN == s].mean(axis=1) for s in SAMPLES})
ang = np.linspace(0, 2 * np.pi, len(radar_pw), endpoint=False)
fig, ax = plt.subplots(figsize=(W1, 80 * MM), subplot_kw=dict(polar=True))
for s in SAMPLES:
    v = np.r_[smean[s].values, smean[s].values[0]]
    ax.plot(np.r_[ang, ang[0]], v, color=COLORS[s], lw=1.4, label=s)
    ax.fill(np.r_[ang, ang[0]], v, color=COLORS[s], alpha=0.08)
ax.set_xticks(ang, radar_pw, fontsize=6.5)
ax.tick_params(axis="y", labelsize=5.5, colors=MUTED)
ax.set_rlabel_position(360 / len(radar_pw) / 2)
ax.spines["polar"].set_color(GRID)
ax.grid(color=GRID)
ax.set_title("Pathway activity fingerprint per sample", pad=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
save(fig, "28_pathway_activity_radar")

# %% Export: multi-page PDF + tables --------------------------------------------------
with PdfPages(OUT / "ALL_PLOTS_python.pdf") as pdf:
    for _, f in PLOTS:
        pdf.savefig(f)
        plt.close(f)
DE.to_csv(OUT / "differential_abundance_anonymized.csv", index=False)
ENR.to_csv(OUT / "pathway_enrichment_anonymized.csv", index=False)
print(f"\n{len(PLOTS)} plots saved to {OUT}")
