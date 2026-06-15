#!/usr/bin/env python3
"""
Feature Correlation Analysis  +  Clustering & Feature Selection
----------------------------------------------------------------
Generates Pearson and Spearman correlation heatmaps + VIF report,
then clusters correlated features via hierarchical clustering and
recommends one representative per cluster so you can safely drop
redundant features.

Usage:
    python feature_correlation_analysis.py --input data.csv
    python feature_correlation_analysis.py --input data.csv --target label_column
    python feature_correlation_analysis.py --input data.csv --target label \
        --corr-threshold 0.85 --vif-threshold 10 \
        --cluster-cutoff 0.15 --cluster-method ward --select-by variance
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ── optional statsmodels for VIF ─────────────────────────────────────────────
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Correlation heatmaps + VIF + hierarchical feature clustering."
    )
    p.add_argument("--input",            required=True,  help="Path to CSV file")
    p.add_argument("--target",           default=None,   help="Target/label column to exclude (optional)")
    p.add_argument("--sep",              default=",",    help="CSV separator (default: ',')")
    p.add_argument("--corr-threshold",   type=float, default=0.90,
                   help="Flag feature pairs with |corr| >= this value (default: 0.90)")
    p.add_argument("--vif-threshold",    type=float, default=10.0,
                   help="Flag features with VIF >= this value (default: 10)")
    p.add_argument("--max-vif-features", type=int,   default=100,
                   help="Max features for VIF (expensive; default: 100)")
    p.add_argument("--sample",           type=int,   default=None,
                   help="Row sample for faster VIF on large datasets")
    p.add_argument("--out-dir",          default=".",
                   help="Output directory (default: current dir)")
    p.add_argument("--dpi",              type=int,   default=150,
                   help="DPI for saved figures (default: 150)")
    # ── clustering ────────────────────────────────────────────────────────────
    p.add_argument("--cluster-cutoff",   type=float, default=0.15,
                   help="Distance cutoff to cut dendrogram into clusters. "
                        "Distance = 1 - |corr|, so 0.15 ≈ |r|>=0.85 (default: 0.15)")
    p.add_argument("--cluster-method",   default="ward",
                   choices=["ward", "average", "complete", "single"],
                   help="Linkage method for hierarchical clustering (default: ward)")
    p.add_argument("--select-by",        default="variance",
                   choices=["variance", "missing"],
                   help="Representative selection strategy per cluster: "
                        "'variance' = keep highest-variance feature (default); "
                        "'missing'  = keep feature with fewest NaNs")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path, sep, target):
    print(f"\n📂  Loading: {path}")
    df = pd.read_csv(path, sep=sep)
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    target_series = None
    if target:
        if target not in df.columns:
            sys.exit(f"[ERROR] Target column '{target}' not found.")
        target_series = df[target]
        df = df.drop(columns=[target])
        print(f"    Target column '{target}' removed from features.")

    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"    Dropping {len(non_numeric)} non-numeric columns: "
              f"{non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
        df = df.select_dtypes("number")

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.std() > 0]
    print(f"    Features after cleanup: {df.shape[1]}")
    return df, target_series


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlations(df):
    print("\n⚙️   Computing Pearson correlation  …", end=" ", flush=True)
    pearson = df.corr(method="pearson")
    print("done")
    print("⚙️   Computing Spearman correlation …", end=" ", flush=True)
    spearman = df.corr(method="spearman")
    print("done")
    return pearson, spearman


def plot_heatmap(corr_matrix, method, out_path, dpi):
    n = len(corr_matrix)
    base       = max(12, min(60, n * 0.18))
    annot      = n <= 40
    linewidths = 0.3 if n <= 80 else 0.0
    fmt        = ".2f" if annot else ""

    fig, ax = plt.subplots(figsize=(base, base * 0.85))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix, ax=ax, cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=linewidths,
        annot=annot, fmt=fmt, annot_kws={"size": 6},
        cbar_kws={"shrink": 0.6, "label": "Correlation coefficient"},
        xticklabels=True, yticklabels=True,
    )
    if n > 60:
        step = max(1, n // 40)
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_visible(i % step == 0)
        for i, lbl in enumerate(ax.get_yticklabels()):
            lbl.set_visible(i % step == 0)

    tick_fs = max(5, min(9, 300 / n))
    ax.tick_params(axis="x", labelsize=tick_fs, rotation=90)
    ax.tick_params(axis="y", labelsize=tick_fs, rotation=0)
    ax.set_title(f"{method} Correlation Matrix  ({n} features)",
                 fontsize=14, pad=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def high_corr_pairs(corr_matrix, threshold):
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    stack = upper.stack().reset_index()
    stack.columns = ["Feature_A", "Feature_B", "Correlation"]
    flagged = stack[stack["Correlation"].abs() >= threshold].copy()
    flagged["Abs_Corr"] = flagged["Correlation"].abs()
    return flagged.sort_values("Abs_Corr", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIF
# ─────────────────────────────────────────────────────────────────────────────
def compute_vif(df, max_features, sample, threshold):
    if not HAS_STATSMODELS:
        print("\n⚠️   statsmodels not found — skipping VIF.")
        print("     Install with:  pip install statsmodels")
        return pd.DataFrame()

    data = df.copy()
    if sample and len(data) > sample:
        data = data.sample(n=sample, random_state=42)
        print(f"    Using {sample:,} row sample for VIF.")

    if data.shape[1] > max_features:
        top_cols = data.var().nlargest(max_features).index.tolist()
        data = data[top_cols]
        print(f"    VIF computed on top {max_features} highest-variance features (of {df.shape[1]}).")

    data = data.dropna()
    X = np.column_stack([np.ones(len(data)), data.values])

    print(f"⚙️   Computing VIF for {data.shape[1]} features …", end=" ", flush=True)
    vif_vals = []
    for i in range(1, X.shape[1]):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        vif_vals.append(v)
    print("done")

    vif_df = pd.DataFrame({"Feature": data.columns, "VIF": vif_vals})
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)


def plot_vif(vif_df, threshold, out_path, dpi):
    if vif_df.empty:
        return
    plot_df = vif_df.head(50).copy()
    plot_df["VIF_plot"] = plot_df["VIF"].clip(upper=100)
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.28)))
    colors = ["#d62728" if v >= threshold else "#1f77b4" for v in plot_df["VIF"]]
    bars = ax.barh(plot_df["Feature"][::-1], plot_df["VIF_plot"][::-1],
                   color=colors[::-1], edgecolor="none")
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.4,
               label=f"Threshold = {threshold}")
    ax.axvline(5, color="#ff7f0e", linestyle=":", linewidth=1.0, label="Moderate = 5")
    for bar, vif_raw in zip(bars, plot_df["VIF"][::-1]):
        lbl = f"{vif_raw:.1f}" if vif_raw < 1000 else f"{vif_raw:.0f}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", ha="left", fontsize=7)
    ax.set_xlabel("VIF  (capped at 100 for display)", fontsize=11)
    ax.set_title(f"Variance Inflation Factor — Top {n} Features\n",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
def cluster_features(pearson, method, cutoff):
    """
    Hierarchical clustering on 1 - |corr| distance matrix.
    Returns (linkage_matrix, cluster_labels_array).
    """
    corr_abs = pearson.abs()
    # distance: perfectly correlated features → 0, uncorrelated → 1
    dist_matrix = 1.0 - corr_abs
    np.fill_diagonal(dist_matrix.values, 0.0)
    dist_condensed = squareform(dist_matrix.values, checks=False)

    print(f"⚙️   Hierarchical clustering (method='{method}', cutoff={cutoff}) …",
          end=" ", flush=True)
    Z = linkage(dist_condensed, method=method)
    labels = fcluster(Z, t=cutoff, criterion="distance")
    print(f"done  →  {labels.max()} clusters found")
    return Z, labels


def select_representative(cluster_df, df_raw, strategy):
    """
    Given a DataFrame with one column 'feature' listing members of a cluster,
    return the name of the representative feature.
    strategy: 'variance' or 'missing'
    """
    members = cluster_df["feature"].tolist()
    sub = df_raw[members]
    if strategy == "variance":
        return sub.var().idxmax()
    else:  # missing — fewest NaNs
        return sub.isna().sum().idxmin()


def build_cluster_table(features, labels, df_raw, strategy):
    """
    Returns a DataFrame with columns:
        cluster_id | n_members | representative | members (comma-sep) | drop_these
    """
    rows = []
    for cid in sorted(set(labels)):
        members = [f for f, l in zip(features, labels) if l == cid]
        cluster_df = pd.DataFrame({"feature": members})
        rep = select_representative(cluster_df, df_raw, strategy)
        to_drop = [m for m in members if m != rep]
        rows.append({
            "cluster_id":     cid,
            "n_members":      len(members),
            "representative": rep,
            "members":        " | ".join(members),
            "drop_these":     " | ".join(to_drop),
        })
    return pd.DataFrame(rows).sort_values("n_members", ascending=False).reset_index(drop=True)


def plot_dendrogram(Z, features, cutoff, out_path, dpi):
    n = len(features)
    fig_w = max(16, min(60, n * 0.12))
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    dendrogram(
        Z,
        ax=ax,
        labels=features,
        leaf_rotation=90,
        leaf_font_size=max(3, min(7, 300 / n)),
        color_threshold=cutoff,
        above_threshold_color="#888888",
    )
    ax.axhline(y=cutoff, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Cut-off = {cutoff}  (≈ |r| ≥ {1-cutoff:.2f})")
    ax.set_title("Hierarchical Clustering Dendrogram\n",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance  (1 − |Pearson r|)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_clustered_heatmap(pearson, labels, features, out_path, dpi):
    """Reordered heatmap so members of the same cluster sit together."""
    # sort features by cluster label
    order = sorted(range(len(features)), key=lambda i: labels[i])
    ordered_feats = [features[i] for i in order]
    ordered_corr = pearson.loc[ordered_feats, ordered_feats]

    n = len(ordered_feats)
    base = max(12, min(60, n * 0.18))
    fig, ax = plt.subplots(figsize=(base, base * 0.85))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        ordered_corr, ax=ax, cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.0,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        xticklabels=False, yticklabels=False,
    )

    # draw cluster boundary lines
    boundaries = []
    prev = labels[order[0]]
    count = 0
    for idx in order:
        count += 1
        if labels[idx] != prev:
            boundaries.append(count - 1)
            prev = labels[idx]
    for b in boundaries:
        ax.axhline(y=b, color="white", linewidth=0.6, alpha=0.8)
        ax.axvline(x=b, color="white", linewidth=0.6, alpha=0.8)

    ax.set_title(f"Clustered Pearson Heatmap  ({labels.max()} clusters, {n} features)\n",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def plot_cluster_sizes(cluster_table, out_path, dpi):
    df = cluster_table[cluster_table["n_members"] > 1].copy()
    if df.empty:
        return
    df = df.head(40)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.3)))

    bars = ax.barh(
        df["representative"][::-1],
        df["n_members"][::-1],
        color="#4c72b0", edgecolor="none",
    )
    for bar, val in zip(bars, df["n_members"][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", ha="left", fontsize=8)

    ax.set_xlabel("Number of features in cluster", fontsize=11)
    ax.set_title(f"Largest Feature Clusters  (top {n} with >1 member)\n",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Reports & summary
# ─────────────────────────────────────────────────────────────────────────────
def save_reports(out_dir, high_p, high_s, vif_df, cluster_table):
    if not high_p.empty:
        p = out_dir / "high_pearson_pairs.csv"
        high_p.to_csv(p, index=False)
        print(f"    Saved → {p}")
    if not high_s.empty:
        p = out_dir / "high_spearman_pairs.csv"
        high_s.to_csv(p, index=False)
        print(f"    Saved → {p}")
    if not vif_df.empty:
        p = out_dir / "vif_report.csv"
        vif_df.to_csv(p, index=False)
        print(f"    Saved → {p}")
    if not cluster_table.empty:
        p = out_dir / "cluster_report.csv"
        cluster_table.to_csv(p, index=False)
        print(f"    Saved → {p}")

        # also save a flat list of features to KEEP (one representative per cluster)
        keep = cluster_table["representative"].tolist()
        pd.DataFrame({"keep_feature": keep}).to_csv(
            out_dir / "features_to_keep.csv", index=False
        )
        print(f"    Saved → {out_dir / 'features_to_keep.csv'}")


def print_summary(df, high_p, high_s, vif_df, cluster_table, args):
    n = len(df.columns)
    n_clusters = cluster_table["cluster_id"].nunique() if not cluster_table.empty else "n/a"
    n_keep     = len(cluster_table) if not cluster_table.empty else "n/a"
    n_drop     = n - n_keep if isinstance(n_keep, int) else "n/a"

    print("\n" + "═" * 64)
    print("  SUMMARY")
    print("═" * 64)
    print(f"  Features analysed           : {n}")
    print(f"  Corr threshold              : |r| ≥ {args.corr_threshold}")
    print(f"  High-corr pairs (Pearson)   : {len(high_p)}")
    print(f"  High-corr pairs (Spearman)  : {len(high_s)}")

    print(f"\n  ── Clustering (cutoff={args.cluster_cutoff}, method={args.cluster_method}) ──")
    print(f"  Clusters found              : {n_clusters}")
    print(f"  Representatives to KEEP     : {n_keep}")
    print(f"  Features you can DROP       : {n_drop}")

    if not cluster_table.empty:
        big = cluster_table[cluster_table["n_members"] > 1]
        print(f"\n  Largest clusters (top 10):")
        for _, row in big.head(10).iterrows():
            print(f"    Cluster {row['cluster_id']:>4}  "
                  f"({row['n_members']:>3} members)  keep → {row['representative']}")

    if not vif_df.empty:
        flagged = vif_df[vif_df["VIF"] >= args.vif_threshold]
        print(f"\n  VIF threshold               : {args.vif_threshold}")
        print(f"  Features with VIF ≥ {args.vif_threshold}      : {len(flagged)}")
        if not flagged.empty:
            print("  Top 10 by VIF:")
            for _, row in vif_df.head(10).iterrows():
                flag = " ⚠" if row["VIF"] >= args.vif_threshold else ""
                print(f"    {row['Feature']:<40} VIF = {row['VIF']:.2f}{flag}")

    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df, _ = load_data(args.input, args.sep, args.target)
    if df.shape[1] < 2:
        sys.exit("[ERROR] Need at least 2 numeric features.")

    # 2. Correlations
    pearson, spearman = compute_correlations(df)

    # 3. Heatmaps
    print("\n🎨  Plotting raw heatmaps …")
    plot_heatmap(pearson,  "Pearson",  out_dir / "heatmap_pearson.png",  args.dpi)
    plot_heatmap(spearman, "Spearman", out_dir / "heatmap_spearman.png", args.dpi)

    # 4. High-correlation pairs
    print("\n🔍  Finding high-correlation pairs …", end=" ", flush=True)
    high_p = high_corr_pairs(pearson,  args.corr_threshold)
    high_s = high_corr_pairs(spearman, args.corr_threshold)
    print(f"done  ({len(high_p)} Pearson, {len(high_s)} Spearman pairs flagged)")

    # 5. VIF
    print("\n📐  Variance Inflation Factor …")
    vif_df = compute_vif(df, args.max_vif_features, args.sample, args.vif_threshold)
    if not vif_df.empty:
        plot_vif(vif_df, args.vif_threshold, out_dir / "vif_barplot.png", args.dpi)

    # 6. Hierarchical clustering
    print("\n🌳  Feature clustering …")
    features = list(pearson.columns)
    Z, labels = cluster_features(pearson, args.cluster_method, args.cluster_cutoff)

    print("🎨  Plotting dendrogram …")
    plot_dendrogram(Z, features, args.cluster_cutoff,
                    out_dir / "cluster_dendrogram.png", args.dpi)

    print("🎨  Plotting clustered heatmap …")
    plot_clustered_heatmap(pearson, labels, features,
                           out_dir / "heatmap_clustered.png", args.dpi)

    print("⚙️   Building cluster table …", end=" ", flush=True)
    cluster_table = build_cluster_table(features, labels, df, args.select_by)
    print("done")

    print("🎨  Plotting cluster sizes …")
    plot_cluster_sizes(cluster_table, out_dir / "cluster_sizes.png", args.dpi)

    # 7. CSV reports
    print("\n💾  Saving reports …")
    save_reports(out_dir, high_p, high_s, vif_df, cluster_table)

    # 8. Summary
    print_summary(df, high_p, high_s, vif_df, cluster_table, args)
    print(f"✅  All outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()