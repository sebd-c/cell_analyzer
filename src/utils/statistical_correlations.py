#!/usr/bin/env python3
"""
Feature Correlation Analysis
------------------------------
Generates Pearson and Spearman correlation heatmaps + VIF report
for datasets with large numbers of features (~300+).

Usage:
    python feature_correlation_analysis.py --input data.csv
    python feature_correlation_analysis.py --input data.csv --target label_column
    python feature_correlation_analysis.py --input data.csv --target label --corr-threshold 0.85 --vif-threshold 10
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
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ── optional statsmodels for VIF ────────────────────────────────────────────
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Correlation heatmaps + VIF for wide datasets.")
    p.add_argument("--input",           required=True,  help="Path to CSV file")
    p.add_argument("--target",          default=None,   help="Target/label column to exclude (optional)")
    p.add_argument("--sep",             default=",",    help="CSV separator (default: ',')")
    p.add_argument("--corr-threshold",  type=float, default=0.90,
                   help="Flag feature pairs with |corr| >= this value (default: 0.90)")
    p.add_argument("--vif-threshold",   type=float, default=10.0,
                   help="Flag features with VIF >= this value (default: 10)")
    p.add_argument("--max-vif-features",type=int,   default=100,
                   help="Max features for VIF calculation (expensive; default: 100)")
    p.add_argument("--sample",          type=int,   default=None,
                   help="Row sample for faster VIF on large datasets")
    p.add_argument("--out-dir",         default=".",
                   help="Directory to save outputs (default: current dir)")
    p.add_argument("--dpi",             type=int,   default=150,
                   help="DPI for saved figures (default: 150)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str, sep: str, target: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
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

    # keep only numeric
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"    Dropping {len(non_numeric)} non-numeric columns: {non_numeric[:5]}{'...' if len(non_numeric)>5 else ''}")
        df = df.select_dtypes("number")

    # drop constant / all-NaN columns
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.std() > 0]
    print(f"    Features after cleanup: {df.shape[1]}")
    return df, target_series


def compute_correlations(df: pd.DataFrame):
    print("\n   Computing Pearson correlation  …", end=" ", flush=True)
    pearson = df.corr(method="pearson")
    print("done")
    print("Computing Spearman correlation …", end=" ", flush=True)
    spearman = df.corr(method="spearman")
    print("done")
    return pearson, spearman


def plot_heatmap(corr_matrix: pd.DataFrame, method: str, out_path: Path, dpi: int):
    n = len(corr_matrix)

    # --- adaptive figure & font sizes ---
    base    = max(12, min(60, n * 0.18))
    annot   = n <= 40
    linewidths = 0.3 if n <= 80 else 0.0
    fmt     = ".2f" if annot else ""

    fig, ax = plt.subplots(figsize=(base, base * 0.85))

    sns.heatmap(
        corr_matrix,
        ax=ax,
        vmin=-1, vmax=1, center=0,
        square=True,
        linewidths=linewidths,
        annot=annot, fmt=fmt, annot_kws={"size": 6},
        cbar_kws={"shrink": 0.6, "label": "Correlation coefficient"},
        xticklabels=True, yticklabels=True,
    )

    # tick labels: hide most when too many features
    if n > 60:
        step = max(1, n // 40)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_visible(i % step == 0)

    tick_fs = max(5, min(9, 300 / n))
    ax.tick_params(axis="x", labelsize=tick_fs, rotation=90)
    ax.tick_params(axis="y", labelsize=tick_fs, rotation=0)

    ax.set_title(f"{method} Correlation Matrix  ({n} features)", fontsize=14, pad=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def high_corr_pairs(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    stack = upper.stack().reset_index()
    stack.columns = ["Feature_A", "Feature_B", "Correlation"]
    flagged = stack[stack["Correlation"].abs() >= threshold].copy()
    flagged["Abs_Corr"] = flagged["Correlation"].abs()
    return flagged.sort_values("Abs_Corr", ascending=False).reset_index(drop=True)


def compute_vif(df: pd.DataFrame, max_features: int, sample: int | None, threshold: float) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        print("\n statsmodels not found — skipping VIF.")
        print("     Install with:  pip install statsmodels")
        return pd.DataFrame()

    data = df.copy()
    if sample and len(data) > sample:
        data = data.sample(n=sample, random_state=42)
        print(f"    Using {sample:,} row sample for VIF.")

    # limit to max_features (select highest-variance ones)
    if data.shape[1] > max_features:
        top_cols = data.var().nlargest(max_features).index.tolist()
        data = data[top_cols]
        print(f"    VIF computed on top {max_features} highest-variance features (of {df.shape[1]}).")

    data = data.dropna()
    # add constant
    X = data.values
    X = np.column_stack([np.ones(len(X)), X])

    print(f" Computing VIF for {data.shape[1]} features …", end=" ", flush=True)
    vif_vals = []
    for i in range(1, X.shape[1]):          # skip constant at index 0
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        vif_vals.append(v)
    print("done")

    vif_df = pd.DataFrame({"Feature": data.columns, "VIF": vif_vals})
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


def plot_vif(vif_df: pd.DataFrame, threshold: float, out_path: Path, dpi: int):
    if vif_df.empty:
        return

    top_n = 50
    plot_df = vif_df.head(top_n).copy()
    # cap display at 100 so the chart stays readable
    plot_df["VIF_plot"] = plot_df["VIF"].clip(upper=100)

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.28)))

    colors = ["#d62728" if v >= threshold else "#1f77b4" for v in plot_df["VIF"]]
    bars = ax.barh(plot_df["Feature"][::-1], plot_df["VIF_plot"][::-1], color=colors[::-1], edgecolor="none")

    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.4, label=f"Threshold = {threshold}")
    ax.axvline(5,         color="#ff7f0e", linestyle=":",  linewidth=1.0, label="Moderate = 5")

    for bar, vif_raw in zip(bars, plot_df["VIF"][::-1]):
        label = f"{vif_raw:.1f}" if vif_raw < 1000 else f"{vif_raw:.0f}"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                label, va="center", ha="left", fontsize=7)

    ax.set_xlabel("VIF  (capped at 100 for display)", fontsize=11)
    ax.set_title(f"Variance Inflation Factor — Top {n} Features\n"
                 f"(Red bars ≥ {threshold} indicate strong multicollinearity)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")


def save_reports(out_dir: Path, high_p: pd.DataFrame, high_s: pd.DataFrame,
                 vif_df: pd.DataFrame, threshold: float, vif_threshold: float):
    # high correlation pairs
    if not high_p.empty:
        p = out_dir / "high_pearson_pairs.csv"
        high_p.to_csv(p, index=False)
        print(f"    Saved → {p}")
    if not high_s.empty:
        p = out_dir / "high_spearman_pairs.csv"
        high_s.to_csv(p, index=False)
        print(f"    Saved → {p}")
    # VIF
    if not vif_df.empty:
        p = out_dir / "vif_report.csv"
        vif_df.to_csv(p, index=False)
        print(f"    Saved → {p}")


def print_summary(df, pearson, spearman, high_p, high_s, vif_df, args):
    n = len(df.columns)
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)
    print(f"  Features analysed      : {n}")
    print(f"  Correlation threshold  : |r| ≥ {args.corr_threshold}")
    print(f"  High-corr pairs (Pearson)  : {len(high_p)}")
    print(f"  High-corr pairs (Spearman) : {len(high_s)}")

    if not vif_df.empty:
        flagged_vif = vif_df[vif_df["VIF"] >= args.vif_threshold]
        print(f"\n  VIF threshold          : {args.vif_threshold}")
        print(f"  Features with VIF ≥ {args.vif_threshold} : {len(flagged_vif)}")
        if not flagged_vif.empty:
            print(f"\n  Top 10 by VIF:")
            for _, row in vif_df.head(10).iterrows():
                flag = " " if row["VIF"] >= args.vif_threshold else ""
                print(f"    {row['Feature']:<40} VIF = {row['VIF']:.2f}{flag}")

    if not high_p.empty:
        print(f"\n  Top 10 Pearson pairs (|r| ≥ {args.corr_threshold}):")
        for _, row in high_p.head(10).iterrows():
            print(f"    {row['Feature_A']:<25} ↔  {row['Feature_B']:<25}  r = {row['Correlation']:+.4f}")

    print("═"*60 + "\n")


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
    print("\n Plotting heatmaps …")
    plot_heatmap(pearson,  "Pearson",  out_dir / "heatmap_pearson.png",  args.dpi)
    plot_heatmap(spearman, "Spearman", out_dir / "heatmap_spearman.png", args.dpi)

    # 4. High-correlation pairs
    print("\n🔍  Finding high-correlation pairs …", end=" ", flush=True)
    high_p = high_corr_pairs(pearson,  args.corr_threshold)
    high_s = high_corr_pairs(spearman, args.corr_threshold)
    print(f"done  ({len(high_p)} Pearson, {len(high_s)} Spearman pairs flagged)")

    # 5. VIF
    print("\n Variance Inflation Factor …")
    vif_df = compute_vif(df, args.max_vif_features, args.sample, args.vif_threshold)
    if not vif_df.empty:
        plot_vif(vif_df, args.vif_threshold, out_dir / "vif_barplot.png", args.dpi)

    # 6. CSV reports
    print("\n Saving CSV reports …")
    save_reports(out_dir, high_p, high_s, vif_df, args.corr_threshold, args.vif_threshold)

    # 7. Summary
    print_summary(df, pearson, spearman, high_p, high_s, vif_df, args)
    print(f"All outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()