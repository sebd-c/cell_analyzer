# -------------------------------------------------------------
# imports
# --------------------------------------------------------------
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_samples,
                             silhouette_score,
                             )
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    UMAP_IMPORT_ERROR = None
except Exception as exc:
    UMAP = None
    UMAP_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LINKAGE_METHOD  = "ward"
RANDOM_STATE    = 42

# Edit these values, then run:
# python3 -m src.models.clustering.get_clustering
INPUT_PATH      = "/home/debs/Documents/work/data/u87/duplo_marcada/HetSenCellDB/whole_image/joined/tabular/all_images/cytnuc_morphofeats.csv"
OUTPUT_DIR      = "/home/debs/Documents/work/data/u87/duplo_marcada/HetSenCellDB/whole_image/joined/clustering2"
LABEL_COLUMN    = "label"
N_CLUSTERS      = 4
K_MIN           = 2
K_MAX           = 10

# ---------------------------------------------------------------------------
# k-selection helpers
# ---------------------------------------------------------------------------

# elbow
def plot_elbow(X_scaled: np.ndarray,
               k_range: range = range(2, 11),
               output_path: str = "elbow.png",
               ) -> pd.DataFrame:
    """
    Fit KMeans for each k in k_range,
    record inertia,
    and plot the elbow curve.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k,
                    random_state=RANDOM_STATE,
                    )
        km.fit(X_scaled)
        records.append({"k": k, "inertia": km.inertia_})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["inertia"], marker="o",
            color="#185FA5", linewidth=1.8, markersize=6)
    ax.set_xlabel("Number of Clusters k", fontsize=11)
    ax.set_ylabel("Inertia within cluster", fontsize=11)
    ax.set_title("Elbow Curve", fontsize=12)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"elbow plot saved → {output_path}")
    return df

# silhouette
def plot_silhouette_scores(X_scaled: np.ndarray,
                           k_range: range = range(2, 11),
                           output_path: str = "silhouette_scores.png",
                           ) -> pd.DataFrame:
    """
    Compute mean silhouette score for each k and plot.
    """
    records = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init="auto").fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        records.append({"k": k, "silhouette_mean": round(score, 4)})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["silhouette_mean"], marker="o",
            color="#0F6E56", linewidth=1.8, markersize=6)
    ax.axhline(0, color="#E24B4A", lw=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of Clusters k", fontsize=11)
    ax.set_ylabel("Mean Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Scores", fontsize=12)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"silhouette scores plot saved → {output_path}")
    return df

# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_kmeans(X_scaled: np.ndarray,
               n_clusters: int,
               random_state: int = RANDOM_STATE,
               ) -> KMeans:
    """
    Fit KMeans and return the fitted model (not just labels),
    so that inertia and cluster centers are accessible downstream.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X_scaled)
    return km


def evaluate_clustering(X_scaled: np.ndarray,
                        labels: np.ndarray,
                        model_name: str = "kmeans",
                        ) -> dict:
    """
    Compute silhouette, Davies-Bouldin, and Calinski-Harabasz scores
    for KMeans labels.
    """
    n_clusters = len(set(labels))

    return {"model": model_name,
            "n_clusters": n_clusters,
            "silhouette": round(silhouette_score(X_scaled, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
            }


def plot_kmeans_clusters(X_scaled: np.ndarray,
                         labels: np.ndarray,
                         output_path: str = "kmeans_clusters_pca.png",
                         random_state: int = RANDOM_STATE,
                         ) -> pd.DataFrame:
    """
    Project scaled features to 2 PCA dimensions and plot final KMeans labels.
    """
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "pc1": components[:, 0],
        "pc2": components[:, 1],
        "cluster": labels,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pca_df["cluster"],
        cmap="tab10",
        s=36,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    ax.set_title("KMeans Clusters (PCA Projection)", fontsize=12)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"kmeans cluster plot saved @ {output_path}")
    return pca_df


def fit_umap_embedding(X_scaled: np.ndarray,
                       n_neighbors: int = 15,
                       min_dist: float = 0.1,
                       metric: str = "euclidean",
                       n_components: int = 2,
                       random_state: int = RANDOM_STATE,
                       **umap_kwargs,
                       ) -> np.ndarray:
    """
    Fit a UMAP embedding for an already-scaled feature matrix.
    """

    reducer = UMAP(n_components=n_components,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   metric=metric,
                   random_state=random_state,
                   **umap_kwargs,
                   )
    return reducer.fit_transform(X_scaled)


def plot_umap_parameter_grid(X_scaled: np.ndarray,
                             labels: np.ndarray | None = None,
                             n_neighbors_values: tuple[int, ...] = (5, 15, 30, 100),
                             min_dist_values: tuple[float, ...] = (0.0, 0.1, 0.5),
                             metric_values: tuple[str, ...] = ("euclidean", "canberra", "mahalanobis", "cosine", "correlation"),
                             output_path: str = "umap_parameter_grid.png",
                             random_state: int = RANDOM_STATE,
                             point_size: float = 18,
                             alpha: float = 0.85,
                             cmap: str = "tab10",
                             **umap_kwargs,
                             ) -> pd.DataFrame:
    """
    Plot a grid of UMAP embeddings across parameter combinations.

    Rows vary by metric, n_neighbors, and n_components; columns vary by min_dist.
    Returns one dataframe with all embedding coordinates and parameters.
    """
    if any(component_count < 2 for component_count in n_components):
        raise ValueError("All n_components values must be >= 2 for 2D plotting.")

    color_values = None
    if labels is not None:
        label_values = np.asarray(labels)
        if np.issubdtype(label_values.dtype, np.number):
            color_values = label_values
        else:
            color_values = pd.factorize(label_values)[0]

    n_rows = len(metric_values) * len(n_neighbors_values) * len(n_components)
    n_cols = len(min_dist_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    records = []
    for row_idx, metric in enumerate(metric_values):
        for neighbor_idx, n_neighbors in enumerate(n_neighbors_values):
            for component_idx, component_count in enumerate(n_components):
                grid_row = (
                    row_idx * len(n_neighbors_values) * len(n_components)
                    + neighbor_idx * len(n_components)
                    + component_idx
                )

                for col_idx, min_dist in enumerate(min_dist_values):
                    ax = axes[grid_row][col_idx]
                    embedding = fit_umap_embedding(
                        X_scaled,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        n_components=component_count,
                        random_state=random_state,
                        **umap_kwargs,
                    )

                    plot_kwargs = {
                        "s": point_size,
                        "alpha": alpha,
                        "edgecolors": "none",
                    }
                    if labels is None:
                        ax.scatter(embedding[:, 0], embedding[:, 1], color="#185FA5", **plot_kwargs)
                    else:
                        ax.scatter(embedding[:, 0], embedding[:, 1], c=color_values, cmap=cmap, **plot_kwargs)

                    ax.set_title(
                        f"metric={metric}\nn_neighbors={n_neighbors}, min_dist={min_dist}, n_components={component_count}",
                        fontsize=10,
                    )
                    ax.set_xlabel("UMAP 1", fontsize=9)
                    ax.set_ylabel("UMAP 2", fontsize=9)
                    ax.tick_params(axis="both", labelsize=8)

                    plot_df = pd.DataFrame({
                        f"umap{component_num + 1}": embedding[:, component_num]
                        for component_num in range(component_count)
                    })
                    plot_df["metric"] = metric
                    plot_df["n_neighbors"] = n_neighbors
                    plot_df["min_dist"] = min_dist
                    plot_df["n_components"] = component_count
                    if labels is not None:
                        plot_df["label"] = labels
                    records.append(plot_df)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"umap parameter grid saved @ {output_path}")
    return pd.concat(records, ignore_index=True)


def build_cluster_profile(X_original: pd.DataFrame,
                          labels: np.ndarray,
                          model_name: str = "kmeans",
                          output_path: str = "cluster_profile.csv",
                          ) -> pd.DataFrame:
    """
    Compute per-cluster descriptive statistics on the original
    (unscaled) feature values and save to CSV.
    """
    df = X_original.copy()
    df["cluster"] = labels

    profile = (df.groupby("cluster").agg(["mean", "median", "std", "min", "max", "count"]))
    profile.columns = ["_".join(c) for c in profile.columns]
    profile = profile.reset_index()
    profile["cluster"] = profile["cluster"].astype(str)
    profile["model"] = model_name
    profile = profile[["model"] + [c for c in profile.columns if c != "model"]]

    profile.to_csv(output_path, index=False)
    print(f" cluster profile saved @ {output_path}")
    return profile


def select_numeric_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep only discrete/continuous numeric feature columns.

    Object columns such as filenames, tuple strings, and contour data are
    intentionally excluded from clustering.
    """
    numeric_cols = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])
    ]
    dropped_cols = [col for col in X.columns if col not in numeric_cols]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    return X[numeric_cols].copy(), dropped_cols


def run_clustering_pipeline(X: pd.DataFrame,
                            output_dir: str = "clustering",
                            k_range: range = range(2, 11),
                            n_clusters: int | None = None,
                            random_state: int = RANDOM_STATE,
                            ) -> dict:
    """
    Full KMeans clustering pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    X_features, dropped_feature_cols = select_numeric_features(X)

    if dropped_feature_cols:
        dropped_path = os.path.join(output_dir, "excluded_non_numeric_columns.csv")
        pd.DataFrame({"excluded_column": dropped_feature_cols}).to_csv(dropped_path, index=False)
        print(f"excluded {len(dropped_feature_cols)} non-numeric columns @ {dropped_path}")

    feature_cols_path = os.path.join(output_dir, "selected_feature_columns.csv")
    pd.DataFrame({"feature_column": X_features.columns}).to_csv(feature_cols_path, index=False)
    print(f"using {len(X_features.columns)} numeric feature columns @ {feature_cols_path}")

    X_scaled = StandardScaler().fit_transform(X_features)

    print("\n── k-selection plots ──")
    elbow_df = plot_elbow(X_scaled, k_range,
                          output_path=os.path.join(output_dir, "elbow.png"),
                          )
    elbow_df.to_csv(os.path.join(output_dir, "elbow.csv"), index=False)

    sil_df = plot_silhouette_scores(X_scaled, k_range,
                                    output_path=os.path.join(output_dir, "silhouette_scores.png"),
                                    )
    sil_df.to_csv(os.path.join(output_dir, "silhouette_scores.csv"), index=False)

    print("\n── fitting KMeans ──")
    km     = fit_kmeans(X_scaled, n_clusters, random_state)
    labels = km.labels_

    labels_df = X.copy()
    labels_df["cluster"] = labels
    labels_path = os.path.join(output_dir, "cluster_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"cluster labels saved @ {labels_path}")

    plot_df = plot_kmeans_clusters(X_scaled, labels,
                                   output_path=os.path.join(output_dir, "kmeans_clusters_pca.png"),
                                   random_state=random_state,
                                   )
    plot_df.to_csv(os.path.join(output_dir, "kmeans_clusters_pca.csv"), index=False)

    print("\n── UMAP parameter grid ──")
    umap_grid_df = plot_umap_parameter_grid(
        X_scaled,
        labels=labels,
        output_path=os.path.join(output_dir, "umap_parameter_grid.png"),
        random_state=random_state,
    )
    umap_grid_df.to_csv(os.path.join(output_dir, "umap_parameter_grid.csv"), index=False)

    print("\n── evaluation metrics ──")
    metrics = evaluate_clustering(X_scaled, labels)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "clustering_metrics.csv"), index=False)
    print(f" metrics saved")
    print(metrics_df.to_string(index=False))

    print("\n── per-cluster profile ──")
    profile = build_cluster_profile(X_original=X_features,
        labels=labels,
        output_path=os.path.join(output_dir, "cluster_profile.csv"),
    )

    return {
        "model":   km,
        "labels":  labels,
        "plot":    plot_df,
        "umap_grid": umap_grid_df,
        "metrics": metrics_df,
        "profile": profile,
        "feature_columns": list(X_features.columns),
        "excluded_columns": dropped_feature_cols,
    }


def main() -> None:
    """
    Run clustering from the editable constants at the top of this file.
    """
    data = pd.read_csv(INPUT_PATH)

    if LABEL_COLUMN in data.columns:
        X = data.drop(columns=[LABEL_COLUMN])
    else:
        X = data

    run_clustering_pipeline(
        X=X,
        output_dir=OUTPUT_DIR,
        k_range=range(K_MIN, K_MAX + 1),
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    main()
