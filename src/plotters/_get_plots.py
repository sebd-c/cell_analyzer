# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis
from typing import Optional
import os


##############################################################################################

# TODO: fix tree plotting here

# def draw_tree()->None:
#     dot_data = StringIO()
#     export_graphviz(clf, out_file=dot_data,
#                     feature_names=feature_cols,
#                     class_names=class_names,
#                     filled=True, rounded=True,
#                     special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png('decision_tree.png')
#     pass


def plot_cv_metrics(scores_df: pd.DataFrame,
                    output_dir: str,
                    file_format: str,
                    model_prefix: str
                    ) -> None:
    """
    Plot train and test cross-validation metrics as simple boxplots.
    """
    train_cols = ["train_balanced_accuracy", "train_recall_macro",
                  "train_precision_macro", "train_f1_macro"]
    test_cols = ["test_balanced_accuracy", "test_recall_macro",
                 "test_precision_macro", "test_f1_macro"]
    metric_labels = ["Balanced\naccuracy", "Macro\nrecall",
                     "Macro\nprecision", "Macro F1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].boxplot(scores_df[train_cols], tick_labels=metric_labels, showmeans=True)
    axes[0].set_title("Train")
    axes[0].set_ylabel("Score")

    axes[1].boxplot(scores_df[test_cols], tick_labels=metric_labels, showmeans=True)
    axes[1].set_title("Validation")

    plt.tight_layout()

    filename = f"{model_prefix}_cv_metrics_boxplot.{file_format}"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)


def plot_feature_summary_heatmap(df: pd.DataFrame,
                                 output_dir: str,
                                 file_format: str,
                                 ) -> None:
    """
    Plot a heatmap of summary statistics across all features.
    """
    summary = pd.DataFrame({"mean": df.mean(),
                            "std": df.std(),
                            "skewness": df.apply(skew),
                            "kurtosis": df.apply(kurtosis),
                            "missing%": df.isnull().mean() * 100,
                            "zeros%": (df == 0).mean() * 100,
                            })

    # order to easy visualization
    summary = summary.sort_values("skewness")

    # Normalize each column so all stats are on the same scale
    normalized = (summary - summary.mean()) / summary.std()

    fig, ax = plt.subplots(figsize=(10, max(8, len(df.columns) // 5)))
    sns.heatmap(normalized,
                ax=ax,
                center=0,
                linewidths=0.3,
                yticklabels=True,
                )

    ax.set_title("Feature Distribution Summary (normalized)")
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"feature_summary_heatmap.{file_format}")
    fig.savefig(filepath)
    plt.close(fig)


def plot_select_k_best_scores(features_df: pd.DataFrame,
                              output_dir: str,
                              file_format: str,
                              model_prefix: str,
                              top_n: Optional[int] = None,
                              ) -> None:
    """
    Plot SelectKBest scores for the selected features.
    """
    plot_df = features_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["feature"], plot_df["score"], color="steelblue")
    ax.set_xlabel("SelectKBest score")
    ax.set_ylabel("Feature")
    ax.set_title("Selected feature scores")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir,
                            f"{model_prefix}_selected_feature_scores.{file_format}")
    fig.savefig(filepath)
    plt.close(fig)


def binary_att_proportions(df, vartypes_dic):
    binary_cols = []
    for col in vartypes_dic:
        if vartypes_dic[col] == "binary":
            binary_cols.append(col)

    proportions = []
    for col in binary_cols:
        count = df[col].value_counts(normalize=True)
        prop_1 = count.get(1, 0)
        prop_0 = count.get(0, 0)
        proportions.append((prop_0, prop_1))

    plot_df = pd.DataFrame(proportions, index=binary_cols, columns=["0", "1"])
    fig, ax = plt.subplots(figsize=(12, max(6, len(binary_cols) * 0.3)))
    plot_df.plot(kind="barh", stacked=True, ax=ax, color=["red", "green"])

    ax.set_xlabel("Proportion")
    ax.set_ylabel("Feature")
    ax.set_title("Binarty atts proprotions")
    ax.legend(["0", "1"])

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if row["0"] > 0.05:
            ax.text(row["0"] / 2, i, f"{row['0']:.1%}", ha="center", va="center")
        if row["1"] > 0.05:
            ax.text(
                row["0"] + row["1"] / 2, i, f"{row['1']:.1%}", ha="center", va="center"
            )

    return plt.show()


def grouping_numeric(df, vartypes_dic, groups):
    numerical_cols = []
    for col in vartypes_dic:
        if vartypes_dic[col] == "numerical":
            numerical_cols.append(col)

    if len(numerical_cols) == 0:
        return []

    attr_ranges = []
    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) > 0:
            amplitude = data.max() - data.min()
            attr_ranges.append((col, amplitude))
        else:
            attr_ranges.append((col, 0))

    attr_ranges.sort(key=lambda x: x[1])

    n_attrs = len(attr_ranges)
    group_size = n_attrs // groups

    numeric_groups = []
    start = 0

    for i in range(groups):
        if i == groups - 1:
            end = n_attrs
        else:
            end = start + group_size

        group_attrs = [attr[0] for attr in attr_ranges[start:end]]
        numeric_groups.append(group_attrs)
        start = end

    return numeric_groups


def boxplot_numerical(df, vartypes_dic, groups_n, ignore_attribut=None):
    numeric_groups = grouping_numeric(df, vartypes_dic, groups_n)
    if ignore_attribut is not None:
        # remove the ignore attribute from the group
        numeric_groups = [[attr for attr in group if attr != ignore_attribut] for group in numeric_groups]

    for i, group in enumerate(numeric_groups):
        figsize = (max(12, len(group) * 0.5), 6)
        fig, ax = plt.subplots(figsize=figsize)

        df[group].boxplot(ax=ax, rot=90, fontsize=8)

        ax.set_title(f"Boxplot - Grupo {i + 1} ({len(group)} atributos)", fontsize=14)
        ax.set_xlabel("Atributos", fontsize=12)
        ax.set_ylabel("Valores", fontsize=12)

        plt.tight_layout()
        plt.show()


def plot_binary_heatmap(df, vartypes_dic):
    binary_cols = [col for col in vartypes_dic if vartypes_dic[col] == "binary"]

    # Calculate proportion of 1's for each variable
    proportions = df[binary_cols].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))
    proportions.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Proportion of 1s')
    ax.set_title(f'Distribution of {len(binary_cols)} Binary Variables')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return proportions
