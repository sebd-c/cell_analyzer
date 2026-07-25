import math
import os

import dalex as dx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#####################################################################
# ----------------------------------------------------------------
# Permutation Importance
# ----------------------------------------------------------------
def run_permutation_importance(clf,
                               X_train,
                               y_train,
                               feature_names: list,
                               n_repeats: int = 30,
                               random_state: int = 42,
                               output_path: str = ""
                               ) -> pd.DataFrame:
    """
    Run permutation importance and save results to CSV.
    """
    r = permutation_importance(clf, X_train, y_train,
                               n_repeats=n_repeats,
                               random_state=random_state)

    df = pd.DataFrame({"feature": feature_names,
                       "importance": r.importances_mean,
                       "std": r.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["importance_minus_2std"] = df["importance"] - 2 * df["std"]

    df.to_csv(output_path, index=False)
    print(f"Permutation importance saved -> {output_path}")
    return df


def plot_permutation_importance(df: pd.DataFrame,
                                model_prefix: str,
                                output_path: str
                                ) -> None:
    """
    Horizontal bar chart of permutation importance with ±1 std error bars.
    Bars are colored by whether importance − 2×std > 0 (reliable vs unreliable).
    """
    colors = ["#185FA5" if v > 0 else "#888780"
              for v in df["importance_minus_2std"]]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.5))
    ax.barh(df["feature"], df["importance"], xerr=df["std"],
            color=colors, capsize=4,
            error_kw={"ecolor": "#5F5E5A", "lw": 1.5})
    ax.axvline(0, color="#E24B4A", lw=1.5, linestyle="--",
               label="mean − 2×std = 0")
    ax.set_xlabel("mean importance")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Permutation importance plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Ceteris paribus (dalex)
# ---------------------------------------------------------------------------

def run_ceteris_paribus(exp: dx.Explainer,
                        X: pd.DataFrame,
                        y: pd.Series,
                        observation_indices: list[int] | None = None,
                        output_dir: str = "cp_profiles",
                        combined_path: str = "cp_profiles_combined.csv",
                        ) -> pd.DataFrame:
    """
    Compute ceteris paribus profiles for a list of observations.
    Profiles are fixed to DIA+
    Saves one CSV per observation and a combined CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    indices = observation_indices or list(range(len(X)))
    all_dfs = []

    for idx in indices:
        observation = X.iloc[idx]
        predicted_class = int(exp.predict(observation.to_frame().T)[0] > 0.5)

        cp = exp.predict_profile(observation)
        df = cp.result.copy()

        df["observation_idx"] = idx
        df["predicted_class"] = predicted_class
        df["true_class"] = int(y.iloc[idx])

        obs_path = os.path.join(output_dir, f"cp_obs_{idx}.csv")
        df.to_csv(obs_path, index=False)
        all_dfs.append(df)

        print(f"obs {idx} "
              f"(true={y.iloc[idx]}, pred={predicted_class}) → {obs_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"combined CP profile saved → {combined_path}")
    return combined


# ----------------------------------------------------------------
# ICE + PDP
# ----------------------------------------------------------------
def run_ice_plots(model,
                  X_train: pd.DataFrame,
                  feature_list: list,
                  output_dir: str,
                  n_cols: int = 2,
                  subsample: float = 0.3,
                  random_state: int = 42,
                  model_prefix: str = "",
                  n_jobs: int | None = -1,
                  joblib_backend: str = "threading",
                  ) -> None:
    """
    Generate ICE + PDP mean plots
    """

    n_rows = math.ceil(len(feature_list) / n_cols)

    pdp_dfs, ice_dfs = [], []

    for target_class, label in {0: "negative (class 0)",
                                1: "positive (class 1)"}.items():

        fig, axes = plt.subplots(nrows=n_rows,
                                 ncols=n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 )
        axes = axes.flatten()

        with parallel_backend(joblib_backend):
            display = PartialDependenceDisplay.from_estimator(model,
                                                              X_train,
                                                              features=feature_list,
                                                              target=target_class,
                                                              kind="both",
                                                              subsample=subsample,
                                                              random_state=random_state,
                                                              n_jobs=n_jobs,
                                                              ax=axes[:len(feature_list)],
                                                              ice_lines_kw={"color": "#85B7EB", "alpha": 0.3, "linewidth": 0.8},
                                                              pd_line_kw={"color": "#185FA5", "linewidth": 2.0},
                                                              )

        for ax in axes[len(feature_list):]:
            ax.set_visible(False)

        fig.suptitle(f"ICE + PDP — {label}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"ice_class_{target_class}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f" ICE plot saved @ {out_path}")

        # extract CSV data from display
        for i, feat in enumerate(feature_list):
            grid = display.pd_results[i]["grid_values"][0]
            pdp = display.pd_results[i]["average"][0]
            ice = display.pd_results[i]["individual"][0]

            pdp_dfs.append(pd.DataFrame({"class": target_class,
                                         "feature": feat,
                                         "x": grid,
                                         "pdp_mean": pdp,
                                         }))

            ice_df = pd.DataFrame(ice, columns=grid)
            ice_df.insert(0, "class", target_class)
            ice_df.insert(1, "feature", feat)
            ice_df.insert(2, "sample_idx", range(len(ice_df)))
            ice_dfs.append(ice_df.melt(id_vars=["class", "feature", "sample_idx"],
                                       var_name="x", value_name="yhat"))

    # save csvs
    pd.concat(pdp_dfs, ignore_index=True).to_csv(os.path.join(output_dir, f"{model_prefix}_ice_pdp_mean.csv"),
                                                 index=False)
    pd.concat(ice_dfs, ignore_index=True).to_csv(os.path.join(output_dir, f"{model_prefix}_ice_lines.csv"),
                                                 index=False)
    print(f" ICE / PDP CSVs @ {output_dir}")


# ---------------------------------------------------------------------------
# SHAP — explainer factory
# ---------------------------------------------------------------------------

def get_shap_explainer(clf,
                       X_train: pd.DataFrame,
                       ) -> tuple:
    """
    Select the appropriate SHAP explainer for the given classifier
    and return (explainer, shap_values) with shap_values normalised
    to class 1 for binary classification.

    Supported classifiers
    ---------------------
    Tree-based  (TreeExplainer — exact, fast):
        DecisionTreeClassifier, RandomForestClassifier, XGBClassifier
    Linear      (LinearExplainer — exact, fast):
        LogisticRegression, LinearSVC
    Any other   (KernelExplainer — approximate, slow):
        KNeighborsClassifier, ComplementNB, etc.
        Background is summarised with k-means (k=50) and
        shap_values are approximated with nsamples=100.

    Parameters
    ----------
    clf : fitted estimator or GridSearchCV / RandomizedSearchCV
        If a search object is passed, the inner classifier is extracted
        from best_estimator_.named_steps["classifier"].
    X_train : pd.DataFrame
        Must already be restricted to selected features.

    Returns
    -------
    tuple : (explainer, sv)
        sv : np.ndarray, shape (n_samples, n_features) — class 1 SHAP values.
    """
    # unwrap pipeline / search object to inspect the underlying classifier
    model = (clf.best_estimator_.named_steps["classifier"]
             if hasattr(clf, "best_estimator_")
             else clf)

    tree_models = (DecisionTreeClassifier,
                   RandomForestClassifier,
                   XGBClassifier)

    kernel_models = (KNeighborsClassifier,
                     ComplementNB)

    if isinstance(model, tree_models):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)

    else:
        X_summary = shap.kmeans(X_train, k=50)
        explainer = shap.KernelExplainer(clf.predict_proba, X_summary)
        shap_values = explainer.shap_values(X_train, nsamples=100)

    # normalize to class 1 across shap versions and explainer types
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    return explainer, sv


# ---------------------------------------------------------------------------
# SHAP — main analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(clf,
                      X_train: pd.DataFrame,
                      observation_indices: list[int] | None = None,
                      selected_features: pd.DataFrame | None = None,
                      output_dir: str = "shap_plots",
                      model_prefix:str = '',
                      ) -> pd.DataFrame:
    """
    Compute SHAP values and save beeswarm summary, bar plot, waterfall
    per observation, and a CSV of all SHAP values.

    Only the features selected by SelectKBest are used — pass
    selected_features to enforce this. If None, all columns of X_train
    are used.

    Parameters
    ----------
    clf : fitted estimator or search object
    X_train : pd.DataFrame
        Full (unfiltered) training feature matrix.
    observation_indices : list[int] or None
        Rows for which to draw waterfall plots. None = first 5.
    selected_features : pd.DataFrame or None
        Output of get_select_k_best_features(). Must contain a "feature"
        column with the selected feature names.
    output_dir : str

    Returns
    -------
    pd.DataFrame
        SHAP values for the selected features (class 1), one row per sample.
    """
    # get selected features that were used by the model
    if selected_features is not None:
        feature_cols = selected_features["feature"].tolist()
        X_shap = X_train[feature_cols]
    else:
        X_shap = X_train

    explainer, sv = get_shap_explainer(clf, X_shap)

    # save csv
    shap_df = pd.DataFrame(sv, columns=X_shap.columns)
    shap_df.insert(0, "sample_idx", range(len(shap_df)))
    shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    print("shap_values.csv saved")

    n_features = X_shap.shape[1]

    # beeswarm
    fig, _ = plt.subplots(figsize=(9, 0.55 * n_features + 1.5))
    shap.summary_plot(sv, X_shap,
                      plot_type="dot",
                      max_display=n_features,
                      show=False,
                      color_bar=True,
                      plot_size=None,
                      )
    plt.gca().set_xlabel("SHAP value (impact on model output)", fontsize=11)
    plt.gca().axvline(0, color="gray", linewidth=0.8, zorder=0)
    plt.tight_layout()
    beeswarm_fig_filename = f"{model_prefix}_shap_beeswarm.png"
    fig.savefig(os.path.join(output_dir, beeswarm_fig_filename),
                dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("shap_beeswarm.png saved")

    # bar plot mean
    fig, _ = plt.subplots(figsize=(7, 0.45 * n_features + 1.5))
    shap.summary_plot(sv, X_shap,
                      plot_type="bar",
                      max_display=n_features,
                      show=False,
                      )
    plt.tight_layout()
    shap_bar_fig_filename = f"{model_prefix}_shap_bar.png"
    fig.savefig(os.path.join(output_dir, shap_bar_fig_filename),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("shap_bar.png saved")

    # waterfall per observation
    # base = (explainer.expected_value[1]
    #         if isinstance(explainer.expected_value, (list, np.ndarray))
    #         else explainer.expected_value)
    #
    # indices = observation_indices or list(range(min(5, len(X_train))))
    #
    # for idx in indices:
    #     explanation = shap.Explanation(values=sv[idx],
    #                                    base_values=base,
    #                                    data=X_shap.iloc[idx].values,
    #                                    feature_names=X_shap.columns.tolist(),
    #                                    )
    #     fig, _ = plt.subplots(figsize=(7, 0.45 * n_features + 1.5))
    #     shap.plots.waterfall(explanation, show=False)
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(output_dir, f"shap_waterfall_obs{idx}.png"),
    #                 dpi=150, bbox_inches="tight")
    #     plt.close(fig)
    #     print(f"shap_waterfall_obs{idx}.png saved")

    return shap_df
