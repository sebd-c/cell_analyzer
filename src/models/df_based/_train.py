# imports
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV,
                                     RepeatedStratifiedKFold,
                                     cross_validate,
                                     )
from sklearn.feature_selection import (mutual_info_classif,
                                       SelectKBest)

#################################################################
DEFAULT_CV_SCORING = {"recall": "recall_weighted",
                      "precision": "precision_weighted",
                      "accuracy": "accuracy",
                      }


#####################################################################
# module with functions of training process
# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

def build_param_grid(model: str) -> dict | None:
    """
    Return the hyperparameter grid for GridSearchCV.
    """

    # define model parameters of interest
    selector_k = {"selector__k": [5, 10, 15, 20, "all"]}

    if model == "decision_tree":
        return {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 3, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__class_weight": ["balanced"],
            "classifier__ccp_alpha": [0, 0.1, 0.5, 1.0, 2.0],
            **selector_k,
        }

    elif model == "random_forest":
        return {
            "classifier__n_estimators": [5, 10, 15, 20],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 3, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__class_weight": ["balanced"],
            "classifier__ccp_alpha": [0, 0.1, 0.5, 1.0, 2.0],
            **selector_k,
        }

    elif model == "knn":
        return {
            "classifier__n_neighbors": [3, 5, 7, 11, 15, 21],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan", "minkowski"],
            "classifier__p": [1, 2],
            **selector_k,
        }


    elif model == "svc":

        return {"classifier__C": [0.1, 1.0, 10.0, 100.0],
                "classifier__kernel": ["rbf", "poly"],
                "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "classifier__class_weight": ["balanced"],
                **selector_k,
                }

    elif model == "xgboost":
        return {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 7, 10],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.3],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__scale_pos_weight": [1, 5, 10, 25],
            "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],
            "classifier__reg_lambda": [1.0, 5.0, 10.0],
            **selector_k,
        }

    else:
        return None


# ---------------------------------------------------------------------------
# Pipeline + search object (inner CV loop)
# ---------------------------------------------------------------------------


def get_select_k_best_features(fitted_model,
                               feature_names,
                               ) -> pd.DataFrame:
    """
    Return SelectKBest feature scores from a fitted GridSearchCV pipeline.
    """
    selector = fitted_model.best_estimator_.named_steps.get("selector")

    features_df = pd.DataFrame({"feature": list(feature_names),
                                "score": selector.scores_,
                                "selected": selector.get_support(),
                                })

    features_df = features_df.sort_values(["selected", "score"],
                                          ascending=[False, False])

    return features_df.reset_index(drop=True)


# First repetition of the cross-validation nest
def build_model(model: str = "dt",
                param_grid: dict | None = None,
                scoring: str = "f1_weighted",
                cv_inner: int = 5,
                random_state: int = 42,
                verbose: int = 3,
                ) -> GridSearchCV:
    """
    Wrapper for the chosen classifier in SelectKBest
    then hand it to GridSearchCV.
    """

    classifiers = {"dt": DecisionTreeClassifier(random_state=random_state),
                   "rf": RandomForestClassifier(random_state=random_state),
                   "knn": KNeighborsClassifier(),
                   "svc": SVC(kernel="rbf",
                              probability=True,
                              class_weight="balanced",
                              max_iter=1000),
                   "xgb": XGBClassifier(random_state=random_state,
                                        eval_metric="logloss",
                                        use_label_encoder=False)
                   }

    scaler = StandardScaler()

    pipe = Pipeline([("scaler", scaler),
                     ("selector", SelectKBest(score_func=mutual_info_classif)),
                     ("classifier", classifiers[model]),
                     ])

    if model in ("xgb", "rf", "dt"):
        search = RandomizedSearchCV(estimator=pipe,
                                    param_distributions=param_grid,
                                    n_iter=50,
                                    scoring=scoring,
                                    cv=cv_inner,
                                    refit=True,
                                    verbose=verbose,
                                    random_state=random_state,
                                    n_jobs=-1
                                    )
    else:  # knn, nb
        search = GridSearchCV(estimator=pipe,
                              param_grid=param_grid,
                              scoring=scoring,
                              cv=cv_inner,
                              refit=True,
                              verbose=verbose,
                              n_jobs=-1
                              )

    return search


# ---------------------------------------------------------------------------
# Outer CV loop
# ---------------------------------------------------------------------------


# Second repetition of the cross-validation nest
def run_cross_validation(model,
                         X,
                         y,
                         n_splits: int = 5,
                         n_repeats: int = 3,
                         random_state: int = 42,
                         scoring: dict | None = None,
                         ) -> pd.DataFrame:
    """
    Evaluate model with RepeatedStratifiedKFold and return a tidy DataFrame
    of per-fold scores.
    """

    # define cross validation settings
    cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats,
                                 random_state=random_state,
                                 )

    # run cross-validation with specified model
    cv_results = cross_validate(model,
                                X,
                                y,
                                cv=cv,
                                scoring=scoring or DEFAULT_CV_SCORING,
                                return_train_score=True,
                                n_jobs=-1
                                )

    scores_df = pd.DataFrame(cv_results)

    return scores_df


# ---------------------------------------------------------------------------
# Saving and loading of Models
# ---------------------------------------------------------------------------


# post model training functions, i.e. run on test dataset, save model...
def save_model(model,
               X_train,
               y_train,
               model_path: str,
               ):
    """
    Fit model on the full training set and saves it as joblib.
    """
    print("[…] Fitting model on full training data…")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Fitted model saved into {model_path}")
    return model


def load_model(model_path: str):
    """
    Load and return a previously saved model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
