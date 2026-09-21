"""
models.py
---------
Trains and evaluates:
  - Classification models (predict UP/DOWN): Random Forest, SVM
  - Regression model (predict % return magnitude): Linear Regression

IMPORTANT: Stock data is a TIME SERIES. We must NOT shuffle train/test splits
randomly (that leaks future information into training). We always split
chronologically: earliest X% = train, latest (1-X)% = test.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)


def chronological_split(df: pd.DataFrame, feature_cols: list, target_col: str, train_size: float = 0.8):
    """
    Splits time-series data by TIME, not randomly.
    Returns X_train, X_test, y_train, y_test (all as numpy arrays / Series).
    """
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_random_forest_classifier(X_train, y_train, n_estimators=200, max_depth=8):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"  # avoids always predicting the majority class (Down or Up)
    )
    model.fit(X_train, y_train)
    return model


def train_svm_classifier(X_train, y_train, X_test=None):
    """
    SVM is distance-based, so features MUST be scaled first.
    Returns (model, scaler) — scaler must be reused on any new data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42,
                class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_rf_regressor(X_train, y_train, n_estimators=200, max_depth=8):
    """Optional stronger regression baseline alongside Linear Regression."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test, scaler=None) -> dict:
    """
    Computes accuracy, precision, recall, F1, confusion matrix.
    If a scaler is passed (SVM case), X_test is scaled with it first.
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]  # P(class == 1 / "Up")

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "predictions": preds,
        "probabilities": proba,
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    """Computes RMSE, MAE, R² for a regression model."""
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2_score": float(r2_score(y_test, preds)),
        "predictions": preds,
    }


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Extracts feature importances from a tree-based model (Random Forest)."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def train_all_models(df: pd.DataFrame, feature_cols: list, train_size: float = 0.8) -> dict:
    """
    Trains all three required models end-to-end and returns everything
    the Streamlit app needs to display results:
      - Random Forest classifier
      - SVM classifier (+ its scaler)
      - Linear Regression

    Returns a dict with models, scalers, splits, and evaluation metrics.
    """
    # --- Classification task (Target_Class: up/down) ---
    X_train, X_test, y_train_cls, y_test_cls, train_df, test_df = chronological_split(
        df, feature_cols, "Target_Class", train_size
    )

    rf_clf = train_random_forest_classifier(X_train, y_train_cls)
    rf_metrics = evaluate_classifier(rf_clf, X_test, y_test_cls)

    svm_clf, svm_scaler = train_svm_classifier(X_train, y_train_cls)
    svm_metrics = evaluate_classifier(svm_clf, X_test, y_test_cls, scaler=svm_scaler)

    # --- Regression task (Target_Return: % move) ---
    _, _, y_train_reg, y_test_reg, _, _ = chronological_split(
        df, feature_cols, "Target_Return", train_size
    )
    lin_reg = train_linear_regression(X_train, y_train_reg)
    lin_metrics = evaluate_regressor(lin_reg, X_test, y_test_reg)

    return {
        "X_train": X_train, "X_test": X_test,
        "train_df": train_df, "test_df": test_df,
        "random_forest": {"model": rf_clf, "metrics": rf_metrics,
                           "importance": get_feature_importance(rf_clf, feature_cols)},
        "svm": {"model": svm_clf, "scaler": svm_scaler, "metrics": svm_metrics},
        "linear_regression": {"model": lin_reg, "metrics": lin_metrics},
    }


def cross_validate_models(df: pd.DataFrame, feature_cols: list,
                           target_col: str = "Target_Class", n_splits: int = 5) -> dict:
    """
    Walk-forward (TimeSeriesSplit) cross-validation for RF and SVM classifiers.

    Why this matters: a SINGLE train/test split (used elsewhere in this file)
    gives one accuracy number that can look good or bad by chance, depending
    on which time window happened to be the test set. TimeSeriesSplit instead
    trains/tests across several different chronological windows and reports
    the average — a much more reliable estimate of how the model actually
    performs, and its standard deviation shows how much that varies.

    Unlike a plain random K-Fold, TimeSeriesSplit always trains on the past
    and tests on the future for every fold — never shuffled — so it doesn't
    leak future information into training.
    """
    X = df[feature_cols]
    y = df[target_col]

    # Guard against too few rows for the requested number of folds
    n_splits = min(n_splits, max(2, len(df) // 40))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rf_scores, svm_scores = [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))

        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42))
        ])
        svm_pipeline.fit(X_train, y_train)
        svm_scores.append(svm_pipeline.score(X_test, y_test))

    return {
        "n_splits": n_splits,
        "rf_fold_scores": [round(s, 4) for s in rf_scores],
        "rf_mean_accuracy": float(np.mean(rf_scores)),
        "rf_std": float(np.std(rf_scores)),
        "svm_fold_scores": [round(s, 4) for s in svm_scores],
        "svm_mean_accuracy": float(np.mean(svm_scores)),
        "svm_std": float(np.std(svm_scores)),
    }