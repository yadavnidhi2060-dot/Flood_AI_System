"""
Train and evaluate the flood-risk classifiers.

Models
------
* Dummy baseline  : always predicts the most frequent class (shows why accuracy alone misleads)
* Logistic Regression (standardised features, balanced class weights)
* Random Forest   (balanced class weights, hyper-parameters tuned with TimeSeriesSplit CV)

Validation strategy
-------------------
Weather data is a time series, so a random split would leak the future into training.
We hold out the LAST 20 % of the timeline as the test set and use forward-chaining
(TimeSeriesSplit) cross-validation inside the training period. The best model is chosen
by cross-validated macro-F1 on the TRAINING period only; the test set is used only
for the final report.

Usage:  python -m src.train
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, preprocessing
from src.evaluate import compute_metrics


def time_split(table: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Chronological split (all stations share the same cut-off date)."""
    dates = np.sort(table["date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    train = table[table["date"] < cutoff].sort_values("date")
    test = table[table["date"] >= cutoff].sort_values("date")
    return train, test, pd.Timestamp(cutoff)


def build_models() -> dict:
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                       C=1.0, random_state=config.RANDOM_STATE)),
        ]),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=config.RANDOM_STATE),
            param_grid={"n_estimators": [200, 400], "max_depth": [8, 14],
                        "min_samples_leaf": [2, 5]},
            cv=TimeSeriesSplit(n_splits=3), scoring="f1_macro", n_jobs=1, refit=True),
    }


def run() -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    table = preprocessing.run() if not config.FEATURE_FILE.exists() else \
        pd.read_csv(config.FEATURE_FILE, parse_dates=["date"])
    table = table.sort_values("date").reset_index(drop=True)
    data_source = str(table["data_source"].iloc[0])

    train, test, cutoff = time_split(table)
    X_tr, y_tr = train[config.FEATURES], train["target_int"]
    X_te, y_te = test[config.FEATURES], test["target_int"]
    print(f"\nTrain: {len(train):,} rows ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"Test : {len(test):,} rows ({test['date'].min().date()} -> {test['date'].max().date()})")

    results, fitted = {}, {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(X_tr, y_tr)
        best_params = getattr(model, "best_params_", None)
        est = getattr(model, "best_estimator_", model)
        # cross-validated macro-F1 on the training period (used for model selection)
        if name.startswith("Baseline"):
            cv_f1 = float(cross_val_score(est, X_tr, y_tr, cv=TimeSeriesSplit(3), scoring="f1_macro").mean())
        elif best_params is not None:
            cv_f1 = float(model.best_score_)
        else:
            cv_f1 = float(cross_val_score(est, X_tr, y_tr, cv=TimeSeriesSplit(3), scoring="f1_macro").mean())
        m = compute_metrics(y_te, est.predict(X_te))
        m["cv_f1_macro_train"] = cv_f1
        m["best_params"] = best_params
        results[name], fitted[name] = m, est
        print(f"  CV macro-F1 (train): {cv_f1:.3f} | Test acc: {m['accuracy']:.3f} | "
              f"macro P/R/F1: {m['precision_macro']:.3f}/{m['recall_macro']:.3f}/{m['f1_macro']:.3f}")

    candidates = [n for n in results if not n.startswith("Baseline")]
    best_name = max(candidates, key=lambda n: results[n]["cv_f1_macro_train"])
    best = fitted[best_name]
    print(f"\nBest model (by train-CV macro-F1): {best_name}")

    # ---- feature importance ----
    rf = fitted["Random Forest"]
    lr = fitted["Logistic Regression"].named_steps["clf"]
    perm = permutation_importance(best, X_te, y_te, scoring="f1_macro", n_repeats=10,
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": config.FEATURES,
        "rf_importance": rf.feature_importances_,
        "lr_abs_coef": np.abs(lr.coef_).mean(axis=0),
        "permutation_importance": perm.importances_mean,
        "permutation_std": perm.importances_std,
    }).sort_values("rf_importance", ascending=False)
    imp.to_csv(config.IMPORTANCE_FILE, index=False)

    # ---- test predictions of the best model ----
    proba = best.predict_proba(X_te)
    pred_df = test[["date", "station"]].copy()
    pred_df["actual"] = test["target"].values
    pred_df["predicted"] = [config.RISK_LABELS[i] for i in proba.argmax(axis=1)]
    for i, lab in enumerate(config.RISK_LABELS):
        pred_df[f"p_{lab}"] = proba[:, i].round(4)
    pred_df.to_csv(config.TEST_PRED_FILE, index=False)

    joblib.dump({"model": best, "model_name": best_name, "features": config.FEATURES,
                 "labels": config.RISK_LABELS, "data_source": data_source,
                 "trained_at": datetime.now().isoformat(timespec="seconds")}, config.MODEL_FILE)

    summary = {
        "data_source": data_source,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "split": {"train_rows": len(train), "test_rows": len(test),
                  "train_start": str(train["date"].min().date()), "train_end": str(train["date"].max().date()),
                  "test_start": str(test["date"].min().date()), "test_end": str(test["date"].max().date())},
        "class_counts_train": {k: int(v) for k, v in train["target"].value_counts().items()},
        "class_counts_test": {k: int(v) for k, v in test["target"].value_counts().items()},
        "features": config.FEATURES,
        "models": results,
    }
    with open(config.METRICS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {config.MODEL_FILE.name}, {config.METRICS_FILE.name}, "
          f"{config.IMPORTANCE_FILE.name}, {config.TEST_PRED_FILE.name} -> {config.MODELS_DIR}")
    return summary


if __name__ == "__main__":
    run()
