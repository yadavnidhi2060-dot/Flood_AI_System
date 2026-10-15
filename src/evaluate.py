"""Model-evaluation helpers: metrics, confusion-matrix plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from src import config

LABEL_IDS = list(range(len(config.RISK_LABELS)))


def compute_metrics(y_true, y_pred) -> dict:
    """Overall + per-class metrics for the 3-class problem (macro = every class counts equally)."""
    kw = dict(labels=LABEL_IDS, zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABEL_IDS,
                                   target_names=config.RISK_LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in config.RISK_LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
    }


def plot_confusion_matrix(cm, title="Confusion matrix", normalize=False, ax=None, cmap="Blues"):
    """Heat-map of a confusion matrix. Rows = actual class, columns = predicted class."""
    cm = np.asarray(cm, dtype=float)
    data = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(data, annot=True, fmt=".0%" if normalize else ".0f", cmap=cmap, cbar=False,
                xticklabels=config.RISK_LABELS, yticklabels=config.RISK_LABELS,
                linewidths=1, linecolor="white", ax=ax, annot_kws={"size": 12})
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Actual risk")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return ax
