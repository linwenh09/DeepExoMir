"""Evaluation metrics for miRNA-target interaction prediction.

Computes standard binary classification metrics using scikit-learn,
with additional biomedically relevant scores (MCC, sensitivity,
specificity) important for imbalanced MTI datasets.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """Static utility class for computing and formatting classification metrics."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, float]:
        """Compute a comprehensive set of binary classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels, shape ``(N,)``.
        y_pred : np.ndarray
            Predicted binary labels (after thresholding), shape ``(N,)``.
        y_prob : np.ndarray
            Predicted probabilities for the positive class, shape ``(N,)``.

        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - ``auc_roc``: Area under the ROC curve.
            - ``auc_pr``: Area under the precision-recall curve.
            - ``accuracy``: Overall accuracy.
            - ``f1``: F1 score (positive class).
            - ``mcc``: Matthews correlation coefficient.
            - ``sensitivity``: True positive rate (recall).
            - ``specificity``: True negative rate.
            - ``precision``: Positive predictive value.
            - ``recall``: Same as sensitivity, included for convenience.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        # Handle degenerate cases where only one class is present
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            auc_roc = 0.0
            auc_pr = 0.0
        else:
            auc_roc = float(roc_auc_score(y_true, y_prob))
            auc_pr = float(average_precision_score(y_true, y_prob))

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        # Sensitivity and specificity from confusion matrix
        tn, fp, fn, tp = Evaluator._safe_confusion_values(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": prec,
            "recall": rec,
        }

    @staticmethod
    def _safe_confusion_values(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Extract TN, FP, FN, TP from a confusion matrix, handling edge cases.

        Returns
        -------
        tuple[int, int, int, int]
            (tn, fp, fn, tp)
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        return tn, fp, fn, tp

    @staticmethod
    def format_metrics(metrics: dict[str, float]) -> str:
        """Format a metrics dictionary as a human-readable table.

        Parameters
        ----------
        metrics : dict[str, float]
            Metrics dictionary as returned by :meth:`compute_metrics`.

        Returns
        -------
        str
            Multi-line formatted table string.
        """
        header = "Metric              | Value"
        separator = "-" * 20 + "-+-" + "-" * 10
        lines = [header, separator]

        display_order = [
            ("AUC-ROC", "auc_roc"),
            ("AUC-PR", "auc_pr"),
            ("Accuracy", "accuracy"),
            ("F1 Score", "f1"),
            ("MCC", "mcc"),
            ("Sensitivity", "sensitivity"),
            ("Specificity", "specificity"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ]

        for display_name, key in display_order:
            value = metrics.get(key, float("nan"))
            lines.append(f"{display_name:<20s} | {value:>8.4f}")

        return "\n".join(lines)
