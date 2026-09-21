"""Metriques de classification, deduites de la matrice de confusion."""

from __future__ import annotations

import numpy as np

FORMULES: dict[str, str] = {
    "accuracy": "(TP+TN) / (TP+TN+FP+FN)",
    "recall": "TP / (TP+FN)",
    "false_positive_rate": "FP / (FP+TN)",
    "precision": "TP / (TP+FP)",
    "auc": "aire sous la courbe ROC (independante du seuil)",
}


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    """TP : churners detectes. FN : churners manques. FP : fausses alertes."""
    (true_negatives, false_positives), (false_negatives, true_positives) = matrix

    return {
        "accuracy": (true_positives + true_negatives) / matrix.sum(),
        "recall": true_positives / (true_positives + false_negatives),
        "false_positive_rate": false_positives / (false_positives + true_negatives),
        "precision": true_positives / (true_positives + false_positives),
    }
