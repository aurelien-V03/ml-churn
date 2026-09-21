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


def _ratio(numerateur: float, denominateur: float) -> float:
    """0.0 quand le denominateur est nul : au seuil 1.0, aucune alerte n'est
    emise et la precision n'est pas definie."""
    return float(numerateur / denominateur) if denominateur else 0.0


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    """TP : churners detectes. FN : churners manques. FP : fausses alertes."""
    (true_negatives, false_positives), (false_negatives, true_positives) = matrix

    return {
        "accuracy": _ratio(true_positives + true_negatives, matrix.sum()),
        "recall": _ratio(true_positives, true_positives + false_negatives),
        "false_positive_rate": _ratio(
            false_positives, false_positives + true_negatives
        ),
        "precision": _ratio(true_positives, true_positives + false_positives),
    }
