"""Metriques de classification, deduites de la matrice de confusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

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


# Precision plancher de l'objectif `recall_sous_contrainte`.
PRECISION_MINIMALE = 0.60


def confusion_at_threshold(
    y_true: pd.Series, probabilities: np.ndarray, seuil: float
) -> np.ndarray:
    """Matrice de confusion obtenue en alertant au-dela de `seuil`."""
    return confusion_matrix(y_true, (probabilities >= seuil).astype(int))


def objective_score(metrics: dict[str, float], objectif: str) -> float:
    """Valeur a maximiser pour un jeu de metriques donne.

    `f1` equilibre precision et rappel sans supposer de cout metier.
    `recall_sous_contrainte` privilegie la detection des churners, avec un
    plancher de precision : un modele qui alerte sur tout le monde a un rappel
    parfait, la contrainte evite cette solution degeneree.
    """
    precision, recall = metrics["precision"], metrics["recall"]

    if objectif == "recall_sous_contrainte":
        return recall if precision >= PRECISION_MINIMALE else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_at_threshold(
    model, X: pd.DataFrame, y: pd.Series, *, seuil: float
) -> tuple[dict[str, float], np.ndarray]:
    """Metriques et matrice de confusion d'un modele deja entraine.

    L'AUC se calcule sur les probabilites, pas sur la matrice : elle ne depend
    d'aucun seuil de decision.
    """
    probabilities = model.predict_proba(X)[:, 1]
    matrice = confusion_at_threshold(y, probabilities, seuil)

    metrics = classification_metrics(matrice)
    metrics["auc"] = roc_auc_score(y, probabilities)

    return metrics, matrice
