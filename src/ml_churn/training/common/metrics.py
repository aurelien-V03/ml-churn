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


# Poids du rappel dans chaque objectif de la famille F-beta. Un beta de 2 dit
# qu'un churner manque coute quatre fois plus cher qu'une fausse alerte : c'est
# le rapport des couts metier, exprime dans la seule unite que le tuning lise.
BETA: dict[str, float] = {"f1": 1.0, "f2": 2.0}

OBJECTIFS = (*BETA, "recall_sous_contrainte")

# Ce que chaque objectif privilegie, pour le rappeler dans les logs.
DESCRIPTIONS_OBJECTIFS: dict[str, str] = {
    "f1": "precision et rappel a poids egal",
    "f2": "le rappel pese 4 fois la precision",
    "recall_sous_contrainte": (
        f"rappel maximal, sous precision >= {PRECISION_MINIMALE:.0%}"
    ),
}


def objective_score(metrics: dict[str, float], objectif: str) -> float:
    """Valeur a maximiser pour un jeu de metriques donne.

    `f1` equilibre precision et rappel sans supposer de cout metier.
    `f2` pondere le rappel quatre fois plus que la precision : a retenir quand
    un churner manque coute plus cher qu'une fausse alerte.
    `recall_sous_contrainte` maximise le rappel sous un plancher de precision.
    Sans ce plancher, alerter sur tout le monde donnerait un rappel parfait.
    """
    if objectif not in OBJECTIFS:
        raise ValueError(f"objectif inconnu : {objectif!r}, attendu {OBJECTIFS}")

    precision, recall = metrics["precision"], metrics["recall"]

    if objectif == "recall_sous_contrainte":
        return recall if precision >= PRECISION_MINIMALE else 0.0

    beta = BETA[objectif]
    denominateur = beta**2 * precision + recall
    if denominateur == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denominateur


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
