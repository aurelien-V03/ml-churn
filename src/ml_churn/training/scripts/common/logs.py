"""Format de log commun aux entrainements.

Tous les modeles de classification affichent le meme bilan : jeu de donnees,
metriques, matrice de confusion, puis poids des features. Seul le dernier bloc
change de nature d'un modele a l'autre -- coefficients pour un modele lineaire,
importances pour un modele a base d'arbres -- d'ou son intitule parametrable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_churn.training.scripts.common.metrics import FORMULES


def log_dataset(
    df: pd.DataFrame,
    features: list[str],
    exclusions: dict[str, str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Volumetrie, features retenues et decoupage train/test."""
    print(f"[DONNEES] : {len(df)} clients, {len(features)} features")
    print(f"  exclues : {', '.join(sorted(exclusions))}")
    print(
        f"  train {len(X_train)} / test {len(X_test)} "
        f"(churn {y_test.mean():.2%} dans le test)"
    )


def log_metrics(metrics: dict[str, float]) -> None:
    """Metriques du jeu de test, avec le rappel de leur formule."""
    print("\n[PERFORMANCE] sur le jeu de test")
    for nom, value in metrics.items():
        print(f"  {nom:<21} {value:.4f}   {FORMULES[nom]}")


def log_confusion_matrix(matrix: np.ndarray) -> None:
    """Matrice de confusion, puis lecture metier de chaque case."""
    (true_negatives, false_positives), (false_negatives, true_positives) = matrix

    print("\n[MATRICE DE CONFUSION]")
    print(f"  {'':<14} {'predit reste':>18} {'predit churn':>18}")
    print(
        f"  {'reel reste':<14} {f'TN = {true_negatives}':>18} "
        f"{f'FP = {false_positives}':>18}"
    )
    print(
        f"  {'reel churn':<14} {f'FN = {false_negatives}':>18} "
        f"{f'TP = {true_positives}':>18}"
    )
    print(f"\n  TP = {true_positives:<5} churners detectes")
    print(f"  FN = {false_negatives:<5} churners manques")
    print(f"  FP = {false_positives:<5} fausses alertes")
    print(f"  TN = {true_negatives:<5} clients fideles correctement identifies")


def log_feature_weights(
    weights: pd.Series, *, titre: str = "COEFFICIENTS", nombre: int = 10
) -> None:
    """Features les plus influentes, de la plus forte a la plus faible."""
    print(f"\n[{titre}] les {nombre} plus influents")
    for nom, value in weights.head(nombre).items():
        direction = "augmente" if value > 0 else "diminue "
        print(f"  {nom:<30} {value:+.4f}  ({direction} le risque)")


def log_classification_training(
    *,
    df: pd.DataFrame,
    features: list[str],
    exclusions: dict[str, str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict[str, float],
    matrix: np.ndarray,
    weights: pd.Series,
    weights_titre: str = "COEFFICIENTS",
) -> None:
    """Bilan complet d'un entrainement de classification."""
    log_dataset(df, features, exclusions, X_train, X_test, y_test)
    log_metrics(metrics)
    log_confusion_matrix(matrix)
    log_feature_weights(weights, titre=weights_titre)
