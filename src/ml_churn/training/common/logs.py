"""Format de log commun aux entrainements.

Tous les modeles de classification affichent le meme bilan : jeu de donnees,
metriques, matrice de confusion, puis poids des features. Seul le dernier bloc
change de nature d'un modele a l'autre -- coefficients pour un modele lineaire,
importances pour un modele a base d'arbres -- d'ou son intitule parametrable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_churn.training.common.metrics import (
    DESCRIPTIONS_OBJECTIFS,
    FORMULES,
    FORMULES_REGRESSION,
)

# Au-dela, la liste des features encombre le log plus qu'elle ne l'informe.
MAX_FEATURES_LISTEES = 15


def log_dataset(
    df: pd.DataFrame,
    features: list[str],
    exclusions: dict[str, str],
    tailles: dict[str, int],
    y_test: pd.Series,
) -> None:
    """Volumetrie, features retenues et decoupage des trois jeux."""
    print(f"[DONNEES] : {len(df)} clients, {len(features)} features")
    print(f"  exclues : {', '.join(sorted(exclusions))}")
    detail = " / ".join(
        f"{nom.removeprefix('n_')} {taille}" for nom, taille in tailles.items()
    )
    print(f"  {detail}  (churn {y_test.mean():.2%} dans le test)")


def log_metrics(
    metrics: dict[str, float], formules: dict[str, str] | None = None
) -> None:
    """Metriques du jeu de test, avec le rappel de leur formule."""
    formules = formules or FORMULES

    print("\n[PERFORMANCE] sur le jeu de test")
    for nom, value in metrics.items():
        print(f"  {nom:<21} {value:>12.2f}   {formules[nom]}")


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
    weights: pd.Series,
    *,
    titre: str = "COEFFICIENTS",
    nombre: int = 10,
    direction: bool = True,
) -> None:
    """Features les plus influentes, de la plus forte a la plus faible.

    `direction=False` pour les importances d'un modele a arbres : toujours
    positives, elles disent l'intensite mais pas le sens de l'effet.
    """
    print(f"\n[{titre}] les {nombre} plus influents")
    for nom, value in weights.head(nombre).items():
        if not direction:
            print(f"  {nom:<30} {value:.2f}")
            continue
        sens = "augmente" if value > 0 else "diminue "
        print(f"  {nom:<30} {value:+.2f}  ({sens} le risque)")


def log_objective(objectif: str) -> None:
    """Objectif maximise par la recherche, et ce qu'il privilegie."""
    print(f"\n[OBJECTIF] {objectif} : {DESCRIPTIONS_OBJECTIFS[objectif]}")


def log_regression_training(
    *,
    df: pd.DataFrame,
    features: list[str],
    tailles: dict[str, int],
    y_test: pd.Series,
    metrics: dict[str, float],
) -> None:
    """Bilan complet d'un entrainement de regression.

    Pas de matrice de confusion ici : la sortie est continue, l'erreur se lit
    dans l'unite de la cible plutot qu'en comptant des cas bien classes. Les
    coefficients ne sont pas affichés non plus : ils restent accessibles dans
    `Result.coefficients`.
    """
    print(f"[DONNEES] : {len(df)} clients, {len(features)} features")
    # Quelques features se listent, des dizaines n'apportent rien a lire.
    if len(features) <= MAX_FEATURES_LISTEES:
        print(f"  retenues : {', '.join(features)}")
    detail = " / ".join(
        f"{nom.removeprefix('n_')} {taille}" for nom, taille in tailles.items()
    )
    print(f"  {detail}")
    print(
        f"  cible sur le test : mediane {y_test.median():,.0f} €, "
        f"moyenne {y_test.mean():,.0f} €, max {y_test.max():,.0f} €"
    )

    log_metrics(metrics, FORMULES_REGRESSION)


def log_carbon_footprint(
    empreinte: dict[str, float], *, titre: str = "de l'entrainement"
) -> None:
    """Energie consommee et CO2 equivalent du bloc mesure."""
    if not empreinte:
        return

    print(f"\n[EMPREINTE CARBONE] {titre}")
    print(f"  ⏱️  duree             {empreinte['duration_s']:.2f} s")
    print(f"  ⚡ energie            {empreinte['energy_mwh']:.2f} mWh")
    print(f"  🌍 emissions          {empreinte['co2_mg']:.2f} mg CO2eq")
    print(f"  💧 eau                {empreinte['water_ml']:.2f} mL")


def log_classification_training(
    *,
    df: pd.DataFrame,
    features: list[str],
    exclusions: dict[str, str],
    tailles: dict[str, int],
    y_test: pd.Series,
    metrics: dict[str, float],
    matrix: np.ndarray,
    weights: pd.Series,
    weights_titre: str = "COEFFICIENTS",
    weights_direction: bool = True,
) -> None:
    """Bilan complet d'un entrainement de classification."""
    log_dataset(df, features, exclusions, tailles, y_test)
    log_metrics(metrics)
    log_confusion_matrix(matrix)
    log_feature_weights(weights, titre=weights_titre, direction=weights_direction)
