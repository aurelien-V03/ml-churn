"""Visuels attaches aux runs d'entrainement."""

from __future__ import annotations

import matplotlib

# Backend sans fenetre : les figures sont enregistrees, jamais affichees.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LIBELLES = ("reste", "churn")
SIGLES = (("TN", "FP"), ("FN", "TP"))


def confusion_matrix_figure(matrix: np.ndarray, *, titre: str) -> plt.Figure:
    """Matrice de confusion en damier, chaque case annotee de son sigle.

    Les cases sont colorees par leur part de la ligne (et non du total) : sans
    cela, sur un jeu desequilibre, seuls les vrais negatifs ressortiraient.
    """
    figure, axes = plt.subplots(figsize=(6, 5))
    parts = matrix / matrix.sum(axis=1, keepdims=True)
    axes.imshow(parts, cmap="Blues", vmin=0, vmax=1)

    for ligne in range(2):
        for colonne in range(2):
            valeur = int(matrix[ligne][colonne])
            couleur = "white" if parts[ligne][colonne] > 0.5 else "#1a1a1a"
            axes.text(
                colonne,
                ligne,
                f"{SIGLES[ligne][colonne]}\n{valeur}\n{parts[ligne][colonne]:.1%}",
                ha="center",
                va="center",
                color=couleur,
                fontsize=13,
            )

    axes.set_xticks([0, 1], [f"predit {nom}" for nom in LIBELLES])
    axes.set_yticks([0, 1], [f"reel {nom}" for nom in LIBELLES])
    axes.set_title(titre)
    figure.tight_layout()

    return figure
