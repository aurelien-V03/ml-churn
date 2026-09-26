"""Visuels attaches aux runs d'entrainement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Les figures sont construites sans passer par pyplot : une bibliotheque ne
# doit toucher ni au backend global -- ce que faisait `matplotlib.use("Agg")`,
# au prix de l'affichage dans le notebook -- ni au registre des figures
# ouvertes. Elles sont enregistrees par l'appelant, jamais affichees ici.

LIBELLES = ("reste", "churn")

# Au-dela, la matrice est trop dense pour porter ses valeurs, et la figure
# depasse ce qu'un ecran affiche.
MAX_ANNOTATIONS = 22
TAILLE_MAX = 16
SIGLES = (("TN", "FP"), ("FN", "TP"))


def confusion_matrix_figure(matrix: np.ndarray, *, titre: str) -> Figure:
    """Matrice de confusion en damier, chaque case annotee de son sigle.

    Les cases sont colorees par leur part de la ligne (et non du total) : sans
    cela, sur un jeu desequilibre, seuls les vrais negatifs ressortiraient.
    """
    figure = Figure(figsize=(6, 5))
    axes = figure.subplots()
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


def correlation_matrix_figure(matrice, *, titre: str) -> Figure:
    """Matrice de correlation en damier, du bleu (-1) au rouge (+1).

    L'echelle est fixee a [-1, 1] et centree sur zero : sans cela, une matrice
    sans correlation forte s'afficherait en couleurs vives et donnerait
    l'illusion de liens qui n'existent pas.
    """
    # Au-dela d'une vingtaine de colonnes, les valeurs ne tiennent plus dans
    # les cases : la figure est plafonnee et seules les couleurs subsistent.
    annoter = len(matrice) <= MAX_ANNOTATIONS
    taille = min(0.55 * len(matrice) + 3, TAILLE_MAX)
    police = max(3, round(60 / len(matrice)))

    figure = Figure(figsize=(taille, taille))
    axes = figure.subplots()

    # Les cases vides d'une matrice filtree ne doivent pas etre coloriees.
    palette = plt.get_cmap("coolwarm").with_extremes(bad="#00000000")
    image = axes.imshow(matrice, cmap=palette, vmin=-1, vmax=1)
    figure.colorbar(image, ax=axes, shrink=0.8, label="correlation de Pearson")

    for ligne in range(len(matrice) if annoter else 0):
        for colonne in range(len(matrice)):
            valeur = matrice.iat[ligne, colonne]
            if pd.isna(valeur):
                continue
            axes.text(
                colonne,
                ligne,
                f"{valeur:.2f}".replace("0.", "."),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(valeur) > 0.55 else "#1a1a1a",
            )

    axes.set_xticks(range(len(matrice)), matrice.columns, rotation=90, fontsize=police)
    axes.set_yticks(range(len(matrice)), matrice.index, fontsize=police)
    axes.set_title(titre)
    figure.tight_layout()

    return figure
