"""Explication des modeles par valeurs de Shapley (SHAP).

Les coefficients d'une regression logistique disent le poids d'une feature,
pas son effet reel : une feature au coefficient eleve mais toujours proche de
zero ne deplace aucune prediction. SHAP mesure la contribution de chaque
feature a chaque prediction, coefficient et valeur observee combines.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# Au-dela, le graphique devient illisible ; SHAP garde les plus contributives.
# `max_features=None` les affiche toutes.
MAX_FEATURES = 20


def shap_values(model: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Contributions SHAP exactes du modele sur `X`.

    `LinearExplainer` calcule les valeurs exactes pour un modele lineaire, sans
    l'echantillonnage des explainers generiques. Sans `max_samples`, SHAP
    n'echantillonnerait que 100 lignes pour estimer la moyenne de reference ;
    ici le jeu tient en memoire, autant la calculer sur tout.
    """
    estimateur = model[-1] if isinstance(model, Pipeline) else model
    donnees = (
        model[:-1].transform(X) if isinstance(model, Pipeline) and len(model) > 1 else X
    )

    reference = shap.maskers.Independent(donnees, max_samples=len(donnees))
    return shap.LinearExplainer(estimateur, reference)(donnees)


def _nombre_de_lignes(max_features: int | None, X: pd.DataFrame) -> int:
    """Nombre de lignes a tracer, `None` valant « toutes les colonnes ».

    SHAP reserve la derniere ligne au cumul des features non affichees : il en
    faut donc une de plus que de colonnes pour qu'aucune ne soit regroupee.
    """
    if max_features is None:
        return X.shape[1] + 1
    return min(max_features, X.shape[1])


def _figure_vierge() -> None:
    """Ouvre une figure neuve avant de laisser SHAP tracer.

    SHAP dessine dans la figure courante (`plt.gca()`). Sans cette precaution,
    le graphique se superpose a la derniere figure restee ouverte -- typiquement
    une matrice de confusion produite par le tuning.
    """
    plt.figure()


def _mettre_en_forme(
    figure: plt.Figure, *, titre: str, lignes: int, largeur: float
) -> plt.Figure:
    """Hauteur proportionnelle au nombre de features affichees."""
    figure.set_size_inches(largeur, 0.32 * lignes + 2)
    figure.suptitle(titre, y=1.0)
    figure.tight_layout()
    return figure


def shap_summary_figure(
    model: Pipeline,
    X: pd.DataFrame,
    *,
    titre: str,
    max_features: int | None = MAX_FEATURES,
) -> plt.Figure:
    """Summary plot (beeswarm) des contributions SHAP, features les plus fortes en haut.

    Un point par ligne et par feature : sa position donne l'effet sur la
    prediction (droite = pousse vers le churn), sa couleur la valeur de la
    feature (rouge = elevee). Une bande rouge a droite se lit donc « plus cette
    feature est elevee, plus le client churne ».
    """
    lignes = _nombre_de_lignes(max_features, X)
    _figure_vierge()
    shap.plots.beeswarm(shap_values(model, X), max_display=lignes, show=False)

    return _mettre_en_forme(plt.gcf(), titre=titre, lignes=lignes, largeur=9)


def shap_bar_figure(
    model: Pipeline,
    X: pd.DataFrame,
    *,
    titre: str,
    max_features: int | None = MAX_FEATURES,
) -> plt.Figure:
    """Importance globale : moyenne des |SHAP| par feature, en barres.

    Meme calcul que le beeswarm, agrege : une barre par feature au lieu d'un
    point par client. On y perd le sens de l'effet (la valeur absolue efface le
    signe) et la dispersion, on y gagne un classement lisible d'un coup d'oeil.
    """
    lignes = _nombre_de_lignes(max_features, X)
    _figure_vierge()
    shap.plots.bar(shap_values(model, X), max_display=lignes, show=False)

    return _mettre_en_forme(plt.gcf(), titre=titre, lignes=lignes, largeur=8)


def display_figure(figure: plt.Figure) -> None:
    """Affiche la figure dans un notebook ; sans effet en ligne de commande.

    Le PNG est rendu ici puis passe a IPython plutot que de laisser le notebook
    afficher l'objet figure : `common.plots` bascule matplotlib sur le backend
    Agg des son import, ce qui neutralise l'affichage automatique.
    """
    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return

    if get_ipython() is None:
        return

    tampon = io.BytesIO()
    figure.savefig(tampon, format="png", dpi=110, bbox_inches="tight")
    display(Image(data=tampon.getvalue()))
    plt.close(figure)
