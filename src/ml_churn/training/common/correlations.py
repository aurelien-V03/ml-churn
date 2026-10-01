"""Correlations entre variables, pour guider le choix des features.

La lecture se fait sur la couche gold, celle que les modeles consomment : les
correlations portent donc exactement sur les colonnes qui leur sont fournies,
modalites one-hot comprises. Une modalite binaire correlee a une variable
continue se lit avec prudence -- le coefficient y mesure surtout un ecart de
moyenne entre les deux groupes -- mais elle a l'avantage d'exister dans le
modele, ce qu'une colonne categorielle de silver ne fait pas.

Deux usages : reperer les features liees a la cible -- candidates a l'entree du
modele -- et reperer celles liees entre elles, dont la redondance destabilise
les coefficients d'un modele lineaire.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from ml_churn.ingestion.db import PROJECT_ROOT
from ml_churn.training.common.data import load_gold, load_silver
from ml_churn.training.common.explain import display_figure
from ml_churn.training.common.plots import correlation_matrix_figure

# Hors de `visualization/graphs`, que `plot_all` vide a chaque execution : ces
# figures viennent du notebook d'entrainement, pas des scripts d'exploration.
CORRELATIONS_DIR = PROJECT_ROOT / "src" / "visualization" / "correlation"

# Colonnes numeriques sans interet pour l'analyse : identifiants techniques.
EXCLUSIONS = ("id",)

# Au-dela, deux features disent a peu près la meme chose.
SEUIL_REDONDANCE = 0.8

# Bornes du filtre : on ne garde que les liens positifs francs et les liens
# negatifs deja notables. La bande est asymetrique parce que les correlations
# negatives sont plus rares ici, et donc informatives plus tot.
SEUIL_POSITIF = 0.5
SEUIL_NEGATIF = -0.2

# Largeur du bandeau qui separe deux recherches dans le log.
LARGEUR_BANDEAU = 70


def log_titre(intitule: str) -> None:
    """Bandeau de separation : deux recherches ne doivent pas se confondre."""
    barre = "=" * LARGEUR_BANDEAU
    print(f"\n{barre}\n{intitule}\n{barre}")


def log_seuils() -> None:
    """Les seuils appliques, pour que le log se lise sans ouvrir le code."""
    print(
        f"[SEUILS] redondance |r| >= {SEUIL_REDONDANCE:.2f}  |  "
        f"filtre de la matrice : r > {SEUIL_POSITIF:+.2f} ou r < {SEUIL_NEGATIF:+.2f}"
    )


def save_correlation_figure(figure: Figure, nom: str) -> Path:
    """Ecrit la figure dans `visualization/correlation` et logue le chemin."""
    CORRELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    chemin = CORRELATIONS_DIR / f"{nom}.png"
    figure.savefig(chemin, dpi=120, bbox_inches="tight")

    print(f"[FIGURE] {chemin.relative_to(PROJECT_ROOT)}")

    return chemin


def _correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Matrice de correlation de Pearson, cible en derniere position."""
    numeriques = df.select_dtypes("number").drop(
        columns=list(EXCLUSIONS), errors="ignore"
    )

    colonnes = [nom for nom in numeriques.columns if nom != target] + [target]
    return numeriques[colonnes].corr()


def silver_correlations(target: str) -> pd.DataFrame:
    """Correlations sur la couche silver : une colonne par variable metier."""
    return _correlations(load_silver(), target)


def gold_correlations(target: str) -> pd.DataFrame:
    """Correlations sur la couche gold : les colonnes vues par le modele."""
    return _correlations(load_gold(), target)


def log_correlations(
    matrice: pd.DataFrame, target: str, *, couche: str, nombre: int = 10
) -> None:
    """Features les plus liees a la cible, puis paires redondantes."""
    avec_cible = matrice[target].drop(target).sort_values(key=abs, ascending=False)

    print(f"\n[CORRELATION] couche {couche}, avec {target}, les {nombre} plus fortes")
    for nom, valeur in avec_cible.head(nombre).items():
        sens = "augmente" if valeur > 0 else "diminue "
        print(f"  {nom:<30} {valeur:+.2f}  ({sens} la cible)")

    # `stack` conserve les NaN, d'ou le filtrage explicite apres coup plutot
    # qu'un `where` en amont.
    paires = matrice.drop(index=target, columns=target).stack()
    paires = paires[[gauche < droite for gauche, droite in paires.index]]
    paires = paires[paires.abs() >= SEUIL_REDONDANCE]

    print(
        f"\n[REDONDANCE] couche {couche}, paires de features liees entre elles "
        f"(seuil |r| >= {SEUIL_REDONDANCE:.2f})"
    )
    if paires.empty:
        print("  aucune")
        return
    for (gauche, droite), valeur in paires.sort_values(
        key=abs, ascending=False
    ).items():
        print(f"  {gauche:<30} {droite:<30} {valeur:+.2f}")


def filter_correlations(matrice: pd.DataFrame, *, echo: bool = True) -> pd.DataFrame:
    """Ne conserve que les correlations marquantes, diagonale exclue.

    Les cases retenues sont celles au-dessus de `SEUIL_POSITIF` ou en dessous
    de `SEUIL_NEGATIF` ; les autres deviennent vides. Les lignes et colonnes
    qui n'en gardent aucune sont retirees : une variable sans lien avec les
    autres n'a rien a montrer.
    """
    retenues = (matrice > SEUIL_POSITIF) | (matrice < SEUIL_NEGATIF)
    filtree = matrice.where(retenues & (matrice != 1))

    lignes = filtree.notna().any(axis=1)
    colonnes = filtree.notna().any(axis=0)
    filtree = filtree.loc[lignes, colonnes]

    if echo:
        print(
            f"\n[FILTRE] cases conservees : r > {SEUIL_POSITIF:+.2f} ou "
            f"r < {SEUIL_NEGATIF:+.2f}, diagonale exclue "
            f"-> {len(filtree)} colonnes sur {len(matrice)}"
        )

    return filtree


def analyser_correlations(*, classification: str, regression: str) -> None:
    """Les deux recherches du notebook, en un appel.

    Les logs sont lus sur la gold, celle que les modeles consomment. Les
    figures viennent de la silver : une colonne par variable metier, donc une
    matrice lisible, la ou les 67 colonnes numeriques de la gold ne le sont pas.
    """
    log_seuils()

    roles = ((classification, "classification"), (regression, "regression"))
    for numero, (cible, role) in enumerate(roles, start=1):
        log_titre(f"ANALYSE {numero} - {cible.upper()} (cible de {role})")
        log_correlations(gold_correlations(cible), cible, couche="gold")

    matrice = silver_correlations(regression)
    figures = (
        (matrice, "Corrélations — couche silver", "correlations_silver"),
        (
            filter_correlations(matrice),
            "Corrélations marquantes — couche silver",
            "correlations_silver_marquantes",
        ),
    )

    for donnees, titre, nom in figures:
        figure = correlation_matrix_figure(donnees, titre=titre)
        save_correlation_figure(figure, nom)
        display_figure(figure)
