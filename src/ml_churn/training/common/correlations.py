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

import pandas as pd

from ml_churn.training.common.data import load_gold, load_silver

# Colonnes numeriques sans interet pour l'analyse : identifiants techniques.
EXCLUSIONS = ("id",)

# Au-dela, deux features disent a peu près la meme chose.
SEUIL_REDONDANCE = 0.8

# Bornes du filtre : on ne garde que les liens positifs francs et les liens
# negatifs deja notables. La bande est asymetrique parce que les correlations
# negatives sont plus rares ici, et donc informatives plus tot.
SEUIL_POSITIF = 0.5
SEUIL_NEGATIF = -0.2


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

    print(f"\n[REDONDANCE] couche {couche}, paires au-dela de {SEUIL_REDONDANCE:.0%}")
    if paires.empty:
        print("  aucune")
        return
    for (gauche, droite), valeur in paires.sort_values(
        key=abs, ascending=False
    ).items():
        print(f"  {gauche:<30} {droite:<30} {valeur:+.2f}")


def filter_correlations(matrice: pd.DataFrame) -> pd.DataFrame:
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
    return filtree.loc[lignes, colonnes]
