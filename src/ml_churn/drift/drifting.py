"""Derive des donnees : Kolmogorov-Smirnov et Population Stability Index.

Deux lectures complementaires du meme ecart entre une population de reference
-- celle sur laquelle le modele a appris -- et une population courante.

`KS` compare les fonctions de repartition et retourne leur ecart maximal, avec
la p-valeur du test. Il est sensible a un deplacement de la distribution, meme
faible, des que l'echantillon est grand.

`PSI` compare les effectifs tranche par tranche. Il ne teste rien : il chiffre
l'ampleur du deplacement, avec des seuils d'usage (`SEUILS_PSI`) qui servent de
convention dans l'industrie du credit d'ou il vient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Tranches utilisees pour le PSI, decoupees sur la population de reference.
TRANCHES = 10

# Observations minimales par tranche. En dessous, la moitie des tranches se
# vide et le logarithme du rapport s'emballe : le PSI annonce alors une derive
# massive la ou il ne mesure que la petitesse de l'echantillon.
MIN_PAR_TRANCHE = 5

# Evite les divisions par zero quand une tranche se vide : la convention est
# d'y substituer un effectif residuel plutot que d'ignorer la tranche.
EPSILON = 1e-6

# Lecture conventionnelle du PSI.
SEUILS_PSI: dict[str, float] = {"stable": 0.10, "moderee": 0.25}

# En deca, une colonne est traitee comme categorielle : ses valeurs font les
# tranches, plutot que des quantiles.
MAX_MODALITES = 12


@dataclass(frozen=True)
class ColonneDerive:
    """Derive mesuree sur une colonne."""

    colonne: str
    ks: float
    p_value: float
    psi: float

    @property
    def verdict(self) -> str:
        """Lecture conventionnelle du PSI, la plus lisible des deux mesures."""
        if self.psi < SEUILS_PSI["stable"]:
            return "stable"
        if self.psi < SEUILS_PSI["moderee"]:
            return "moderee"
        return "importante"


def ks(reference: pd.Series, courant: pd.Series) -> tuple[float, float]:
    """Statistique de Kolmogorov-Smirnov a deux echantillons, et sa p-valeur.

    La statistique est l'ecart maximal entre les deux fonctions de repartition,
    entre 0 et 1. La p-valeur repond a « un tel ecart est-il explicable par le
    hasard de l'echantillonnage ? » -- sous 0.05, non.
    """
    a = pd.to_numeric(reference, errors="coerce").dropna()
    b = pd.to_numeric(courant, errors="coerce").dropna()
    if a.empty or b.empty:
        return float("nan"), float("nan")

    resultat = ks_2samp(a, b)
    return float(resultat.statistic), float(resultat.pvalue)


def psi(reference: pd.Series, courant: pd.Series, tranches: int = TRANCHES) -> float:
    """Population Stability Index entre deux distributions.

    Somme sur les tranches de `(part_courante - part_reference) x
    ln(part_courante / part_reference)`. La formule est symetrique et penalise
    autant une tranche qui se vide qu'une qui se remplit.
    """
    a = pd.to_numeric(reference, errors="coerce").dropna()
    b = pd.to_numeric(courant, errors="coerce").dropna()
    if a.empty or b.empty:
        return float("nan")

    modalites = np.union1d(a.unique(), b.unique())
    if len(modalites) <= MAX_MODALITES:
        # Colonne categorielle ou binaire : chaque valeur est sa propre tranche.
        part_a = a.value_counts(normalize=True).reindex(modalites, fill_value=0)
        part_b = b.value_counts(normalize=True).reindex(modalites, fill_value=0)
    else:
        # Le nombre de tranches s'adapte au plus petit des deux echantillons.
        tranches = max(2, min(tranches, min(len(a), len(b)) // MIN_PAR_TRANCHE))

        # Les bornes viennent de la reference : c'est elle qui definit la
        # normalite a laquelle le courant est compare.
        bornes = np.unique(np.quantile(a, np.linspace(0, 1, tranches + 1)))
        bornes[0], bornes[-1] = -np.inf, np.inf
        part_a = pd.cut(a, bornes).value_counts(normalize=True).sort_index()
        part_b = pd.cut(b, bornes).value_counts(normalize=True).sort_index()

    ref = np.clip(part_a.to_numpy(dtype=float), EPSILON, None)
    cur = np.clip(part_b.to_numpy(dtype=float), EPSILON, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def rapport_derive(
    reference: pd.DataFrame, courant: pd.DataFrame, colonnes: list[str] | None = None
) -> list[ColonneDerive]:
    """Derive de chaque colonne commune aux deux populations.

    Triee par PSI decroissant : les colonnes qui ont le plus bouge en premier.
    """
    communes = colonnes or [
        colonne for colonne in reference.columns if colonne in courant.columns
    ]

    mesures = []
    for colonne in communes:
        if not pd.api.types.is_numeric_dtype(
            pd.to_numeric(reference[colonne], errors="coerce")
        ):
            continue
        statistique, p_value = ks(reference[colonne], courant[colonne])
        if np.isnan(statistique):
            continue
        mesures.append(
            ColonneDerive(
                colonne=colonne,
                ks=statistique,
                p_value=p_value,
                psi=psi(reference[colonne], courant[colonne]),
            )
        )

    return sorted(mesures, key=lambda mesure: mesure.psi, reverse=True)


def rapport_derive_csv(
    reference: Path | str, courant: Path | str, colonnes: list[str] | None = None
) -> list[ColonneDerive]:
    """Meme rapport, a partir de deux fichiers CSV."""
    return rapport_derive(pd.read_csv(reference), pd.read_csv(courant), colonnes)
