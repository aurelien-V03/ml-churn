"""Nuages de points : taux de churn en fonction d'une variable.

Un point par valeur observee de la variable : abscisse la valeur, ordonnee le
taux de churn des clients qui la partagent.

Usage :
    uv run python src/visualization/scripts/plot_scatters.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.scripts.common import CSV_PATH, export_figure, load_dataset

# Ce graphique croise une colonne avec la cible : dossier propre au type.
FOLDER = "scatter"
PREFIX = "scatter"

CIBLE = "churn"


@dataclass(frozen=True)
class ScatterSpec:
    """Une variable a croiser avec le taux de churn."""

    column: str
    label: str


SCATTERS: tuple[ScatterSpec, ...] = (
    ScatterSpec(
        column="delai_reponse_support_h",
        label="Delai de reponse du support (h)",
    ),
)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Garde l'index, pour pouvoir croiser plusieurs colonnes."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def _churn_rate(df: pd.DataFrame, spec: ScatterSpec) -> pd.Series:
    """Taux de churn pour chaque valeur observee de la variable."""
    donnees = pd.DataFrame(
        {"valeur": _numeric(df, spec.column), CIBLE: _numeric(df, CIBLE)}
    ).dropna()
    return donnees.groupby("valeur")[CIBLE].mean()


def _render(df: pd.DataFrame, spec: ScatterSpec, *, show: bool) -> Path:
    taux = _churn_rate(df, spec)

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.scatter(taux.index, taux.to_numpy() * 100, s=12, color="#4c72b0", alpha=0.6)

    axes.set_title(f"Taux de churn selon {spec.label}")
    axes.set_xlabel(spec.label)
    axes.set_ylabel("Taux de churn (%)")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(alpha=0.3)

    return export_figure(figure, FOLDER, f"{PREFIX}_{spec.column}", show=show)


def plot_scatters(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un nuage de points par variable. Retourne les PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{CSV_PATH.name} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in SCATTERS:
        paths.append(_render(df, spec, show=show))

        if echo:
            taux = _churn_rate(df, spec)
            print(f"  {spec.column:<26} {len(taux)} valeurs distinctes en abscisse")

    return paths


if __name__ == "__main__":
    plot_scatters(show=False)
