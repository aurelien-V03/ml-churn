"""Nuages de points : taux de churn en fonction d'une variable.

Un point par valeur observee de la variable : abscisse la valeur, ordonnee le
taux de churn des clients qui la partagent.

Usage :
    uv run python src/visualization/scripts/plot_scatters.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.scripts.common import CSV_PATH, export_figure, load_dataset

PREFIX = "scatter"

CIBLE = "churn"


@dataclass(frozen=True)
class ScatterSpec:
    """Une variable a croiser avec le taux de churn.

    `calcul` permet de tracer une grandeur derivee plutot qu'une colonne du
    CSV ; `column` sert alors uniquement a nommer le fichier.
    """

    column: str
    label: str
    # Dossier d'export : celui de la colonne source, qui n'est pas toujours
    # `column` quand la grandeur est derivee.
    folder: str
    calcul: Callable[[pd.DataFrame], pd.Series] | None = None


def _taux_fonctionnalites(df: pd.DataFrame) -> pd.Series:
    """Part des fonctionnalites du plan reellement utilisees, en pourcentage.

    Arrondi a l'entier : le rapport brut prend des centaines de valeurs
    distinctes (n/8, n/16, n/26, n/40 selon le plan).
    """
    utilisees = _numeric(df, "fonctionnalites_utilisees")
    total = _numeric(df, "fonctionnalites_total").replace(0, pd.NA)
    return (utilisees / total * 100).round()


SCATTERS: tuple[ScatterSpec, ...] = (
    ScatterSpec(
        column="taux_fonctionnalites",
        label="Fonctionnalites utilisees (% du plan)",
        folder="fonctionnalites_utilisees",
        calcul=_taux_fonctionnalites,
    ),
)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Garde l'index, pour pouvoir croiser plusieurs colonnes."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def _churn_rate(df: pd.DataFrame, spec: ScatterSpec) -> pd.DataFrame:
    """Taux de churn et effectif pour chaque valeur observee de la variable."""
    valeurs = spec.calcul(df) if spec.calcul else _numeric(df, spec.column)
    donnees = pd.DataFrame({"valeur": valeurs, CIBLE: _numeric(df, CIBLE)}).dropna()
    return donnees.groupby("valeur")[CIBLE].agg(clients="size", taux="mean")


def _render(df: pd.DataFrame, spec: ScatterSpec, *, show: bool) -> Path:
    stats = _churn_rate(df, spec)

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.scatter(
        stats.index,
        stats["taux"] * 100,
        s=12,
        color="#4c72b0",
        alpha=0.6,
        label="taux observe",
    )

    # Droite de tendance ponderee par l'effectif : une valeur partagee par
    # 300 clients pese plus qu'une valeur isolee.
    pente, ordonnee = np.polyfit(
        stats.index, stats["taux"] * 100, deg=1, w=stats["clients"]
    )
    axes.plot(
        stats.index,
        pente * np.asarray(stats.index) + ordonnee,
        color="#55a868",
        linewidth=2,
        label=f"tendance ({pente:+.2f} pt par unite)",
    )
    axes.legend()

    axes.set_title(f"Taux de churn selon {spec.label}")
    axes.set_xlabel(spec.label)
    axes.set_ylabel("Taux de churn (%)")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(alpha=0.3)

    return export_figure(figure, spec.folder, f"{PREFIX}_{spec.column}", show=show)


def plot_scatters(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un nuage de points par variable. Retourne les PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{CSV_PATH.name} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in SCATTERS:
        paths.append(_render(df, spec, show=show))

        if echo:
            stats = _churn_rate(df, spec)
            pente, _ = np.polyfit(
                stats.index, stats["taux"] * 100, deg=1, w=stats["clients"]
            )
            print(
                f"  {spec.column:<26} {len(stats)} valeurs distinctes | "
                f"tendance {pente:+.2f} pt par unite"
            )

    return paths


if __name__ == "__main__":
    plot_scatters(show=False)
