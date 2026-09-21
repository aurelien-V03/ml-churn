"""Taux de churn par tranche d'une variable.

Le bon rendu face a une cible binaire : un boxplot groupe par churn donne deux
boites qui se chevauchent, alors que le taux de churn par tranche montre a
partir de quel seuil le risque decolle -- une information directement
actionnable.

Usage :
    uv run python src/visualization/scripts/plot_churn_rates.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.scripts.common import CSV_PATH, export_figure, load_dataset

# Ce graphique croise une colonne avec la cible : dossier propre au type.
FOLDER = "churn_rate"
PREFIX = "churnrate"

CIBLE = "churn"


@dataclass(frozen=True)
class ChurnRateSpec:
    """Une variable, et les tranches dans lesquelles mesurer le taux de churn."""

    column: str
    label: str
    # Variable numerique : bornes des tranches et leurs etiquettes.
    bornes: tuple[float, ...] = ()
    etiquettes: tuple[str, ...] = ()
    # Variable categorielle : modalites dans l'ordre d'affichage voulu.
    ordre: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"{PREFIX}_{self.column}"


CHURN_RATES: tuple[ChurnRateSpec, ...] = (
    ChurnRateSpec(
        column="delai_reponse_support_h",
        label="Delai de reponse du support",
        # Bornes calquees sur les SLA du catalogue : 24 h Starter, 12 h Pro,
        # 6 h Business, 3 h Enterprise.
        bornes=(0, 6, 12, 24, np.inf),
        etiquettes=("0-6h", "6-12h", "12-24h", "24h+"),
    ),
    ChurnRateSpec(
        column="tickets_support_90j",
        label="Tickets support (90 jours)",
        # Premiere borne a -1 : la tranche "0" doit contenir les clients qui
        # n'ont ouvert aucun ticket.
        bornes=(-1, 0, 2, 5, 8, np.inf),
        etiquettes=("0", "1-2", "3-5", "6-8", "9+"),
    ),
    ChurnRateSpec(
        column="retards_paiement_12m",
        label="Retards de paiement (12 mois)",
        # Premiere borne a -1 : la tranche "0" doit contenir les clients sans
        # aucun retard, plus de la moitie de l'effectif.
        bornes=(-1, 0, 1, 2, np.inf),
        etiquettes=("0", "1", "2", "3+"),
    ),
    ChurnRateSpec(
        column="taille_entreprise",
        label="Taille d'entreprise",
        # Categorielle : de la plus petite structure a la plus grande.
        ordre=("TPE", "PME", "ETI", "GE"),
    ),
)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Garde l'index, pour pouvoir croiser plusieurs colonnes."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def _modalites(df: pd.DataFrame, spec: ChurnRateSpec) -> pd.Series:
    """Casse et espaces harmonises, puis modalites limitees a `ordre`."""
    valeurs = df[spec.column].astype("string").str.strip().str.upper()
    return pd.Categorical(valeurs, categories=list(spec.ordre), ordered=True)


def _taux_par_tranche(df: pd.DataFrame, spec: ChurnRateSpec) -> pd.DataFrame:
    if spec.ordre:
        tranches = _modalites(df, spec)
    else:
        tranches = pd.cut(
            _numeric(df, spec.column),
            bins=list(spec.bornes),
            labels=list(spec.etiquettes),
        )

    groupes = pd.DataFrame({"tranche": tranches, "churn": _numeric(df, CIBLE)}).dropna()

    return groupes.groupby("tranche", observed=True)["churn"].agg(
        clients="size", taux="mean"
    )


def _render(df: pd.DataFrame, spec: ChurnRateSpec, *, show: bool) -> Path:
    stats = _taux_par_tranche(df, spec)
    taux_global = _numeric(df, CIBLE).mean()

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.scatter(
        stats.index.astype(str),
        stats["taux"] * 100,
        s=stats["clients"] / 4,
        color="#4c72b0",
        alpha=0.8,
        edgecolor="white",
        zorder=3,
        label="taux observe (taille = effectif)",
    )
    # Droite de tendance sur la position des tranches, ponderee par l'effectif :
    # les tranches les plus peuplees pesent davantage.
    positions = np.arange(len(stats))
    pente, ordonnee = np.polyfit(
        positions, stats["taux"] * 100, deg=1, w=stats["clients"]
    )
    axes.plot(
        stats.index.astype(str),
        pente * positions + ordonnee,
        color="#55a868",
        linewidth=2,
        zorder=2,
        label=f"tendance ({pente:+.1f} pt par tranche)",
    )
    axes.axhline(
        taux_global * 100,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label=f"taux global ({taux_global:.1%})",
    )

    # L'effectif conditionne la fiabilite de chaque point : on l'affiche.
    for tranche, ligne in stats.iterrows():
        axes.annotate(
            f"{ligne['taux']:.1%}\n(n={int(ligne['clients'])})",
            (str(tranche), ligne["taux"] * 100),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9,
        )

    axes.set_title(f"Taux de churn par tranche — {spec.label}")
    axes.set_xlabel(spec.label)
    axes.set_ylabel("Taux de churn (%)")
    axes.set_ylim(0, max(stats["taux"]) * 100 * 1.35)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)
    axes.legend()

    return export_figure(figure, FOLDER, spec.filename, show=show)


def plot_churn_rates(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un graphique de taux de churn par variable. Retourne les PNG."""
    df = load_dataset()
    if echo:
        print(f"{CSV_PATH.name} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in CHURN_RATES:
        paths.append(_render(df, spec, show=show))

        if echo:
            stats = _taux_par_tranche(df, spec)
            detail = " ".join(
                f"{tranche}:{ligne['taux']:.1%}(n={int(ligne['clients'])})"
                for tranche, ligne in stats.iterrows()
            )
            print(f"  {spec.column:<26} {detail}")

    return paths


if __name__ == "__main__":
    plot_churn_rates(show=False)
