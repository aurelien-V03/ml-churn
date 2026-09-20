"""Boites a moustaches : reperer les valeurs extremes de chaque variable.

Complementaire des histogrammes : la ou l'histogramme montre la forme de la
distribution, le boxplot isole individuellement chaque point hors moustaches
(au-dela de 1.5 x IQR), meme s'il n'y en a qu'un sur 5000.

Les variables tres asymetriques sont tracees en echelle logarithmique : sur une
distribution log-normale, le critere IQR classe une grande partie de la traine
en valeurs extremes, ce qui ecrase la boite et rend le graphique illisible.

Usage :
    uv run python src/visualization/scripts/plot_boxplots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.scripts.common import (
    CSV_PATH,
    export_figure,
    load_dataset,
    numeric_column,
)

PREFIX = "box"


@dataclass(frozen=True)
class BoxSpec:
    """Une variable a tracer, et l'echelle qui la rend lisible."""

    column: str
    label: str
    log_y: bool = False


BOXPLOTS: tuple[BoxSpec, ...] = (
    # Distributions fortement asymetriques : echelle logarithmique.
    BoxSpec("sieges_souscrits", "Sieges souscrits", log_y=True),
    BoxSpec("utilisateurs_actifs", "Utilisateurs actifs", log_y=True),
    BoxSpec(
        "revenu_mensuel_recurrent_eur", "Revenu mensuel recurrent (EUR)", log_y=True
    ),
    BoxSpec("valeur_vie_client_eur", "Valeur vie client (EUR)", log_y=True),
    BoxSpec("heures_usage_30j", "Heures d'usage (30 jours)", log_y=True),
    BoxSpec(
        "derniere_connexion_jours", "Jours depuis la derniere connexion", log_y=True
    ),
    # Variables bornees ou peu etalees : echelle lineaire.
    BoxSpec("anciennete_mois", "Anciennete (mois)"),
    BoxSpec("taux_adoption_pct", "Taux d'adoption (%)"),
    BoxSpec("csat", "CSAT (1 a 5)"),
    BoxSpec("connexions_30j", "Connexions (30 jours)"),
    BoxSpec("fonctionnalites_utilisees", "Fonctionnalites utilisees"),
    BoxSpec("nb_integrations", "Integrations"),
    BoxSpec("tickets_support_90j", "Tickets support (90 jours)"),
    BoxSpec("delai_reponse_support_h", "Delai de reponse du support (h)"),
    BoxSpec("retards_paiement_12m", "Retards de paiement (12 mois)"),
    BoxSpec("sante_compte_fin_periode", "Sante du compte en fin de periode"),
)


def _outlier_count(values: pd.Series) -> int:
    """Nombre de points au-dela de 1.5 x IQR, le critere du boxplot."""
    first_quartile, third_quartile = values.quantile([0.25, 0.75])
    iqr = third_quartile - first_quartile
    return int(
        (
            (values < first_quartile - 1.5 * iqr)
            | (values > third_quartile + 1.5 * iqr)
        ).sum()
    )


def _render(values: pd.Series, spec: BoxSpec, *, show: bool) -> Path:
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.boxplot(
        values.to_numpy(),
        orientation="vertical",
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#4c72b0", "alpha": 0.6},
        medianprops={"color": "#c44e52", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )

    if spec.log_y:
        # Une valeur nulle n'a pas de logarithme : l'echelle demarre au minimum
        # strictement positif.
        axes.set_yscale("symlog", linthresh=1)

    axes.set_title(spec.label)
    axes.set_ylabel(spec.label)
    axes.set_xticks([])
    axes.spines[["top", "right", "bottom"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    return export_figure(figure, spec.column, f"{PREFIX}_{spec.column}", show=show)


def plot_boxplots(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un boxplot par variable numerique. Retourne les PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{CSV_PATH.name} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in BOXPLOTS:
        values = numeric_column(df, spec.column)
        path = _render(values, spec, show=show)
        paths.append(path)

        if echo:
            scale = "log" if spec.log_y else "lineaire"
            print(
                f"  {spec.column:<30} {_outlier_count(values):>4} points hors "
                f"moustaches ({scale})"
            )

    return paths


if __name__ == "__main__":
    plot_boxplots(show=False)
