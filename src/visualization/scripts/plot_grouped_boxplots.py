"""Boites a moustaches groupees : une distribution par modalite d'une autre colonne.

Le bon rendu quand une variable est discrete a peu de modalites : un nuage de
points y superposerait des milliers de clients sur quelques positions, alors
qu'un boxplot par groupe montre le deplacement des distributions.

Usage :
    uv run python src/visualization/scripts/plot_grouped_boxplots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.scripts.common import SOURCE, export_figure, load_dataset

# Ces graphiques croisent deux colonnes : ils ne peuvent pas aller dans le
# dossier de l'une d'elles, d'ou un dossier propre au type.
FOLDER = "grouped_boxplot"
PREFIX = "grouped"


@dataclass(frozen=True)
class GroupedBoxSpec:
    """La variable mesuree, et celle qui definit les groupes."""

    value_column: str
    group_column: str
    value_label: str
    group_label: str

    @property
    def filename(self) -> str:
        return f"{PREFIX}_{self.value_column}_par_{self.group_column}"


GROUPED_BOXPLOTS: tuple[GroupedBoxSpec, ...] = (
    GroupedBoxSpec(
        value_column="tickets_support_90j",
        group_column="csat",
        value_label="Tickets support (90 jours)",
        group_label="CSAT",
    ),
    GroupedBoxSpec(
        value_column="delai_reponse_support_h",
        group_column="csat",
        value_label="Delai de reponse du support (h)",
        group_label="CSAT",
    ),
)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Garde l'index, contrairement a common.numeric_column qui filtre les vides."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def _render(df: pd.DataFrame, spec: GroupedBoxSpec, *, show: bool) -> Path:
    values = _numeric(df, spec.value_column)
    groups = _numeric(df, spec.group_column)

    modalites = sorted(groups.dropna().unique())
    distributions = [values[(groups == m) & values.notna()] for m in modalites]

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.boxplot(
        distributions,
        tick_labels=[f"{m:g}" for m in modalites],
        patch_artist=True,
        boxprops={"facecolor": "#4c72b0", "alpha": 0.6},
        medianprops={"color": "#c44e52", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    axes.set_title(f"{spec.value_label} selon {spec.group_label}")
    axes.set_xlabel(spec.group_label)
    axes.set_ylabel(spec.value_label)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    return export_figure(figure, FOLDER, spec.filename, show=show)


def plot_grouped_boxplots(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un boxplot groupe par relation. Retourne les PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{SOURCE} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in GROUPED_BOXPLOTS:
        paths.append(_render(df, spec, show=show))

        if echo:
            values = _numeric(df, spec.value_column)
            groups = _numeric(df, spec.group_column)
            medianes = " ".join(
                f"{m:g}:{values[groups == m].median():g}"
                for m in sorted(groups.dropna().unique())
            )
            print(
                f"  {spec.value_column:<26} par {spec.group_column} | "
                f"correlation {values.corr(groups):+.3f} | medianes {medianes}"
            )

    return paths


if __name__ == "__main__":
    plot_grouped_boxplots(show=False)
