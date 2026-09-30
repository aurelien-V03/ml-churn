"""Moyenne, ou repartition des modalites, d'une variable par tranche d'une autre.

Le bon rendu quand la question porte sur un niveau et non sur une forme : un
histogramme montre la distribution d'ensemble, une boite a moustaches la
dispersion, mais ni l'un ni l'autre ne dit si le niveau moyen se deplace d'une
tranche a l'autre. La repartition va plus loin que la moyenne : elle montre
quelles modalites se deplacent quand la moyenne, elle, ne bouge pas.

Usage :
    uv run python src/visualization/scripts/plot_averages.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization.scripts.common import (
    SOURCE,
    annee_souscription,
    export_figure,
    load_dataset,
    mois_souscription,
)

# Ce qui est mesure par tranche : prefixe de fichier et libelle des ordonnees.
MESURES: dict[str, tuple[str, str]] = {
    "moyenne": ("moyenne", ""),
    "somme": ("volume", ""),
    "repartition": ("repartition", "Part des clients (%)"),
}

# Palette ordonnee du rouge au vert : les modalites d'une note se lisent dans
# l'ordre, une palette categorielle le masquerait.
PALETTE_ORDONNEE = "RdYlGn"

# Au-dela, les etiquettes par barre se chevauchent : on ne garde que l'axe.
MAX_ETIQUETTES = 8

# En dessous, une tranche n'est pas representative : quelques clients suffisent
# a faire bouger une moyenne de plusieurs points.
EFFECTIF_MINIMUM = 10


@dataclass(frozen=True)
class AverageSpec:
    """La variable moyennee, et celle qui definit les tranches."""

    value_column: str
    value_label: str
    group_label: str
    # Rend les tranches a partir du DataFrame : elles sont calculees, pas lues.
    deriver: Callable[[pd.DataFrame], pd.Categorical]
    group_name: str
    # Bornes de l'axe des ordonnees, quand l'echelle de la variable est connue.
    limites: tuple[float, float] | None = None
    # Mesure portee par l'axe des ordonnees : cle de MESURES.
    mesure: str = "moyenne"
    # Dossier de sortie, quand le graphique appartient a une serie deja etablie
    # plutot qu'au dossier de sa propre colonne.
    dossier: str | None = None

    @property
    def folder(self) -> str:
        """Le dossier de la variable mesuree, a cote de ses autres graphiques."""
        return self.dossier or self.value_column

    @property
    def filename(self) -> str:
        prefixe, _ordonnees = MESURES[self.mesure]
        return f"{prefixe}_{self.value_column}_par_{self.group_name}"


AVERAGES: tuple[AverageSpec, ...] = (
    AverageSpec(
        value_column="csat",
        value_label="Note CSAT moyenne",
        group_label="Annee de souscription",
        deriver=annee_souscription,
        group_name="annee",
        # Echelle complete de la note : une echelle tronquee amplifierait
        # visuellement un ecart de 0.1 point.
        limites=(0, 5),
    ),
    AverageSpec(
        value_column="tickets_support_90j",
        value_label="Tickets support (total)",
        group_label="Mois de souscription",
        deriver=mois_souscription,
        group_name="mois",
        mesure="somme",
    ),
    AverageSpec(
        value_column="delai_reponse_support_h",
        value_label="Delai de reponse moyen (h)",
        group_label="Mois de souscription",
        deriver=mois_souscription,
        group_name="mois",
    ),
    AverageSpec(
        value_column="csat",
        value_label="Note CSAT",
        group_label="Annee de souscription",
        deriver=annee_souscription,
        group_name="annee",
        mesure="repartition",
        # Range avec les autres graphiques qui croisent une variable avec le
        # churn : c'est la meme lecture par cohorte.
        dossier="churn_rate",
    ),
)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Garde l'index, pour pouvoir croiser plusieurs colonnes."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def _moyenne_par_tranche(df: pd.DataFrame, spec: AverageSpec) -> pd.DataFrame:
    groupes = pd.DataFrame(
        {"tranche": spec.deriver(df), "valeur": _numeric(df, spec.value_column)}
    ).dropna()

    stats = groupes.groupby("tranche", observed=True)["valeur"].agg(
        clients="size", moyenne="mean", somme="sum"
    )

    return stats[stats["clients"] >= EFFECTIF_MINIMUM]


def _repartition_par_tranche(df: pd.DataFrame, spec: AverageSpec) -> pd.DataFrame:
    """Part de chaque modalite dans chaque tranche, en pourcentage.

    Les lignes sans valeur sont ecartees : chaque tranche totalise 100 % des
    clients qu'elle contient et qui ont une note.
    """
    groupes = pd.DataFrame(
        {"tranche": spec.deriver(df), "valeur": _numeric(df, spec.value_column)}
    ).dropna()

    effectifs = pd.crosstab(groupes["tranche"], groupes["valeur"].astype(int))

    return effectifs.div(effectifs.sum(axis=1), axis=0) * 100


def _render_repartition(df: pd.DataFrame, spec: AverageSpec) -> plt.Figure:
    """Barres empilees a 100 % : une couleur par modalite, de la plus basse a la plus haute."""
    parts = _repartition_par_tranche(df, spec)
    couleurs = plt.get_cmap(PALETTE_ORDONNEE)(
        [index / max(len(parts.columns) - 1, 1) for index in range(len(parts.columns))]
    )

    figure, axes = plt.subplots(figsize=(10, 5.5))
    bas = pd.Series(0.0, index=parts.index)

    for couleur, modalite in zip(couleurs, parts.columns, strict=True):
        hauteurs = parts[modalite]
        axes.bar(
            parts.index.astype(str),
            hauteurs,
            bottom=bas,
            color=couleur,
            zorder=3,
            label=f"{spec.value_label} {modalite}",
        )

        # Etiquette au centre du segment, tant qu'il reste lisible.
        for tranche in parts.index:
            hauteur = hauteurs[tranche]
            if hauteur >= 3:
                axes.annotate(
                    f"{hauteur:.1f}%",
                    (str(tranche), bas[tranche] + hauteur / 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        bas = bas + hauteurs

    axes.set_ylim(0, 100)
    # Legende hors du trace : les barres occupent toute la hauteur.
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    return figure


def _render_barres(df: pd.DataFrame, spec: AverageSpec) -> plt.Figure:
    """Une barre par tranche, pour la moyenne comme pour la somme."""
    stats = _moyenne_par_tranche(df, spec)
    hauteurs = stats[spec.mesure]

    figure, axes = plt.subplots(figsize=(11, 5.5))
    axes.bar(stats.index.astype(str), hauteurs, color="#4c72b0", zorder=3)

    # La moyenne se compare a la moyenne d'ensemble ; une somme par tranche,
    # non : le total de reference dependrait de la taille des tranches.
    if spec.mesure == "moyenne":
        moyenne_globale = _numeric(df, spec.value_column).mean()
        axes.axhline(
            moyenne_globale,
            color="#c44e52",
            linestyle="--",
            linewidth=1.2,
            zorder=4,
            label=f"moyenne globale ({moyenne_globale:.2f})",
        )
        axes.legend()

    # L'effectif conditionne la fiabilite de chaque tranche : on l'affiche, tant
    # que les etiquettes ne se chevauchent pas.
    if len(stats) <= MAX_ETIQUETTES:
        gabarit = "{:.0f}" if spec.mesure == "somme" else "{:.2f}"
        for tranche, ligne in stats.iterrows():
            axes.annotate(
                gabarit.format(ligne[spec.mesure]) + f"\n(n={int(ligne['clients'])})",
                (str(tranche), ligne[spec.mesure]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

    if spec.limites is not None:
        axes.set_ylim(*spec.limites)

    return figure


def _render(df: pd.DataFrame, spec: AverageSpec, *, show: bool) -> Path:
    _prefixe, ordonnees = MESURES[spec.mesure]
    dessiner = _render_repartition if spec.mesure == "repartition" else _render_barres
    figure = dessiner(df, spec)

    axes = figure.axes[0]
    axes.set_title(f"{spec.value_label} par {spec.group_label.lower()}")
    axes.set_xlabel(spec.group_label)
    axes.set_ylabel(ordonnees or spec.value_label)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)
    if len(axes.get_xticks()) > MAX_ETIQUETTES:
        axes.tick_params(axis="x", labelsize=8, rotation=90)

    return export_figure(figure, spec.folder, spec.filename, show=show)


def plot_averages(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un graphique de moyenne par tranche. Retourne les PNG."""
    df = load_dataset()
    if echo:
        print(f"{SOURCE} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in AVERAGES:
        paths.append(_render(df, spec, show=show))

        if echo:
            if spec.mesure == "repartition":
                parts = _repartition_par_tranche(df, spec)
                detail = " ".join(
                    f"{tranche}:" + "/".join(f"{part:.0f}%" for part in ligne)
                    for tranche, ligne in parts.iterrows()
                )
                detail = f"notes {'/'.join(map(str, parts.columns))} | {detail}"
            else:
                stats = _moyenne_par_tranche(df, spec)
                gabarit = "{:.0f}" if spec.mesure == "somme" else "{:.2f}"
                # Series longues : un point sur six suffit a lire la tendance.
                apercu = stats if len(stats) <= MAX_ETIQUETTES else stats.iloc[::6]
                detail = " ".join(
                    f"{tranche}:{gabarit.format(ligne[spec.mesure])}"
                    f"(n={int(ligne['clients'])})"
                    for tranche, ligne in apercu.iterrows()
                )
            manquantes = int(_numeric(df, spec.value_column).isna().sum())
            suffixe = (
                f" [{manquantes} valeurs manquantes ignorees]" if manquantes else ""
            )
            print(f"  {spec.value_column:<8} {spec.mesure:<12} {detail}{suffixe}")

    return paths


if __name__ == "__main__":
    plot_averages(show=False)
