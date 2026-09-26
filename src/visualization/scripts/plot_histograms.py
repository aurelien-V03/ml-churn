"""Graphiques exploratoires sur le dataset churn brut, lu en couche bronze.

Les valeurs categorielles sont normalisees (casse et espaces parasites) : la
table contient par exemple "TPE", " TPE " et "tpe", qui designent la meme
chose.

Un PNG par graphique dans src/visualization/graphs, nomme
<date>_<type_de_donnee>.png.

Usage :
    uv run python src/visualization/scripts/plot_histograms.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.scripts.common import (
    GRAPHS_DIR,
    PROJECT_ROOT,
    SOURCE,
    load_dataset,
)

# Les trois formats de date presents dans les donnees brutes.
DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y")

# Ordre naturel de la semaine, sinon matplotlib trie par frequence.
WEEKDAYS = [
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
]


@dataclass(frozen=True)
class ChartSpec:
    """Un graphique : la colonne source, son titre et son rendu."""

    column: str
    title: str
    xlabel: str
    display: Callable[[str], str] = str.capitalize
    order: tuple[str, ...] | None = None
    log_x: bool = False
    # Un dossier par colonne, le type de graphique en prefixe du fichier
    # (graphs/pays/histo_pays.png).
    prefix: str = "histo"


CHARTS: tuple[ChartSpec, ...] = (
    ChartSpec(
        column="jour_souscription",
        title="Nombre de souscriptions par jour de la semaine",
        xlabel="Jour de souscription",
        order=tuple(WEEKDAYS),
    ),
    ChartSpec(
        column="secteur",
        title="Nombre de clients par secteur",
        xlabel="Secteur",
    ),
    ChartSpec(
        column="pays",
        title="Nombre de clients par pays",
        xlabel="Pays",
    ),
    ChartSpec(
        column="taille_entreprise",
        title="Nombre de clients par taille d'entreprise",
        xlabel="Taille d'entreprise",
        display=str.upper,
        order=("tpe", "pme", "eti", "ge"),
    ),
    ChartSpec(
        column="plan",
        title="Nombre de clients par plan",
        xlabel="Plan",
        display=str.capitalize,
        order=("starter", "pro", "business", "enterprise"),
    ),
)

DATE_SPEC = ChartSpec(
    column="date_souscription",
    title="Nombre de souscriptions par mois",
    xlabel="Mois de souscription",
)

DELAI_SPEC = ChartSpec(
    column="delai_reponse_support_h",
    title="Distribution du delai de reponse du support",
    xlabel="Delai de reponse (heures)",
)

HEURES_USAGE_SPEC = ChartSpec(
    column="heures_usage_30j",
    title="Distribution des heures d'usage sur 30 jours",
    xlabel="Heures d'usage (30 jours)",
)

REVENU_SPEC = ChartSpec(
    column="revenu_mensuel_recurrent_eur",
    title="Distribution du revenu mensuel recurrent",
    xlabel="Revenu mensuel recurrent (EUR, scale log)",
    log_x=True,
)

# Cible du modele de regression : etalee de quelques centaines d'euros a deux
# millions, d'ou l'echelle logarithmique comme pour le revenu.
VALEUR_VIE_SPEC = ChartSpec(
    column="valeur_vie_client_eur",
    title="Distribution de la valeur vie client",
    xlabel="Valeur vie client (EUR, scale log)",
    log_x=True,
)

RETARDS_SPEC = ChartSpec(
    column="retards_paiement_12m",
    title="Nombre de clients par retards de paiement (12 mois)",
    xlabel="Retards de paiement sur 12 mois",
)

INTEGRATIONS_SPEC = ChartSpec(
    column="nb_integrations",
    title="Nombre de clients par nombre d'integrations",
    xlabel="Nombre d'integrations",
)

SIEGES_SPEC = ChartSpec(
    column="sieges_souscrits",
    title="Nombre de clients par nombre de sieges souscrits",
    xlabel="Sieges souscrits",
)

UTILISATEURS_ACTIFS_SPEC = ChartSpec(
    column="utilisateurs_actifs",
    title="Nombre de clients par nombre d'utilisateurs actifs",
    xlabel="Utilisateurs actifs",
)

ADOPTION_SPEC = ChartSpec(
    column="taux_adoption_pct",
    title="Nombre de clients par taux d'adoption",
    xlabel="Taux d'adoption (%) = utilisateurs actifs / sieges souscrits",
)

CONNEXIONS_SPEC = ChartSpec(
    column="connexions_30j",
    title="Nombre de clients par nombre de connexions sur 30 jours",
    xlabel="Connexions (30 jours)",
)

DERNIERE_CONNEXION_SPEC = ChartSpec(
    column="derniere_connexion_jours",
    title="Nombre de clients par anciennete de la derniere connexion",
    xlabel="Jours depuis la derniere connexion",
)

TICKETS_SPEC = ChartSpec(
    column="tickets_support_90j",
    title="Nombre de clients par nombre de tickets support (90 jours)",
    xlabel="Tickets support (90 jours)",
)

SANTE_COMPTE_SPEC = ChartSpec(
    column="sante_compte_fin_periode",
    title="Nombre de clients par score de sante du compte",
    xlabel="Sante du compte en fin de periode (0 a 100)",
)

CSAT_SPEC = ChartSpec(
    column="csat",
    title="Nombre de clients par note de satisfaction",
    xlabel="CSAT (1 a 5)",
)

ANCIENNETE_SPEC = ChartSpec(
    column="anciennete_mois",
    title="Nombre de clients par anciennete",
    xlabel="Anciennete (mois)",
)


def _normalize(series: pd.Series) -> pd.Series:
    """Aligne les variantes de casse et d'espaces, et ecarte les valeurs vides."""
    cleaned = series.astype("string").str.strip().str.casefold()
    return cleaned[cleaned.notna() & (cleaned != "")]


def _counts(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    counts = _normalize(df[spec.column]).value_counts()
    if spec.order is not None:
        counts = counts.reindex(list(spec.order), fill_value=0)
    else:
        counts = counts.sort_values(ascending=False)
    counts.index = [spec.display(str(value)) for value in counts.index]
    return counts


def _subscription_date_counts(df: pd.DataFrame) -> pd.Series:
    """Les trois formats de date du CSV sont ramenes a une date, agregee par mois.

    Le jour est trop fin pour etre lisible (plus de 1000 dates distinctes).
    """
    raw = df[DATE_SPEC.column].astype("string").str.strip()
    dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for format_source in DATE_FORMATS:
        restantes = raw.where(dates.isna())
        dates = dates.fillna(
            pd.to_datetime(restantes, format=format_source, errors="coerce")
        )

    non_converties = int(dates.isna().sum())
    if non_converties:
        print(f"  WARNING : {non_converties} dates non reconnues, ignorees")

    return dates.dt.strftime("%Y-%m").value_counts().sort_index()


def _numeric_values(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    return _numeric_column(df, spec.column)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Le CSV melange virgule decimale et unites ("3.1 h", "280.62 €", "40,0")."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(
        cleaned.str.replace(",", ".", regex=False), errors="coerce"
    ).dropna()


def _adoption_rate_counts(df: pd.DataFrame) -> pd.Series:
    """Taux recalcule plutot que lu dans le CSV.

    La colonne `taux_adoption_pct` est vide sur 250 lignes alors que le rapport
    est toujours calculable ; sur les lignes ou les deux existent, l'ecart est
    inferieur a 0.1 point. Arrondi a l'entier : le rapport brut donnerait des
    centaines de valeurs distinctes.
    """
    actifs = _numeric_column(df, "utilisateurs_actifs")
    sieges = _numeric_column(df, "sieges_souscrits")
    taux = (actifs / sieges * 100).where(sieges > 0)
    return taux.dropna().round().astype(int).value_counts().sort_index()


def _numeric_counts(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    """Valeurs entieres (anciennete, note) : tri par valeur croissante, pas par frequence."""
    return _numeric_values(df, spec).astype(int).value_counts().sort_index()


def _output_path(spec: ChartSpec) -> Path:
    return GRAPHS_DIR / spec.column / f"{spec.prefix}_{spec.column}.png"


def _render(counts: pd.Series, spec: ChartSpec, *, show: bool) -> Path:
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.bar([str(label) for label in counts.index], counts.to_numpy(), color="#4c72b0")
    axes.set_title(spec.title)
    axes.set_xlabel(spec.xlabel)
    axes.set_ylabel("Nombre de clients")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    # Beaucoup de modalites (anciennete) : on allege les etiquettes.
    if len(counts) > 15:
        axes.tick_params(axis="x", labelsize=7, rotation=90)
    else:
        axes.tick_params(axis="x", rotation=30)

    return _export_figure(figure, spec, show=show)


def _export_figure(figure: plt.Figure, spec: ChartSpec, *, show: bool) -> Path:
    figure.tight_layout()

    output_path = _output_path(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)

    if show:
        plt.show()
    plt.close(figure)

    return output_path


def _render_numeric_bars(counts: pd.Series, spec: ChartSpec, *, show: bool) -> Path:
    """Comptage par valeur sur un axe numerique.

    Comme pour un barplot classique, y est le nombre de clients ayant cette
    valeur exacte -- mais l'axe x reste numerique, seule facon de rester lisible
    quand la variable prend des centaines de valeurs distinctes.
    """
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.bar(counts.index.to_numpy(), counts.to_numpy(), width=1.0, color="#4c72b0")
    axes.set_title(spec.title)
    axes.set_xlabel(spec.xlabel)
    axes.set_ylabel("Nombre de clients")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    return _export_figure(figure, spec, show=show)


def _render_histogram(values: pd.Series, spec: ChartSpec, *, show: bool) -> Path:
    """Variable continue : un histogramme, pas une barre par valeur distincte."""
    figure, axes = plt.subplots(figsize=(10, 5))

    if spec.log_x:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
        axes.set_xscale("log")
    else:
        bins = 30

    axes.hist(values.to_numpy(), bins=bins, color="#4c72b0", edgecolor="white")
    axes.set_title(spec.title)
    axes.set_xlabel(spec.xlabel)
    axes.set_ylabel("Nombre de clients")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    return _export_figure(figure, spec, show=show)


def plot_histograms(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere les six graphiques et retourne les chemins des PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{SOURCE} : {len(df)} lignes lues")

    paths = [_render(_subscription_date_counts(df), DATE_SPEC, show=show)]
    paths += [_render(_counts(df, spec), spec, show=show) for spec in CHARTS]
    paths.append(
        _render(_numeric_counts(df, ANCIENNETE_SPEC), ANCIENNETE_SPEC, show=show)
    )
    paths.append(_render(_numeric_counts(df, CSAT_SPEC), CSAT_SPEC, show=show))
    for spec in (RETARDS_SPEC, INTEGRATIONS_SPEC, TICKETS_SPEC):
        paths.append(_render(_numeric_counts(df, spec), spec, show=show))

    for spec in (
        SIEGES_SPEC,
        UTILISATEURS_ACTIFS_SPEC,
        CONNEXIONS_SPEC,
        DERNIERE_CONNEXION_SPEC,
        SANTE_COMPTE_SPEC,
    ):
        paths.append(_render_numeric_bars(_numeric_counts(df, spec), spec, show=show))

    paths.append(
        _render_numeric_bars(_adoption_rate_counts(df), ADOPTION_SPEC, show=show)
    )

    for spec in (DELAI_SPEC, HEURES_USAGE_SPEC, REVENU_SPEC, VALEUR_VIE_SPEC):
        paths.append(_render_histogram(_numeric_values(df, spec), spec, show=show))

    if echo:
        for path in paths:
            print(f"  export : {path.relative_to(PROJECT_ROOT)}")

    return paths


if __name__ == "__main__":
    plot_histograms(show=False)
