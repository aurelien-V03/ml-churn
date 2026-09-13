"""Graphiques exploratoires sur le dataset churn brut (docs/churn_saas_complet.csv).

Lecture directe du CSV, donc utilisable avant toute ingestion en base.
Les valeurs categorielles sont normalisees (casse et espaces parasites) : le
CSV contient par exemple "TPE", " TPE " et "tpe", qui designent la meme chose.

Un PNG par graphique dans src/visualization/graphs, nomme
<date>_<type_de_donnee>.png.

Usage :
    uv run python src/visualization/scripts/plot_donnees_clients.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "docs" / "churn_saas_complet.csv"
GRAPHS_DIR = PROJECT_ROOT / "src" / "visualization" / "graphs"

# Les trois formats de date presents dans le CSV brut.
FORMATS_DATE: tuple[str, ...] = ("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y")

# Ordre naturel de la semaine, sinon matplotlib trie par frequence.
JOURS_SEMAINE = [
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


CHARTS: tuple[ChartSpec, ...] = (
    ChartSpec(
        column="jour_souscription",
        title="Nombre de souscriptions par jour de la semaine",
        xlabel="Jour de souscription",
        order=tuple(JOURS_SEMAINE),
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
    ),
    ChartSpec(
        column="plan",
        title="Nombre de clients par plan",
        xlabel="Plan",
        display=str.capitalize,
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
    xlabel="Revenu mensuel recurrent (EUR, echelle log)",
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


def load_dataset() -> pd.DataFrame:
    """Tout en texte : le CSV brut n'est pas encore type (couche bronze)."""
    return pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig", keep_default_na=False)


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


def _date_souscription_counts(df: pd.DataFrame) -> pd.Series:
    """Les trois formats de date du CSV sont ramenes a une date, agregee par mois.

    Le jour est trop fin pour etre lisible (plus de 1000 dates distinctes).
    """
    brut = df[DATE_SPEC.column].astype("string").str.strip()
    dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for format_source in FORMATS_DATE:
        restantes = brut.where(dates.isna())
        dates = dates.fillna(
            pd.to_datetime(restantes, format=format_source, errors="coerce")
        )

    non_converties = int(dates.isna().sum())
    if non_converties:
        print(f"  WARNING : {non_converties} dates non reconnues, ignorees")

    return dates.dt.strftime("%Y-%m").value_counts().sort_index()


def _valeurs_numeriques(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    """Le CSV melange virgule decimale et unites ("3.1 h", "280.62 €", "40,0")."""
    brut = df[spec.column].astype("string").str.strip()
    nettoye = brut.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(
        nettoye.str.replace(",", ".", regex=False), errors="coerce"
    ).dropna()


def _counts_numeriques(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    """Valeurs entieres (anciennete, note) : tri par valeur croissante, pas par frequence."""
    return _valeurs_numeriques(df, spec).astype(int).value_counts().sort_index()


def _output_path(column: str) -> Path:
    jour = datetime.now().astimezone().date().isoformat()
    return GRAPHS_DIR / f"{jour}_{column}.png"


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

    return _exporter(figure, spec, show=show)


def _exporter(figure: plt.Figure, spec: ChartSpec, *, show: bool) -> Path:
    figure.tight_layout()

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(spec.column)
    figure.savefig(output_path, dpi=120)

    if show:
        plt.show()
    plt.close(figure)

    return output_path


def _render_histogramme(valeurs: pd.Series, spec: ChartSpec, *, show: bool) -> Path:
    """Variable continue : un histogramme, pas une barre par valeur distincte."""
    figure, axes = plt.subplots(figsize=(10, 5))

    if spec.log_x:
        bins = np.logspace(np.log10(valeurs.min()), np.log10(valeurs.max()), 30)
        axes.set_xscale("log")
    else:
        bins = 30

    axes.hist(valeurs.to_numpy(), bins=bins, color="#4c72b0", edgecolor="white")
    axes.set_title(spec.title)
    axes.set_xlabel(spec.xlabel)
    axes.set_ylabel("Nombre de clients")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)

    return _exporter(figure, spec, show=show)


def plot_donnees_clients(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere les six graphiques et retourne les chemins des PNG exportes."""
    df = load_dataset()
    if echo:
        print(f"{CSV_PATH.name} : {len(df)} lignes lues")

    paths = [_render(_date_souscription_counts(df), DATE_SPEC, show=show)]
    paths += [_render(_counts(df, spec), spec, show=show) for spec in CHARTS]
    paths.append(
        _render(_counts_numeriques(df, ANCIENNETE_SPEC), ANCIENNETE_SPEC, show=show)
    )
    paths.append(_render(_counts_numeriques(df, CSAT_SPEC), CSAT_SPEC, show=show))
    for spec in (RETARDS_SPEC, INTEGRATIONS_SPEC):
        paths.append(_render(_counts_numeriques(df, spec), spec, show=show))

    for spec in (DELAI_SPEC, HEURES_USAGE_SPEC, REVENU_SPEC):
        paths.append(
            _render_histogramme(_valeurs_numeriques(df, spec), spec, show=show)
        )

    if echo:
        for path in paths:
            print(f"  export : {path.relative_to(PROJECT_ROOT)}")

    return paths


if __name__ == "__main__":
    plot_donnees_clients(show=False)
