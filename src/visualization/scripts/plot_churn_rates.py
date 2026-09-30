"""Taux de churn par tranche d'une variable.

Le bon rendu face a une cible binaire : un boxplot groupe par churn donne deux
boites qui se chevauchent, alors que le taux de churn par tranche montre a
partir de quel seuil le risque decolle -- une information directement
actionnable.

Usage :
    uv run python src/visualization/scripts/plot_churn_rates.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_churn.ingestion.scripts.ingest_gold import NIVEAUX_ANCIENNETE
from visualization.scripts.common import (
    SOURCE,
    annee_souscription,
    dates_souscription,
    export_figure,
    load_dataset,
    mois_souscription,
)

# Ce graphique croise une colonne avec la cible : dossier propre au type.
FOLDER = "churn_rate"

# Ce qui est mesure par tranche : prefixe de fichier, titre et axe des ordonnees.
# Le taux dit ou le risque est eleve, le volume dit ou les pertes se concentrent.
MESURES: dict[str, tuple[str, str, str]] = {
    "taux": ("churnrate", "Taux de churn par tranche", "Taux de churn (%)"),
    "volume": ("churnvolume", "Volume de churn par tranche", "Nombre de clients"),
    "taux_glissant": (
        "churnrate_glissant",
        "Taux de churn par niveau d'anciennete, mois par mois",
        "Taux de churn (%)",
    ),
}

# Une couleur par niveau d'anciennete, du plus installe au plus fragile.
COULEURS_NIVEAU: dict[str, str] = {
    "ANCIEN": "#4c72b0",
    "ETABLI": "#55a868",
    "RECENT": "#8172b3",
}

CIBLE = "churn"

# Au-dela, les etiquettes par point se chevauchent : on ne garde que la courbe.
MAX_ETIQUETTES = 8

# En dessous, une tranche ne porte pas un taux exploitable : un seul client
# donne 0 % ou 100 %, ce qui ecrase l'echelle sans rien apprendre.
EFFECTIF_MINIMUM = 10


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
    # Variable a calculer : la fonction rend directement les tranches.
    deriver: Callable[[pd.DataFrame], pd.Categorical] | None = None
    # Unite de la pente, pour la legende de la droite de tendance.
    unite_tendance: str = "tranche"
    # Mesure portee par l'axe des ordonnees : cle de MESURES.
    mesure: str = "taux"

    @property
    def filename(self) -> str:
        prefixe, _titre, _ordonnees = MESURES[self.mesure]
        return f"{prefixe}_{self.column}"


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
        column="date_souscription",
        label="Annee de souscription",
        # Les cohortes recentes ont moins d'anciennete : le taux monte avec
        # l'annee, mais le volume se concentre sur l'annee la plus peuplee.
        deriver=annee_souscription,
        mesure="volume",
    ),
    ChurnRateSpec(
        column="date_souscription",
        label="Mois de souscription",
        # Une tranche par mois calendaire. Les cohortes recentes ont moins
        # d'anciennete : la hausse suit l'exposition, pas une degradation.
        deriver=mois_souscription,
        unite_tendance="mois",
    ),
    ChurnRateSpec(
        column="date_souscription",
        label="Mois d'observation",
        # Pas de `deriver` : les fenetres glissantes sont construites a partir
        # des dates de souscription, mois d'observation par mois d'observation.
        mesure="taux_glissant",
    ),
    ChurnRateSpec(
        column="csat",
        label="Note CSAT",
        # Note entiere de 1 a 5 : une tranche par note, sans regroupement.
        bornes=(0, 1, 2, 3, 4, 5),
        etiquettes=("1", "2", "3", "4", "5"),
        unite_tendance="point",
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
    if spec.deriver is not None:
        tranches = spec.deriver(df)
    elif spec.ordre:
        tranches = _modalites(df, spec)
    else:
        tranches = pd.cut(
            _numeric(df, spec.column),
            bins=list(spec.bornes),
            labels=list(spec.etiquettes),
        )

    groupes = pd.DataFrame({"tranche": tranches, "churn": _numeric(df, CIBLE)}).dropna()

    stats = groupes.groupby("tranche", observed=True)["churn"].agg(
        clients="size", taux="mean", resiliations="sum"
    )

    return stats[stats["clients"] >= EFFECTIF_MINIMUM]


def _bornes_niveaux() -> list[tuple[float, float, str]]:
    """(borne basse, borne haute, libelle) pour chaque niveau, tel que defini en gold."""
    hautes = [borne for borne, _libelle in NIVEAUX_ANCIENNETE]
    basses = [0.0, *hautes[:-1]]

    return [
        (bas, haut, libelle)
        for bas, haut, (_borne, libelle) in zip(
            basses, hautes, NIVEAUX_ANCIENNETE, strict=True
        )
    ]


def _taux_glissant_par_niveau(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pour chaque mois d'observation, le taux de churn de chaque niveau.

    A la date M, un client souscrit au mois S a `M - S` mois d'anciennete : il
    tombe donc dans l'un des trois niveaux. Les fenetres glissent avec M, si
    bien que les trois series couvrent tout l'axe au lieu de se partager en
    trois segments disjoints.
    """
    donnees = pd.DataFrame(
        {
            "mois": dates_souscription(df).dt.to_period("M"),
            "churn": _numeric(df, CIBLE),
        }
    ).dropna()

    souscriptions = pd.PeriodIndex(donnees["mois"]).asi8
    churn = donnees["churn"].to_numpy()
    calendrier = pd.period_range(donnees["mois"].min(), donnees["mois"].max(), freq="M")

    series: dict[str, pd.DataFrame] = {}
    for bas, haut, libelle in _bornes_niveaux():
        lignes = []
        for observation in calendrier:
            anciennete = observation.ordinal - souscriptions
            fenetre = (anciennete > bas) & (anciennete <= haut)
            effectif = int(fenetre.sum())

            if effectif >= EFFECTIF_MINIMUM:
                lignes.append(
                    {
                        "mois": str(observation),
                        "clients": effectif,
                        "taux": float(churn[fenetre].mean()),
                    }
                )

        series[libelle] = pd.DataFrame(lignes).set_index("mois")

    return series


def _render_glissant(
    series: dict[str, pd.DataFrame], spec: ChurnRateSpec, taux_global: float
) -> plt.Figure:
    """Une courbe par niveau, sur un axe commun a toutes les series."""
    calendrier = sorted({mois for stats in series.values() for mois in stats.index})
    rang = {mois: position for position, mois in enumerate(calendrier)}

    figure, axes = plt.subplots(figsize=(11, 5.5))

    for libelle, stats in series.items():
        if stats.empty:
            continue

        positions = [rang[mois] for mois in stats.index]
        axes.plot(
            positions,
            stats["taux"] * 100,
            color=COULEURS_NIVEAU.get(libelle, "#999999"),
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
            label=f"{libelle} ({stats['clients'].iloc[-1]} clients au dernier mois)",
        )

    axes.axhline(
        taux_global * 100,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label=f"taux global ({taux_global:.1%})",
    )

    axes.set_xticks(range(len(calendrier)))
    axes.set_xticklabels(calendrier)
    axes.set_ylim(0, 100)
    axes.legend()

    return figure


def _render_volume(
    stats: pd.DataFrame, spec: ChurnRateSpec, taux_global: float
) -> plt.Figure:
    """Nombre de clients resilies par tranche, effectif de la tranche en repere."""
    figure, axes = plt.subplots(figsize=(10, 5.5))
    resiliations = stats["resiliations"]

    axes.bar(
        stats.index.astype(str),
        stats["clients"],
        color="#dddddd",
        zorder=2,
        label="clients de la tranche",
    )
    axes.bar(
        stats.index.astype(str),
        resiliations,
        color="#c44e52",
        zorder=3,
        label="clients resilies",
    )

    for tranche, ligne in stats.iterrows():
        axes.annotate(
            f"{int(ligne['resiliations'])} / {int(ligne['clients'])}\n"
            f"({ligne['taux']:.1%})",
            (str(tranche), ligne["resiliations"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    axes.set_ylim(0, stats["clients"].max() * 1.15)
    axes.legend()

    return figure


def _tendance(stats: pd.DataFrame, positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Droite ponderee par l'effectif : les tranches peuplees pesent davantage."""
    pente, ordonnee = np.polyfit(
        positions, stats["taux"] * 100, deg=1, w=stats["clients"]
    )

    return pente, pente * positions + ordonnee


def _render_points(
    stats: pd.DataFrame, spec: ChurnRateSpec, taux_global: float
) -> plt.Figure:
    """Nuage de points : la taille porte l'effectif, la droite porte la tendance."""
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
    pente, valeurs = _tendance(stats, np.arange(len(stats)))
    axes.plot(
        stats.index.astype(str),
        valeurs,
        color="#55a868",
        linewidth=2,
        zorder=2,
        label=f"tendance ({pente:+.1f} pt par {spec.unite_tendance})",
    )
    axes.axhline(
        taux_global * 100,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label=f"taux global ({taux_global:.1%})",
    )

    # L'effectif conditionne la fiabilite de chaque point : on l'affiche, tant
    # que les etiquettes ne se chevauchent pas.
    if len(stats) <= MAX_ETIQUETTES:
        for tranche, ligne in stats.iterrows():
            axes.annotate(
                f"{ligne['taux']:.1%}\n(n={int(ligne['clients'])})",
                (str(tranche), ligne["taux"] * 100),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                fontsize=9,
            )

    # Un taux ne depasse pas 100 % : la marge haute ne doit pas le laisser croire.
    axes.set_ylim(0, min(stats["taux"].max() * 100 * 1.35, 100))
    axes.legend()

    return figure


def _render(df: pd.DataFrame, spec: ChurnRateSpec, *, show: bool) -> Path:
    taux_global = _numeric(df, CIBLE).mean()
    _prefixe, titre, ordonnees = MESURES[spec.mesure]

    if spec.mesure == "taux_glissant":
        series = _taux_glissant_par_niveau(df)
        figure = _render_glissant(series, spec, taux_global)
        nombre_de_tranches = max(len(stats) for stats in series.values())
    else:
        stats = _taux_par_tranche(df, spec)
        dessiner = _render_volume if spec.mesure == "volume" else _render_points
        figure = dessiner(stats, spec, taux_global)
        nombre_de_tranches = len(stats)

    axes = figure.axes[0]
    axes.set_title(f"{titre} — {spec.label}")
    axes.set_xlabel(spec.label)
    axes.set_ylabel(ordonnees)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.3)
    if nombre_de_tranches > MAX_ETIQUETTES:
        axes.tick_params(axis="x", labelsize=8, rotation=90)

    return export_figure(figure, FOLDER, spec.filename, show=show)


def plot_churn_rates(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Genere un graphique de taux de churn par variable. Retourne les PNG."""
    df = load_dataset()
    if echo:
        print(f"{SOURCE} : {len(df)} lignes lues")

    paths: list[Path] = []
    for spec in CHURN_RATES:
        paths.append(_render(df, spec, show=show))

        if echo:
            if spec.mesure == "taux_glissant":
                for libelle, stats in _taux_glissant_par_niveau(df).items():
                    detail = " ".join(
                        f"{mois}:{ligne['taux']:.1%}"
                        for mois, ligne in stats.iloc[::6].iterrows()
                    )
                    print(f"  {libelle:<8} {spec.mesure:<15} {detail}")
                continue

            stats = _taux_par_tranche(df, spec)
            if spec.mesure == "volume":
                detail = " ".join(
                    f"{tranche}:{int(ligne['resiliations'])}(n={int(ligne['clients'])})"
                    for tranche, ligne in stats.iterrows()
                )
            else:
                detail = " ".join(
                    f"{tranche}:{ligne['taux']:.1%}(n={int(ligne['clients'])})"
                    for tranche, ligne in stats.iterrows()
                )
            print(f"  {spec.column:<26} {spec.mesure:<7} {detail}")

    return paths


if __name__ == "__main__":
    plot_churn_rates(show=False)
