"""Briques partagees par les scripts de visualisation.

Chaque type de graphique a son script (`plot_histograms.py` pour les
histogrammes, `plot_boxplots.py` pour les boites a moustaches) ; la lecture des
donnees et l'export des figures sont communs.

Les graphiques sont construits sur la couche bronze, ou le CSV est stocke tel
quel : memes valeurs brutes qu'a l'origine, mais lues depuis la base plutot que
du fichier. L'ingestion bronze doit donc avoir tourne.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml_churn.ingestion.db import get_engine
from ml_churn.ingestion.models.bronze import ChurnSaasCompletBronze

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_DIR = PROJECT_ROOT / "src" / "visualization" / "graphs"

# Table lue par tous les scripts, et libelle affiche dans leurs logs.
SOURCE = ChurnSaasCompletBronze.__table__.fullname

# Colonnes ajoutees par l'ingestion : sans interet pour l'exploration.
COLONNES_TECHNIQUES = ("id", "_source_file", "_source_line", "_ingested_at")


def load_dataset() -> pd.DataFrame:
    """Couche bronze : le CSV stocke tel quel, tout en texte.

    Les valeurs manquantes reviennent en `None` depuis la base ; elles sont
    ramenees a la chaine vide, comme lors de la lecture du CSV, pour que les
    comptages par modalite restent inchanges.
    """
    df = pd.read_sql(f"select * from {SOURCE} order by _source_line", get_engine())
    return df.drop(columns=list(COLONNES_TECHNIQUES)).fillna("")


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Le CSV melange virgule decimale et unites ("3.1 h", "280.62 €", "40,0")."""
    raw = df[column].astype("string").str.strip()
    cleaned = raw.str.replace(r"[^0-9,.+-]", "", regex=True)
    return pd.to_numeric(
        cleaned.str.replace(",", ".", regex=False), errors="coerce"
    ).dropna()


def export_figure(
    figure: plt.Figure, folder: str, filename: str, *, show: bool
) -> Path:
    """graphs/<folder>/<filename>.png

    Les graphiques d'une seule colonne vont dans le dossier de cette colonne ;
    ceux qui en croisent deux ont leur propre dossier par type.
    """
    figure.tight_layout()

    path = GRAPHS_DIR / folder / f"{filename}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)

    if show:
        plt.show()
    plt.close(figure)

    return path
