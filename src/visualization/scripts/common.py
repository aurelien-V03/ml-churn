"""Briques partagees par les scripts de visualisation.

Chaque type de graphique a son script (`plot_histograms.py` pour les
histogrammes, `plot_boxplots.py` pour les boites a moustaches) ; la lecture du
CSV et l'export des figures sont communs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "docs" / "churn_saas_complet.csv"
GRAPHS_DIR = PROJECT_ROOT / "src" / "visualization" / "graphs"


def load_dataset() -> pd.DataFrame:
    """Tout en texte : le CSV brut n'est pas encore type (couche bronze)."""
    return pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig", keep_default_na=False)


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
