"""Genere tous les graphiques : histogrammes et boites a moustaches.

Un script par type de graphique ; celui-ci les enchaine.

Usage :
    uv run python src/visualization/scripts/plot_all.py
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from visualization.scripts.common import GRAPHS_DIR
from visualization.scripts.plot_boxplots import plot_boxplots
from visualization.scripts.plot_histograms import plot_histograms

Generation = Callable[..., list[Path]]

GENERATIONS: tuple[tuple[str, Generation], ...] = (
    ("HISTOGRAMMES", plot_histograms),
    ("BOITES A MOUSTACHES", plot_boxplots),
)


def _vider_graphs(*, echo: bool) -> None:
    """Repart d'un dossier vide : aucun graphique obsolete ne subsiste.

    Garde-fou : on ne supprime que le dossier de sortie attendu, jamais un
    chemin qui aurait ete redefini ailleurs.
    """
    assert GRAPHS_DIR.parts[-2:] == ("visualization", "graphs"), GRAPHS_DIR

    if not GRAPHS_DIR.exists():
        return

    anciens = len(list(GRAPHS_DIR.rglob("*.png")))
    shutil.rmtree(GRAPHS_DIR)

    if echo:
        print(f"{GRAPHS_DIR.name}/ vide : {anciens} graphiques supprimes")


def plot_all(*, show: bool = True, echo: bool = True) -> list[Path]:
    """Enchaine les generations. Retourne tous les PNG exportes.

    Le dossier de sortie est vide au prealable.
    """
    _vider_graphs(echo=echo)

    paths: list[Path] = []

    for titre, generation in GENERATIONS:
        if echo:
            print(f"\n{'=' * 60}\n{titre}\n{'=' * 60}")
        paths += generation(show=show, echo=echo)

    if echo:
        print(f"\n{len(paths)} graphiques generes")

    return paths


if __name__ == "__main__":
    plot_all(show=False)
