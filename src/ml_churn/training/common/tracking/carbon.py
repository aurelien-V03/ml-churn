"""Mesure de l'empreinte carbone des entrainements, via codecarbon.

codecarbon echantillonne la puissance appelee par le processeur et la memoire
pendant le bloc mesure, en deduit l'energie consommee, puis la convertit en
CO2 equivalent avec l'intensite carbone du reseau electrique du pays detecte,
et en eau consommee avec le facteur `WUE` ci-dessous.

Les valeurs sont exprimees en milligrammes et milliwattheures : une recherche
d'hyperparametres sur 3000 lignes consomme de l'ordre du wattheure, qu'un
affichage en kg et kWh reduirait a des zeros.

La mesure ne vaut que pour un bloc nettement plus long que l'intervalle
d'echantillonnage : l'energie d'une tache est calculee depuis le dernier
releve, de sorte qu'un `fit` de quelques centiemes de seconde se verrait
imputer l'inactivite qui le precede. D'ou la mesure d'une recherche complete
plutot que de chaque entrainement.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from codecarbon import EmissionsTracker

# codecarbon signale a chaque demarrage qu'il tolere plusieurs instances.
logging.getLogger("codecarbon").setLevel(logging.ERROR)

# Intervalle d'echantillonnage de la puissance, en secondes. Le defaut (15 s)
# depasse la duree d'un entrainement : sans cela, aucune mesure ne serait prise.
INTERVALLE = 1

# WUE (Water Usage Effectiveness) : litres d'eau consommes par kWh d'electricite.
# codecarbon ne devine pas cette valeur -- son defaut est 0, qui rend la
# consommation d'eau nulle par construction. 1.8 L/kWh est l'ordre de grandeur
# couramment retenu pour un centre de donnees ; a ajuster selon l'endroit ou
# tourne l'entrainement, la valeur n'etant pas mesuree mais postulee.
WUE = 1.8

_tracker: EmissionsTracker | None = None


def _instance() -> EmissionsTracker | None:
    """Un seul tracker par processus : sa construction detecte le materiel.

    `None` si codecarbon ne peut pas s'initialiser -- materiel inconnu, pas de
    reseau pour geolocaliser. La mesure est alors abandonnee, jamais fatale :
    l'empreinte carbone ne doit pas empecher un entrainement.
    """
    global _tracker

    if _tracker is None:
        try:
            _tracker = EmissionsTracker(
                save_to_file=False,
                log_level="error",
                measure_power_secs=INTERVALLE,
                wue=WUE,
            )
        except Exception:  # noqa: BLE001 - codecarbon remonte des erreurs variees
            return None

    return _tracker


@contextmanager
def track_emissions(nom: str) -> Iterator[dict[str, float]]:
    """Mesure l'empreinte du bloc et remplit le dictionnaire cede.

    Le dictionnaire n'est renseigne qu'a la sortie du bloc : il se lit apres,
    pas dedans.
    """
    mesure: dict[str, float] = {}
    tracker = _instance()

    if tracker is None:
        yield mesure
        return

    tracker.start_task(nom)
    try:
        yield mesure
    finally:
        donnees = tracker.stop_task()
        mesure.update(
            {
                "co2_mg": donnees.emissions * 1e6,
                "energy_mwh": donnees.energy_consumed * 1e6,
                "water_ml": donnees.water_consumed * 1e3,
                "duration_s": donnees.duration,
            }
        )
