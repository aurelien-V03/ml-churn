"""Formats de log communs aux trois couches du medaillon."""

from __future__ import annotations

# Prefixes de meme largeur : les noms de fichiers et de tables s'alignent en
# colonne, un ecart entre lu et ecrit se voit alors d'un coup d'oeil.
PREFIXE_LECTURE_CSV = "[LECTURE CSV]        "
PREFIXE_LECTURE_BASE = "[LECTURE POSTGRESQL] "
PREFIXE_ECRITURE_BASE = "[ECRITURE POSTGRESQL]"


ICONE_SUCCES = "\u2705"
ICONE_ALERTE = "\u26a0\ufe0f"

# Famille de la derniere action loguee, pour aerer entre deux familles.
_groupe_precedent: str | None = None


def _separer_groupe(groupe: str) -> None:
    """Ligne vide au passage d'une famille d'actions a une autre."""
    global _groupe_precedent
    if _groupe_precedent is not None and groupe != _groupe_precedent:
        print()
    _groupe_precedent = groupe


def reinitialiser_groupes() -> None:
    """Repart d'un etat neutre : deux executions successives restent lisibles."""
    global _groupe_precedent
    _groupe_precedent = None


def pourcentage(part: float) -> str:
    """Part arrondie au centieme, sans decimale inutile."""
    return f"{part:.2f}".rstrip("0").rstrip(".") + "%"


def part_couverte(traitees: int, total: int) -> str:
    """Couverture de l'action : checkmark si elle a couvert toutes les valeurs."""
    part = 100.0 if total == 0 else traitees / total * 100
    icone = ICONE_SUCCES if traitees == total else ICONE_ALERTE
    return f"{icone} {pourcentage(part)}"


def log_action(
    action: str,
    traitees: int,
    total: int,
    unite: str,
    *,
    resultat: str,
    description: str = "",
    detail: str = "",
) -> None:
    """Format commun a toutes les actions de transformation, silver et gold.

    `[ACTION] comment l'action procede : icone part des valeurs <resultat>`. La
    description evite d'avoir a ouvrir le script pour savoir ce qu'une ligne du
    log represente, la part dit si l'action a couvert tout ce qu'elle visait.
    """
    _separer_groupe(action.split()[0])
    quoi = f" {description}" if description else ""
    couverture = f"{part_couverte(traitees, total)} des {unite} {resultat}"
    print(f"[{action.upper()}]{quoi} : {couverture}{detail}")


def log_total(lignes_par_table: dict[str, int], titre: str = "TOTAL") -> None:
    """Bilan de fin d'ingestion, identique pour bronze, silver et gold.

    `titre` precise ce qui est totalise quand la couche lit une source et en
    ecrit une autre : l'ingestion bronze compte des lignes de CSV d'un cote et
    des lignes de table de l'autre.
    """
    print(f"\n{titre} :")
    for table, lignes in lignes_par_table.items():
        print(f"      - {table} : {lignes} lignes")
