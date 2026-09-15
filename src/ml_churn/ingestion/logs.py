"""Formats de log communs aux trois couches du medaillon."""

from __future__ import annotations


def log_total(lignes_par_table: dict[str, int]) -> None:
    """Bilan de fin d'ingestion, identique pour bronze, silver et gold."""
    print("\nTOTAL :")
    for table, lignes in lignes_par_table.items():
        print(f"      - {table} : {lignes} lignes")
