#!/usr/bin/env python3
"""
explore_metadata.py

Liest die Vogue-Runway-Metadaten-Datei "VogueRunway.parquet" ein und gibt
einen übersichtlichen Überblick über Inhalt und Struktur aus.
"""

import pandas as pd
from pathlib import Path


def main() -> None:
    # Pfad zur Parquet-Datei (erwartet im selben Verzeichnis wie das Skript)
    parquet_path = Path("VogueRunway.parquet")

    # Prüfen, ob die Datei existiert, damit wir eine verständliche Fehlermeldung ausgeben können
    if not parquet_path.is_file():
        print(f"Fehler: Datei nicht gefunden: {parquet_path.resolve()}")
        print("Bitte lege 'VogueRunway.parquet' in das gleiche Verzeichnis wie dieses Skript.")
        return

    # Parquet-Datei mit pandas und pyarrow als Engine einlesen
    print(f"Lese '{parquet_path}' ein ...\n")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # 1. Anzahl der Zeilen (Gesamtbilder)
    print(f"Gesamtzahl der Bilder (Zeilen): {len(df):,}\n")

    # 2. Spaltennamen und Datentypen
    print("Spalten und Datentypen:")
    print(df.dtypes)
    print()

    # 3. Die ersten 5 Zeilen als Beispiel
    print("Erste 5 Zeilen:")
    print(df.head().to_string())
    print()

    # 4. Eindeutige Designer-Namen (alphabetisch sortiert, maximal 30)
    unique_designers = sorted(df["designer"].dropna().unique())
    print(f"Eindeutige Designer (erste 30 von {len(unique_designers)}):")
    for designer in unique_designers[:30]:
        print(f"  - {designer}")
    print()

    # 5. Eindeutige Werte in der "season"-Spalte (maximal 30)
    unique_seasons = df["season"].dropna().unique()
    print(f"Eindeutige Seasons (erste 30 von {len(unique_seasons)}):")
    for season in unique_seasons[:30]:
        print(f"  - {season}")
    print()

    # Zusätzlich: Kurzer Blick auf die URL-Spalte, um zu beurteilen,
    # ob es sich um vollständige URLs oder nur Pfad-Fragmente handelt.
    print("Beispiele aus der 'url'-Spalte:")
    for idx, url in enumerate(df["url"].dropna().head(5), start=1):
        print(f"  {idx}. {url}")


if __name__ == "__main__":
    main()
