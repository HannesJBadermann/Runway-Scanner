#!/usr/bin/env python3
"""
download_reference_images.py

Filtert den VogueRunway-Datensatz nach Gucci / Fall / 2025, lädt die
zugehörigen Bilder herunter und erzeugt eine input_data.json, die vom
bestehenden generate_embedding.py weiterverarbeitet werden kann.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


def find_similar_designers(df: pd.DataFrame, target: str) -> list[str]:
    """
    Hilfsfunktion: Sucht case-insensitive nach Designer-Namen, die den
    übergebenen Suchbegriff enthalten. Wird genutzt, um dem Nutzer bei
    0 Treffern mögliche Schreibweisen anzuzeigen.
    """
    mask = df["designer"].str.contains(target, case=False, na=False)
    return sorted(df.loc[mask, "designer"].unique())


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Konfiguration
    # ------------------------------------------------------------------
    parquet_path = Path("VogueRunway.parquet")
    output_dir = Path("reference_images/gucci_fall2025")
    input_data_path = Path("input_data.json")
    target_designer = "Gucci"
    target_season = "Fall"
    target_year = 2023
    sample_size = 80
    download_timeout = 10  # Sekunden pro Bild
    sleep_seconds = 0.5    # Pause zwischen Downloads

    # Browser-ähnlicher User-Agent, damit der Server das Skript nicht blockt
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        )
    }

    # ------------------------------------------------------------------
    # 2. Parquet-Datei einlesen
    # ------------------------------------------------------------------
    if not parquet_path.is_file():
        print(f"Fehler: '{parquet_path}' nicht gefunden.")
        return

    print(f"Lese '{parquet_path}' ein ...")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # ------------------------------------------------------------------
    # 3. Datensatz filtern
    # ------------------------------------------------------------------
    filtered = df[
        (df["designer"] == target_designer)
        & (df["season"] == target_season)
        & (df["year"] == target_year)
    ].copy()

    print(
        f"\nFilter: designer='{target_designer}', season='{target_season}', "
        f"year={target_year}"
    )
    print(f"Gefundene Treffer: {len(filtered)}")

    # Falls keine Treffer: Warnung ausgeben und ähnliche Designer-Namen zeigen
    if len(filtered) == 0:
        print("\nWARNUNG: 0 Treffer gefunden.")
        similar = find_similar_designers(df, "gucci")
        if similar:
            print("Mögliche ähnliche Designer-Namen (case-insensitive):")
            for name in similar[:20]:
                print(f"  - {name}")
        else:
            print("Keine ähnlichen Designer-Namen gefunden.")
        return

    # ------------------------------------------------------------------
    # 4. Treffer nach category aufschlüsseln
    # ------------------------------------------------------------------
    print("\nAufteilung nach 'category':")
    category_counts = filtered["category"].value_counts(dropna=False)
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    # ------------------------------------------------------------------
    # 5. Stichprobe ziehen (reproduzierbar)
    # ------------------------------------------------------------------
    if len(filtered) > sample_size:
        selected = filtered.sample(n=sample_size, random_state=42)
        print(f"\nEs wurden mehr als {sample_size} Treffer gefunden.")
        print(f"Ziehe zufällige Stichprobe von {sample_size} Bildern.")
    else:
        selected = filtered
        print(f"\nNehme alle {len(selected)} verfügbaren Bilder.")

    # ------------------------------------------------------------------
    # 6. Ausgabeordner anlegen
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 7. Bilder herunterladen und input_data.json aufbauen
    # ------------------------------------------------------------------
    input_data: list[dict] = []
    successful = 0
    failed = 0
    total_bytes = 0

    # Iteration über die ausgewählten Zeilen
    for _, row in selected.iterrows():
        key = str(row["key"])
        url = row["url"]
        filename = output_dir / f"{key}.jpg"

        print(f"\nLade Bild {key} herunter ...")
        print(f"  URL: {url}")

        try:
            # HTTP-GET mit Timeout und User-Agent-Header
            response = requests.get(url, headers=headers, timeout=download_timeout)
            # Bei Fehler-Statuscodes (z. B. 404) eine Ausnahme auslösen
            response.raise_for_status()

            # Bild auf Festplatte speichern
            with open(filename, "wb") as f:
                f.write(response.content)

            file_size = len(response.content)
            total_bytes += file_size
            successful += 1
            print(f"  Gespeichert: {filename} ({file_size:,} Bytes)")

            # Eintrag für input_data.json erstellen
            input_data.append({
                "image_path": str(filename),
                "designer": target_designer,
                "collection": f"{target_season} {target_year}",
                "season": target_season,
                "look_number": key,
            })

        except requests.exceptions.RequestException as e:
            # Timeouts, Verbindungsfehler, 404 etc. abfangen
            failed += 1
            print(f"  WARNUNG: Download fehlgeschlagen – {e}")

        # Höfliche Pause vor dem nächsten Request
        time.sleep(sleep_seconds)

    # ------------------------------------------------------------------
    # 8. input_data.json schreiben
    # ------------------------------------------------------------------
    with open(input_data_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=2, ensure_ascii=False)

    print(f"\n{input_data_path} mit {len(input_data)} Einträgen erstellt.")

    # ------------------------------------------------------------------
    # 9. Zusammenfassung
    # ------------------------------------------------------------------
    total_mb = total_bytes / (1024 * 1024)
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"Gefilterte Treffer gesamt:    {len(filtered)}")
    print(f"Ausgewählte Bilder:           {len(selected)}")
    print(f"Erfolgreich heruntergeladen:  {successful}")
    print(f"Fehlgeschlagene Downloads:    {failed}")
    print(f"Gesamter Speicherbedarf:      {total_mb:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()
