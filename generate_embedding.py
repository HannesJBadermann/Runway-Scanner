"""
generate_embedding.py
=====================
Baustein eines Fashion-Matching-Tools:
Liest eine Liste von Bildern aus input_data.json, berechnet für jedes
Bild ein CLIP-Embedding und speichert Vektor sowie Metadaten ab.
Später können Nutzerfotos mit diesen Referenz-Embeddings verglichen
werden.
"""

import os
import json
import numpy as np

from clip_utils import load_clip_model, get_image_embedding

# ---------------------------------------------------------------------------
# 1. Konfiguration.
# ---------------------------------------------------------------------------
INPUT_FILE = "input_data.json"   # JSON-Datei mit den zu verarbeitenden Bildern.
CATALOG_FILE = "catalog.json"    # Ausgabe-Datei mit allen Metadaten.

# ---------------------------------------------------------------------------
# 2. Ausgabe-Ordner anlegen, falls sie noch nicht existieren.
# ---------------------------------------------------------------------------
# os.makedirs erstellt fehlende Ordner und macht nichts, wenn sie schon da sind
# (dank exist_ok=True).
# Der Eingabe-Ordner für die Bilder wird NICHT hier angelegt; der Pfad kommt
# aus "image_path" in input_data.json und kann beliebig sein.
os.makedirs("embeddings", exist_ok=True)

# ---------------------------------------------------------------------------
# 3. CLIP-Modell und Bildvorverarbeitung laden.
# ---------------------------------------------------------------------------
# Die Logik dafür liegt in clip_utils.py, damit generate_embedding.py und
# find_similar.py dasselbe Modell auf dieselbe Weise laden.
model, preprocess = load_clip_model(device="cpu")

# ---------------------------------------------------------------------------
# 4. Eingabedaten aus input_data.json laden.
# ---------------------------------------------------------------------------
# Diese Datei enthält eine Liste von Objekten. Jedes Objekt beschreibt ein
# Bild mit Pfad und Metadaten (Designer, Kollektion, Saison, Look-Nummer).
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    input_data = json.load(f)

print(f"{len(input_data)} Einträge aus {INPUT_FILE} geladen.")

# ---------------------------------------------------------------------------
# 5. Bestehenden Katalog laden (oder leere Liste anlegen).
# ---------------------------------------------------------------------------
# catalog.json ist die zentrale Ausgabe-Datei. Sie wird nur erweitert oder
# aktualisiert, niemals komplett neu geschrieben, damit vorhandene Einträge
# erhalten bleiben.
if os.path.exists(CATALOG_FILE):
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)
else:
    catalog = []

# Hilfsliste, um schnell zu prüfen, ob eine ID schon existiert.
existing_ids = [item["id"] for item in catalog]

# ---------------------------------------------------------------------------
# 6. Jedes Bild in der Eingabeliste verarbeiten.
# ---------------------------------------------------------------------------
for item in input_data:
    image_path = item["image_path"]
    image_filename = os.path.basename(image_path)

    # ID automatisch aus dem Dateinamen ableiten (ohne Dateiendung).
    look_id = os.path.splitext(image_filename)[0]

    print(f"Verarbeite {image_filename} ... ", end="", flush=True)

    # Fehlerbehandlung: Falls das Bild nicht existiert, überspringen.
    if not os.path.exists(image_path):
        print("übersprungen (Datei nicht gefunden)")
        print(f"  WARNUNG: {image_path} existiert nicht.")
        continue

    # -----------------------------------------------------------------------
    # 6.1 Embedding über die gemeinsame Hilfsfunktion berechnen.
    # -----------------------------------------------------------------------
    # clip_utils.get_image_embedding kümmert sich um Bildladen,
    # Vorverarbeitung, Modellvorhersage und Normalisierung.
    embedding_np = get_image_embedding(image_path, model, preprocess)

    # -----------------------------------------------------------------------
    # 6.3 Embedding als NumPy-Array speichern.
    # -----------------------------------------------------------------------
    embedding_path = f"embeddings/{look_id}.npy"
    np.save(embedding_path, embedding_np)

    # -----------------------------------------------------------------------
    # 6.4 Katalog-Eintrag zusammenbauen.
    # -----------------------------------------------------------------------
    entry = {
        "id": look_id,
        "image_path": image_path,
        "embedding_path": embedding_path,
        "designer": item["designer"],
        "collection": item["collection"],
        "season": item["season"],
        "look_number": item["look_number"],
    }

    # Falls ein Eintrag mit derselben ID schon existiert, wird er überschrieben,
    # ansonsten wird der neue Eintrag hinten angehängt.
    if look_id in existing_ids:
        index = existing_ids.index(look_id)
        catalog[index] = entry
    else:
        catalog.append(entry)
        existing_ids.append(look_id)

    print("fertig")

# ---------------------------------------------------------------------------
# 7. Katalog zurückschreiben.
# ---------------------------------------------------------------------------
# Nachdem alle Bilder verarbeitet wurden, speichern wir den gesamten Katalog
# einmal mit schöner Formatierung.
with open(CATALOG_FILE, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print(f"\nKatalog gespeichert unter: {CATALOG_FILE}")
