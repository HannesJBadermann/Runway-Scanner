"""
find_similar.py
===============
Vergleicht ein hochgeladenes Query-Bild mit allen Einträgen im Katalog
und gibt die ähnlichsten Runway-Looks aus.

Das Query-Bild liegt in einem eigenen Ordner (query_images/), getrennt
von den Referenzbildern (reference_images/), weil es konzeptionell etwas
anderes ist: ein Nutzer-Upload statt ein Look aus einer Kollektion.
"""

import os
import json
import numpy as np

from clip_utils import load_clip_model, get_image_embedding


def find_similar_images(query_embedding, catalog, top_k=3):
    """
    Vergleicht ein Query-Embedding mit allen Einträgen eines Katalogs
    und gibt die ähnlichsten Treffer zurück.

    Args:
        query_embedding: NumPy-Array der Form (1, 512) mit dem normalisierten
            Embedding des Query-Bildes.
        catalog: Liste von Katalog-Einträgen (Dictionaries). Jeder Eintrag
            muss mindestens das Feld "embedding_path" enthalten.
        top_k: Anzahl der zurückgegebenen Top-Treffer. Standard ist 3.

    Returns:
        Eine Liste mit den top_k ähnlichsten Treffern. Jedes Element ist ein
        Dictionary mit den Schlüsseln "similarity" (float) und "item"
        (der zugehörige Katalog-Eintrag).
    """
    results = []

    for item in catalog:
        embedding_path = item["embedding_path"]

        # Prüfen, ob die gespeicherte Embedding-Datei existiert.
        if not os.path.exists(embedding_path):
            print(f"WARNUNG: Embedding-Datei nicht gefunden, überspringe: {embedding_path}")
            continue

        # Gespeichertes Embedding laden.
        catalog_embedding = np.load(embedding_path)

        # Kosinus-Ähnlichkeit = Skalarprodukt der normalisierten Vektoren.
        # .item() entpackt das 1x1-Array zu einem Python-Float (vermeidet
        # DeprecationWarning bei neueren NumPy-Versionen).
        similarity = float(np.dot(query_embedding, catalog_embedding.T).item())

        results.append({
            "similarity": similarity,
            "item": item,
        })

    # Absteigend nach Ähnlichkeit sortieren (höchster Wert zuerst).
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Falls weniger Treffer als top_k vorhanden sind, passen wir die Anzahl an.
    return results[:top_k]


# ---------------------------------------------------------------------------
# 1. Konfiguration.
# ---------------------------------------------------------------------------
QUERY_IMAGE_PATH = "query_images/query.png"  # Pfad zum Nutzer-Upload.
CATALOG_FILE = "catalog.json"                # Katalog mit Referenz-Looks.
TOP_K = 3                                    # Anzahl der ausgegebenen Treffer.

# ---------------------------------------------------------------------------
# 2. Query-Ordner anlegen und prüfen, ob ein Bild vorhanden ist.
# ---------------------------------------------------------------------------
os.makedirs("query_images", exist_ok=True)

if not os.path.exists(QUERY_IMAGE_PATH):
    print("Bitte lege ein Vergleichsbild unter query_images/ ab.")
    print(f"Erwarteter Pfad: {QUERY_IMAGE_PATH}")
    exit(0)  # Beendet das Skript ohne Fehler (es fehlt nur das Bild).

# ---------------------------------------------------------------------------
# 3. Katalog laden und prüfen, ob genügend Einträge vorhanden sind.
# ---------------------------------------------------------------------------
if not os.path.exists(CATALOG_FILE):
    print(f"Katalog '{CATALOG_FILE}' nicht gefunden.")
    print("Bitte führe zuerst generate_embedding.py aus, um Referenz-Embeddings zu erstellen.")
    exit(0)

with open(CATALOG_FILE, "r", encoding="utf-8") as f:
    catalog = json.load(f)

if len(catalog) == 0:
    print("Der Katalog enthält noch keine Einträge.")
    print("Bitte führe zuerst generate_embedding.py aus, um Referenz-Embeddings zu erstellen.")
    exit(0)

if len(catalog) < TOP_K:
    print(f"Hinweis: Der Katalog enthält nur {len(catalog)} Einträge.")
    print(f"Es werden daher nur die Top {len(catalog)} angezeigt.")

# ---------------------------------------------------------------------------
# 4. CLIP-Modell laden.
# ---------------------------------------------------------------------------
# Das Modell wird einmal geladen und für Query- und Katalog-Bilder
# wiederverwendet. So stellen wir sicher, dass alle Embeddings mit
# derselben Modellkonfiguration berechnet wurden.
model, preprocess = load_clip_model(device="cpu")

# ---------------------------------------------------------------------------
# 5. Embedding für das Query-Bild berechnen.
# ---------------------------------------------------------------------------
print(f"Berechne Embedding für Query-Bild: {QUERY_IMAGE_PATH}")
query_embedding = get_image_embedding(QUERY_IMAGE_PATH, model, preprocess)

# ---------------------------------------------------------------------------
# 6. Ähnlichkeiten zu allen Katalog-Einträgen berechnen.
# ---------------------------------------------------------------------------
# Die eigentliche Suchlogik wurde in die Funktion find_similar_images()
# ausgelagert, damit sie auch von anderen Skripten (z. B. app.py) genutzt
# werden kann, ohne dupliziert zu werden.
results = find_similar_images(query_embedding, catalog, top_k=TOP_K)

# ---------------------------------------------------------------------------
# 7. Ergebnisse ausgeben.
# ---------------------------------------------------------------------------
print(f"\nTop {len(results)} ähnlichste Looks für '{QUERY_IMAGE_PATH}':\n")

for rank, result in enumerate(results):
    item = result["item"]
    similarity = round(result["similarity"], 3)

    print(f"{rank + 1}. Platz")
    print(f"   Designer:     {item['designer']}")
    print(f"   Kollektion:   {item['collection']}")
    print(f"   Saison:       {item['season']}")
    print(f"   Look-Nummer:  {item['look_number']}")
    print(f"   Similarity:   {similarity}")
    print()
