# Runway Look Finder

Eine kleine Python-Anwendung, mit der du ein Foto hochladen und den passendsten Look aus einer Runway-Kollektion finden kannst.

Die Anwendung berechnet für jedes Bild ein **CLIP-Embedding** und vergleicht es über **Cosine Similarity** mit den Embeddings einer Referenz-Kollektion. Das Ergebnis: die visuell ähnlichsten Runway-Looks.

> **Hinweis:** Die eigentlichen Referenzbilder sind **nicht** Teil dieses Repositories. Sie unterliegen dem Urheberrecht von Vogue und würden das Repository unnötig aufblähen. Stattdessen kannst du sie über ein mitgeliefertes Skript selbst herunterladen (siehe unten).

---

## Inhaltsverzeichnis

- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Referenzdaten erzeugen](#referenzdaten-erzeugen)
- [Streamlit-App starten](#streamlit-app-starten)
- [Technischer Überblick](#technischer-überblick)
- [Projektstruktur](#projektstruktur)
- [Hinweise zur Nutzung](#hinweise-zur-nutzung)

---

## Voraussetzungen

- Python 3.9 oder höher
- Eine lokale Kopie der **VogueRunway.parquet**-Datei im Projektverzeichnis (wird für den Download der Referenzbilder benötigt)

> Die Parquet-Datei ist ebenfalls **nicht** im Repository enthalten. Du musst sie separat beschaffen und in das Projektverzeichnis legen.

---

## Installation

1. Repository klonen:

   ```bash
   git clone <repository-url>
   cd Runway-Scanner
   ```

2. Virtuelle Umgebung erstellen (empfohlen):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # .venv\Scripts\activate    # Windows
   ```

3. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

---

## Referenzdaten erzeugen

Die Referenzbilder und deren Embeddings werden in drei Schritten erzeugt:

### Schritt 1: Metadaten erkunden (optional)

Prüfe, welche Designer, Saisons und Jahre in der Parquet-Datei verfügbar sind:

```bash
python3 explore_metadata.py
```

### Schritt 2: Referenzbilder herunterladen

Das Skript filtert den Datensatz nach Designer, Saison und Jahr, lädt die passenden Bilder herunter und erzeugt eine `input_data.json`:

```bash
python3 download_reference_images.py
```

Standardmäßig werden **Gucci / Fall / 2023**-Looks heruntergeladen. Du kannst die Werte `target_designer`, `target_season` und `target_year` direkt im Skript anpassen.

### Schritt 3: Embeddings berechnen

Berechne für alle heruntergeladenen Referenzbilder die CLIP-Embeddings und erstelle den `catalog.json`:

```bash
python3 generate_embedding.py
```

Danach liegen vor:

- `embeddings/` – ein `.npy`-Vektor pro Bild
- `catalog.json` – Metadaten und Pfade zu allen referenzierten Looks

---

## Streamlit-App starten

Sobald der Katalog existiert, startest du die Weboberfläche mit:

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser (standardmäßig unter `http://localhost:8501`).

Lade ein Foto hoch – die App zeigt dir die drei ähnlichsten Runway-Looks aus dem Katalog an.

---

## Technischer Überblick

### Pipeline

1. **Bild-Upload**  
   Der Nutzer lädt ein Foto über die Streamlit-Oberfläche hoch.

2. **CLIP-Embedding**  
   Das Bild wird mit dem Modell `ViT-B-32-quickgelu` (Gewichte `openai`) in einen 512-dimensionalen Vektor umgewandelt. Alle Embeddings werden auf Länge 1 normalisiert.

3. **Ähnlichkeitsvergleich**  
   Für jeden Katalog-Eintrag wird die gespeicherte `.npy`-Embedding-Datei geladen. Die Ähnlichkeit ergibt sich aus dem **Skalarprodukt** der normalisierten Vektoren – das ist mathematisch identisch mit der **Cosine Similarity**.

4. **Ranking**  
   Die Ergebnisse werden absteigend nach Ähnlichkeit sortiert und die Top-3-Treffer angezeigt.

### Verwendete Bibliotheken

- [open_clip_torch](https://github.com/mlfoundations/open_clip) – CLIP-Modell und Vorverarbeitung
- [torch](https://pytorch.org/) – Deep-Learning-Backend
- [numpy](https://numpy.org/) – Vektoroperationen
- [Pillow](https://python-pillow.org/) – Bildverarbeitung
- [streamlit](https://streamlit.io/) – Weboberfläche
- [pandas](https://pandas.pydata.org/) & [pyarrow](https://arrow.apache.org/docs/python/) – Einlesen der Parquet-Metadaten
- [requests](https://requests.readthedocs.io/) – Herunterladen der Referenzbilder

---

## Projektstruktur

```text
Runway-Scanner/
├── app.py                      # Streamlit-Weboberfläche
├── clip_utils.py               # CLIP-Modell laden + Embedding berechnen
├── download_reference_images.py # Filtert Parquet-Daten und lädt Bilder
├── explore_metadata.py         # Übersicht über die Parquet-Metadaten
├── find_similar.py             # Ähnlichkeitssuche gegen den Katalog
├── generate_embedding.py       # Erzeugt Embeddings und catalog.json
├── requirements.txt            # Python-Abhängigkeiten
├── .gitignore                  # Ausschluss generierter Dateien
├── README.md                   # Diese Datei
├── VogueRunway.parquet         # Extern bereitzustellen (nicht im Repo)
├── input_data.json             # Wird generiert
├── catalog.json                # Wird generiert
├── embeddings/                 # Wird generiert
├── reference_images/           # Wird heruntergeladen
└── query_images/               # Nutzer-Uploads
```

---

## Hinweise zur Nutzung

- Alle Pfade im Code sind **relativ** zum Projektverzeichnis angelegt, sodass das Projekt nach dem Klonen direkt lauffähig ist.
- Es werden **keine API-Schlüssel oder Passwörter** benötigt; alles läuft lokal.
- Die heruntergeladenen Bilder dienen ausschließlich privaten, nicht-kommerziellen Zwecken. Beachte die Nutzungsbedingungen und Urheberrechte von Vogue.
